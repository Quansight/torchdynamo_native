#include "ATen/core/dispatch/Dispatcher.h"
#include "ATen/ops/add.h"
#include "ATen/ops/tensor.h"
#include "ATen/ops/ones.h"
#include "c10/core/Device.h"
#include "c10/core/DeviceType.h"
#include "c10/core/DispatchKey.h"
#include "c10/core/TensorOptions.h"
#include "c10/util/Exception.h"
#include "pybind11/pybind11.h"
#include "pybind11/pytypes.h"
#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/ExecutionEngine/Orc/ThreadSafeModule.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/Support/TargetSelect.h"

#include <iostream>
#include <iterator>
#include <memory>

#include <torch/csrc/utils/pybind.h>
#include <unordered_map>

void dump_tensor(const at::Tensor &t) { std::cout << t << std::endl; }

at::Tensor one_dim_tensor(int i) { return at::ones({i}); }

namespace {

using namespace llvm;

static orc::LLJIT *get_jit() {
  static std::unique_ptr<orc::LLJIT> jit(nullptr);

  if (jit.get() == nullptr) {
    jit.reset(cantFail(orc::LLJITBuilder().create()).release());
  }

  return jit.get();
}

static Type *get_pointer_type(LLVMContext &c) { return Type::getInt8PtrTy(c); }

static Type *get_tensor_type(LLVMContext &c) {
  return StructType::get(get_pointer_type(c));
}

static Type *get_tensor_pointer_type(LLVMContext &c) {
  return PointerType::getUnqual(get_tensor_type(c));
}

static void initialize_llvm_jit_engine() {
  InitializeNativeTarget();
  InitializeNativeTargetAsmParser();
  InitializeNativeTargetAsmPrinter();
}

static std::string stdstr(const py::object &obj) { return py::str(obj); }

} // namespace

namespace tdnat {

py::cpp_function jit_compile_test() {
  auto c = std::make_unique<LLVMContext>();
  auto m = std::make_unique<Module>("torchdynamo", *c);

  auto entry_fn =
      Function::Create(FunctionType::get(llvm::Type::getVoidTy(*c), {}, false),
                       Function::ExternalLinkage, "entry_fn", m.get());
  auto dump_tensor_fn =
      Function::Create(FunctionType::get(Type::getVoidTy(*c),
                                         {get_tensor_pointer_type(*c)}, false),
                       Function::ExternalLinkage, "dump_tensor_fn", m.get());
  auto one_dim_tensor_fn = Function::Create(
      FunctionType::get(llvm::Type::getVoidTy(*c),
                        {get_tensor_type(*c), llvm::Type::getInt32Ty(*c)},
                        false),
      Function::ExternalLinkage, "one_dim_tensor_fn", m.get());

  auto bb = BasicBlock::Create(*c, "entry", entry_fn);
  IRBuilder<> builder(bb);

  auto tensor = builder.CreateAlloca(get_tensor_type(*c));
  builder.CreateCall(one_dim_tensor_fn, {tensor, builder.getInt32(5)});
  builder.CreateCall(dump_tensor_fn, {tensor});
  builder.CreateRetVoid();

  m->print(errs(), nullptr);
  JITSymbolFlags flags;
  flags |= JITSymbolFlags::Absolute;
  flags |= JITSymbolFlags::Callable;

  auto jit = get_jit();
  cantFail(jit->addIRModule({std::move(m), std::move(c)}));

  llvm::cantFail(jit->define(orc::absoluteSymbols(
      {{jit->mangleAndIntern("at_add"),
        JITEvaluatedSymbol::fromPointer(
            (at::Tensor(*)(const at::Tensor &, const at::Tensor &,
                           const at::Scalar &)) &
                at::add,
            flags)},
       {jit->mangleAndIntern("at_tensor"),
        JITEvaluatedSymbol::fromPointer(
            (at::Tensor(*)(at::ArrayRef<int>)) & at::tensor, flags)},
       {jit->mangleAndIntern("dump_tensor_fn"),
        JITEvaluatedSymbol::fromPointer(&dump_tensor, flags)},
       {jit->mangleAndIntern("one_dim_tensor_fn"),
        JITEvaluatedSymbol::fromPointer(&one_dim_tensor, flags)}})));

  return py::cpp_function([&](int64_t i) {
    std::cout << "Calling JIT'd." << std::endl;
    auto sym = cantFail(jit->lookup("entry_fn"));
    void (*fn)() = (void (*)())sym.getAddress();
    fn();
    return i;
  });
}

struct Node {};

struct Input : public Node {
  std::string name_;
  py::handle type_;
  Input(const std::string &name, const py::object &type)
      : name_(name), type_(type) {}
};

struct Statement : public Node {};

struct Output : public Node {
  py::handle type_;
  Output(const py::handle &type) : type_(type) {}
};

struct Program {
  std::vector<Input> input_;
  std::vector<Statement> statements_;
  std::vector<Output> output_;

  void dump();

  static Program from_obj(const py::object &obj);
};

Program Program::from_obj(const py::object &obj) {
  using namespace py::literals;

  auto root = obj.attr("_root");
  auto node = root.attr("next");

  Program program;
  program.input_.reserve(obj.attr("_len").cast<size_t>());

  std::unordered_map<std::string, Node *> name_map;
  while (!node.is(root)) {
    py::object op = node.attr("op");
    std::cout << "Op: " << py::str(op) << std::endl;

    if (op.equal("placeholder"_s)) {
      auto name = stdstr(node.attr("name"));
      program.input_.push_back({name, node.attr("type")});
      name_map[name] = &program.input_.back();
    } else if (op.equal("output"_s)) {
      auto ret = (py::tuple)node.attr("_args");
      program.output_.push_back({ret[0].get_type()});
    } else if (op.equal("get_attr"_s)) {
      assert(false && "no support for 'get_attr'");
    } else {
      auto target = node.attr("target");
      std::string fnname = target.attr("__name__").cast<std::string>();
      std::cout << "Found: " << fnname << std::endl;

      c10::OperatorName opname(fnname, std::string());
      auto op = c10::Dispatcher::singleton().findOp(opname);
      if (op.has_value()) {
        std::cout << "Got it!: "
                  << op->hasKernelForDispatchKey(c10::DispatchKey::CPU)
                  << std::endl;
      } else {
        std::cout << "Failed..." << std::endl;
      }
    }

    node = node.attr("next");
  }

  return program;
}

void dump_graph(const py::object &obj) { Program::from_obj(obj); }

void dump_operations() {
  auto &dispatcher = c10::Dispatcher::singleton();
  auto opnames = dispatcher.getAllOpNames();
  std::cout << "Len: " << opnames.size() << std::endl;
  for (auto &op : opnames) {
    std::cout << "(" << op.name.size() << ", " << op.overload_name.size() << ")"
              << std::endl;
    std::cout << "> " << op.name << "." << op.overload_name << std::endl;
  }
}

} // namespace _C

namespace {

PYBIND11_MODULE(_C, m) {
  initialize_llvm_jit_engine();

  // Testing a simple JIT example.
  m.def("jit_test", &tdnat::jit_compile_test);

  // Try to parse the FX graph and dump it.
  m.def("dump_graph", &tdnat::dump_graph);

  // Dump the operations found in the dispatcher.
  m.def("dump_operations", &tdnat::dump_operations);
}

} // namespace
