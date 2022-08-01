#include <torchdynamo_native/csrc/ops.h>

#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/Support/TargetSelect.h>

#include <torch/csrc/utils/pybind.h>

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

static LLVMContext *get_context() {
  static std::unique_ptr<LLVMContext> context(nullptr);
  if (context.get() == nullptr) {
    context.reset(new LLVMContext());
  }
  return context.get();
}

static at::IntArrayRef get_intarray_ref(int64_t *begin, int64_t size) {
  return {begin, (size_t)size};
}

static std::string stdstr(const py::object &obj) { return py::str(obj); }

} // namespace

namespace tdnat {

/*
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
*/

struct Arg {
  llvm::Value *val_;
};

struct ModuleBuilder {
  std::unordered_map<std::string, llvm::Value *> symbolmap_;
  std::unordered_map<std::string, Addr> opaddrmap_;
  llvm::IRBuilder<> builder_;

  std::unique_ptr<llvm::Module> mod_;
  llvm::Function *fn_;

  ModuleBuilder(const std::string &id, size_t in_tensors, size_t out_tensors)
      : symbolmap_(), opaddrmap_(), builder_(*get_context()) {
    auto ctx = get_context();

    // Instantiate LLVM module.
    mod_.reset(new llvm::Module(id, *ctx));

    // Create the corresponding LLVM function.
    // 1. Parameter type list: every input is a Tensor, so we only need
    //    a list of pointers, here.
    auto param_types =
        std::vector<llvm::Type *>{in_tensors, get_pointer_type(*ctx)};
    // 2. Return type: it's either void or a list of tensors (pointer to
    //    the actual list)
    auto ret_type = (out_tensors > 0) ? get_pointer_type(*ctx)
                                      : llvm::Type::getVoidTy(*ctx);
    // 3. Create the function with the specified function type.
    fn_ = llvm::Function::Create(
        llvm::FunctionType::get(ret_type, param_types, false),
        llvm::GlobalValue::ExternalLinkage, id, *mod_);

    // Move builder to the first basic block of the function.
    builder_.SetInsertPoint(llvm::BasicBlock::Create(*ctx, "entry", fn_));
  }

  void add_tensor_placeholder(int i, const std::string &name) {
    symbolmap_[name] = fn_->getOperand(i);
  }

  void add_call_function(const std::string &symbolname,
                         const std::string &opname,
                         const std::vector<Arg> &args) {
    auto opref = get_aten_op(opname).value();
    opaddrmap_[opname] = opref.cpufn_;
  }

  void add_statement(const std::string &op, const std::string &name,
                     const std::string &aten_op_name, py::list args,
                     py::dict kwargs) {}

  Arg build_int(int64_t n) { return {builder_.getInt64(n)}; }

  Arg build_intarray(const std::vector<int64_t> &v) {
    auto size = builder_.getInt64(v.size());
    auto space =
        builder_.CreateAlloca(llvm::Type::getInt64Ty(*get_context()), size);

    for (size_t i = 0; i < v.size(); i++) {
      auto addr = builder_.CreateGEP(space, builder_.getInt64(i));
      builder_.CreateStore(builder_.getInt64(v[i]), addr);
    }

    auto getfn = mod_->getFunction("get_intarray_ref");
    return {builder_.CreateCall(getfn, {space, size})};
  }
};

} // namespace tdnat

namespace {

PYBIND11_MODULE(_C, m) {
  initialize_llvm_jit_engine();

  // Testing a simple JIT example.
  // m.def("jit_test", &tdnat::jit_compile_test);

  // Try to parse the FX graph and dump it.
  // m.def("dump_graph", &tdnat::dump_graph);

  // Dump the operations found in the dispatcher.
  // m.def("dump_operations", &tdnat::dump_operations);

  py::class_<tdnat::ModuleBuilder>(m, "Module")
      .def("add_statement", &tdnat::ModuleBuilder::add_statement);
}

} // namespace
