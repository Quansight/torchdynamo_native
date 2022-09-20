#include <tdnat/function.h>
#include <tdnat/llvm_function_type.h>

#include <llvm/ExecutionEngine/JITSymbol.h>
#include <llvm/ExecutionEngine/Orc/Core.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Support/Error.h>

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <sstream>

using namespace tdnat;

llvm::Function *Function::__add_aten_op_decl(ATenOpRef ref) {
  if (fnaddrmap_.find(ref.name()) == fnaddrmap_.end()) {
    fnaddrmap_[ref.name()] = ref.cpu();
    auto llvm_fn = llvm::Function::Create(ref.llvm_function_type(*mod_),
                                          llvm::GlobalValue::ExternalLinkage,
                                          ref.name(), *mod_);
    ref.add_attributes(llvm_fn);
  }
  return mod_->getFunction(ref.name());
}

void Function::__check_finalized(bool expected) {
  if (finalized_ != expected) {
    std::ostringstream msg;

    msg << "Function should";
    if (!expected) {
      msg << " not";
    }
    msg << " be finalized.";

    TORCH_CHECK(finalized_ == expected, msg.str());
  }
}

Function::Function(const FunctionData &data)
    : data_(data), ctx_(new llvm::LLVMContext()),
      mod_(new llvm::Module(data_.id_, *ctx_)), builder_(*ctx_),
      finalized_(false) {
  // Instantiate LLVM module.
  auto mod_id = std::string("Module_for_") + data_.id_;
  mod_.reset(new llvm::Module(mod_id, *ctx_));

  // The function signature will be:
  //
  //   void <function-name>(Tensor* in_tensors, Tensor* out_tensors);
  //
  // This allow us to create a function with a variable number of input
  // and output tensors.
  fn_ = llvm::Function::Create(
      ABILLVMFunctionType<void (*)(at::Tensor *, at::Tensor *)>::get(*mod_),
      llvm::GlobalValue::ExternalLinkage, data_.id_, *mod_);

  // Move builder to the first basic block of the function.
  builder_.SetInsertPoint(llvm::BasicBlock::Create(*ctx_, "entry", fn_));
}

Function::Function(Function &&fn)
    : data_(fn.data_), ctx_(fn.ctx_.release()), mod_(fn.mod_.release()),
      fn_(fn.fn_), builder_(*ctx_), symbolmap_(fn.symbolmap_),
      fnaddrmap_(fn.fnaddrmap_), finalized_(false) {
  builder_.SetInsertPoint(&fn_->getEntryBlock());
}

Value Function::set_placeholder(int i, const std::string &name) {
  __check_finalized();

  // Get the i-th input tensor.
  symbolmap_[name] = builder_.CreateGEP(fn_->getArg(0), builder_.getInt64(i));
  return {symbolmap_[name]};
}

std::vector<Value> Function::set_outputs(const std::vector<Value> &outputs) {
  __check_finalized();

  std::vector<Value> real_outputs;

  for (size_t i = 0; i < data_.out_tensors_; i++) {
    auto ptr = outputs[i].val_;
    auto value = builder_.CreateLoad(__get_type<at::Tensor>(), ptr);

    auto out_ptr = builder_.CreateGEP(fn_->getArg(1), builder_.getInt64(i));
    builder_.CreateStore(value, out_ptr);

    real_outputs.push_back({out_ptr});
  }

  return real_outputs;
}

Value Function::set_output(const Value &output) {
  return set_outputs({output})[0];
}

Value Function::add_call(const std::string &symbolname,
                         const std::string &opname,
                         const std::vector<Value> &args) {
  __check_finalized();

  auto opref_ = get_aten_op(opname);
  TORCH_CHECK(opref_.has_value(), "PyTorch operation not registered: ", opname);

  auto opref = opref_.value();
  auto opfn = __add_aten_op_decl(opref);

  auto values = std::vector<llvm::Value *>();
  std::transform(args.begin(), args.end(), std::back_inserter(values),
                 [](Value arg) { return arg.val_; });

  if (opref.returns_on_memory()) {
    auto alloc = opref.allocate_mem_for_ret(builder_);
    values.insert(values.begin(), alloc);
  }

  auto call = builder_.CreateCall(opfn, values);

  if (opref.returns_on_memory()) {
    symbolmap_[symbolname] = values[0];
  } else {
    symbolmap_[symbolname] = call;
  }

  return {symbolmap_[symbolname]};
}

Value Function::build_bool(bool b) {
  __check_finalized();
  return {builder_.getInt1(b)};
}

Value Function::build_scalar_type(at::ScalarType type) {
  __check_finalized();
  return {build_integer(static_cast<int8_t>(type))};
}

Value Function::build_optional_tensorlist(const std::vector<Value> &v) {
  using OptionalTensor = c10::optional<at::Tensor>;

  __check_finalized();

  auto optional_tensorlist_fn =
      __add_factory_decl<factory::OptionalTensorList>();

  auto size = build_integer(v.size()).val_;
  auto alloca = builder_.CreateAlloca(__get_type<OptionalTensor>(), size);
  auto alloca_ret =
      builder_.CreateAlloca(__get_type<c10::List<OptionalTensor>>());

  for (size_t i = 0; i < v.size(); i++) {
    auto ptr = v[i].val_;
    auto load = builder_.CreateLoad(__get_type<OptionalTensor>(), ptr);
    auto elem_ptr = builder_.CreateGEP(alloca, builder_.getInt64(i));
    builder_.CreateStore(load, elem_ptr);
  }

  builder_.CreateCall(optional_tensorlist_fn, {alloca_ret, alloca, size});
  return {alloca_ret};
}

Value Function::build_scalar(int64_t n) {
  __check_finalized();
  return __build_scalar<int64_t>({builder_.getInt64(n)});
}

void Function::dump() { mod_->print(llvm::outs(), nullptr); }

void Function::finalize() {
  __check_finalized();

  builder_.CreateRetVoid();
  finalized_ = true;

  bool any_errors = llvm::verifyModule(*mod_, &llvm::errs());
  if (any_errors) {
    mod_->print(llvm::errs(), nullptr);
    TORCH_CHECK(false, "Bad module");
  }
}

JITFunction Function::into_jit() {
  __check_finalized(true);

  auto id = fn_->getName();
  auto jit = llvm::cantFail(llvm::orc::LLJITBuilder().create());

  llvm::cantFail(jit->addIRModule({std::move(mod_), std::move(ctx_)}));

  auto symbols = llvm::orc::SymbolMap(fnaddrmap_.size());
  for (auto &pair : fnaddrmap_) {
    symbols.insert({jit->mangleAndIntern(pair.first),
                    llvm::JITEvaluatedSymbol(pair.second,
                                             llvm::JITSymbolFlags::Exported)});
  }
  llvm::cantFail(jit->define(llvm::orc::absoluteSymbols(symbols)));

  return {jit.release(), data_};
}
