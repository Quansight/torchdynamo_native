#include <tdnat/llvm_function_builder.h>
#include <tdnat/llvm_function_type.h>

using namespace tdnat;

LLVMFunctionBuilder::LLVMFunctionBuilder(const std::string &id,
                                         size_t in_tensors, size_t out_tensors)
    : in_tensors_(in_tensors), out_tensors_(out_tensors),
      ctx_(new llvm::LLVMContext()), mod_(new llvm::Module(id, *ctx_)),
      builder_(*ctx_), symbolmap_(), opaddrmap_() {
  // Instantiate LLVM module.
  mod_.reset(new llvm::Module(id, *ctx_));

  // The function signature will be:
  //
  //   void <function-name>(Tensor* in_tensors, Tensor* out_tensors);
  //
  // This allow us to create a function with a variable number of input
  // and output tensors.
  fn_ = llvm::Function::Create(
      ABILLVMFunctionType<void (*)(at::Tensor *, at::Tensor *)>::get(*ctx_),
      llvm::GlobalValue::ExternalLinkage, id, *mod_);

  // Move builder to the first basic block of the function.
  builder_.SetInsertPoint(llvm::BasicBlock::Create(*ctx_, "entry", fn_));
}

void LLVMFunctionBuilder::add_tensor_placeholder(int i,
                                                 const std::string &name) {
  // Get the i-th input tensor.
  symbolmap_[name] = builder_.CreateGEP(fn_->getArg(0), builder_.getInt64(i));
}

void LLVMFunctionBuilder::add_call(const std::string &symbolname,
                                   const std::string &opname,
                                   const std::vector<Value> &args) {
  auto opref = get_aten_op(opname).value();
  auto opfn = get_or_declare_op(opref);

  auto values = std::vector<llvm::Value *>();
  std::transform(args.begin(), args.end(), std::back_inserter(values),
                 [](Value arg) { return arg.val_; });

  auto call = builder_.CreateCall(opfn, values);

  if (opref.returns_on_memory()) {
    symbolmap_[symbolname] = opfn->getArg(0);
  } else {
    symbolmap_[symbolname] = call;
  }
}

void LLVMFunctionBuilder::add_outputs(const std::vector<std::string> &outputs) {
  for (size_t i = 0; i < out_tensors_; i++) {
    auto value = symbolmap_[outputs[i]];
    auto out_ptr = builder_.CreateGEP(fn_->getArg(1), builder_.getInt64(i));
    builder_.CreateStore(value, out_ptr);
  }
}

void LLVMFunctionBuilder::add_output(const std::string &output) {
  add_outputs({output});
}

llvm::Function *LLVMFunctionBuilder::get_or_declare_op(ATenOpRef ref) {
  if (opaddrmap_.find(ref.name()) == opaddrmap_.end()) {
    opaddrmap_[ref.name()] = ref.cpu();
    return llvm::Function::Create(ref.llvm_function_type(mod_->getContext()),
                                  llvm::GlobalValue::ExternalLinkage,
                                  ref.name(), *mod_);
  }
  return mod_->getFunction(ref.name());
}

Value LLVMFunctionBuilder::build_int(int64_t n) {
  return {builder_.getInt64(n)};
}

Value LLVMFunctionBuilder::build_intarray(const std::vector<int64_t> &v) {
  auto size = builder_.getInt64(v.size());
  auto space = builder_.CreateAlloca(llvm::Type::getInt64Ty(*ctx_), size);

  for (size_t i = 0; i < v.size(); i++) {
    auto addr = builder_.CreateGEP(space, builder_.getInt64(i));
    builder_.CreateStore(builder_.getInt64(v[i]), addr);
  }

  auto getfn = mod_->getFunction("get_intarray_ref");
  return {builder_.CreateCall(getfn, {space, size})};
}
