#include <tdnat/function.h>
#include <tdnat/llvm_function_type.h>

#include <llvm/ExecutionEngine/JITSymbol.h>
#include <llvm/ExecutionEngine/Orc/Core.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/raw_ostream.h>

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <memory>
#include <sstream>
#include <utility>

using namespace tdnat;

llvm::Function *Function::_add_aten_op_decl(ATenOpRef ref)
{
  auto &mod = *module_.getModuleUnlocked();
  if (fnaddrmap_.find(ref.name()) == fnaddrmap_.end()) {
    fnaddrmap_[ref.name()] = ref.cpu();
    auto llvm_fn = llvm::Function::Create(
        ref.llvm_function_type(mod),
        llvm::GlobalValue::ExternalLinkage,
        ref.name(),
        mod
    );
    ref.add_attributes(llvm_fn);
  }
  return mod.getFunction(ref.name());
}

Function::Function(
    std::unique_ptr<llvm::Module> mod,
    std::unique_ptr<llvm::LLVMContext> ctx,
    FunctionData data
) :
    data_(std::move(data)),
    module_(std::move(mod), std::move(ctx)),
    builder_(*module_.getContext().getContext())
{
  auto &module = *module_.getModuleUnlocked();

  // The function signature will be:
  //
  //   void <function-name>(Tensor* in_tensors, Tensor** out_tensors);
  //
  // This allow us to create a function with a variable number of input
  // and output tensors.
  fn_ = llvm::Function::Create(
      ABILLVMFunctionType<void (*)(at::Tensor *, at::Tensor **)>::get(module),
      llvm::GlobalValue::ExternalLinkage,
      data_.id_,
      module
  );

  // Move builder to the first basic block of the function.
  builder_.SetInsertPoint(llvm::BasicBlock::Create(module.getContext(), "entry", fn_));
}

Value Function::set_placeholder(int i, const std::string &name)
{
  // Get the i-th input tensor.
  symbolmap_[name] = builder_.CreateGEP(fn_->getArg(0), builder_.getInt64(i));
  return {symbolmap_[name]};
}

std::vector<Value> Function::set_output_from_refs(const std::vector<Value> &outputs)
{
  std::vector<Value> output_values;

  for (size_t i = 0; i < data_.out_tensors_; i++) {
    auto out_ptr = builder_.CreateGEP(fn_->getArg(1), builder_.getInt64(i));
    builder_.CreateStore(*(outputs[i]), out_ptr);
    output_values.push_back({out_ptr});
  }

  return output_values;
}

Value Function::set_output_from_ref(const Value &output)
{
  return set_output_from_refs({output})[0];
}

Value Function::add_call(
    const std::string &symbolname,
    const std::string &opname,
    const std::vector<Value> &args
)
{
  auto opref_ = get_aten_op(opname);
  TORCH_CHECK(opref_.has_value(), "PyTorch operation not registered: ", opname);

  auto opref = opref_.value();
  auto opfn = _add_aten_op_decl(opref);

  auto values = std::vector<llvm::Value *>();
  std::transform(args.begin(), args.end(), std::back_inserter(values), [](Value arg) {
    return *arg;
  });

  if (opref.returns_on_memory()) {
    auto alloc = opref.allocate_mem_for_ret(builder_);
    values.insert(values.begin(), alloc);
  }

  auto function_type = opref.llvm_function_type(*module_.getModuleUnlocked());
  if (function_type->getNumParams() != values.size()) {
    std::string buf;
    llvm::raw_string_ostream rso(buf);
    function_type->print(rso);
    TORCH_CHECK(false, "Unexpected number of parameters for ", rso.str(), ": got ", values.size());
  }

  auto call = builder_.CreateCall(opfn, values);

  if (opref.returns_on_memory()) {
    symbolmap_[symbolname] = values[0];
  } else {
    symbolmap_[symbolname] = call;
  }

  return {symbolmap_[symbolname]};
}

Value Function::build_bool(bool b)
{
  return {builder_.getInt1(b)};
}

Value Function::build_load(Value val)
{
  TORCH_CHECK((*val)->getType()->isPointerTy(), "Can't load value from non-pointer type.");
  return {builder_.CreateLoad(*val)};
}

Value Function::build_str(const std::string &s)
{
  return {builder_.CreateGlobalStringPtr(s)};
}

void Function::dump()
{
  module_.getModuleUnlocked()->print(llvm::outs(), nullptr);
}

JITFunction Function::into_jit()
{
  if (builder_.GetInsertBlock()->getTerminator() == nullptr) {
    builder_.CreateRetVoid();
  }

  auto &mod = *module_.getModuleUnlocked();
  if (llvm::verifyModule(mod, &llvm::errs())) {
    mod.print(llvm::errs(), nullptr);
    TORCH_CHECK(false, "Bad module");
  }

  auto jit = llvm::cantFail(llvm::orc::LLJITBuilder().create());
  auto symbols = llvm::orc::SymbolMap(fnaddrmap_.size());

  llvm::cantFail(jit->addIRModule(llvm::orc::cloneToNewContext(module_)));
  for (auto &pair : fnaddrmap_) {
    symbols.insert(std::make_pair(
        jit->mangleAndIntern(pair.first),
        llvm::JITEvaluatedSymbol(pair.second, llvm::JITSymbolFlags::Exported)
    ));
  }

  llvm::cantFail(jit->define(llvm::orc::absoluteSymbols(symbols)));
  return {jit.release(), data_};
}
