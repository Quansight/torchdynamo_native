#include <memory>
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
#include <utility>

using namespace tdnat;

llvm::Function *Function::_add_aten_op_decl(ATenOpRef ref)
{
  if (fnaddrmap_.find(ref.name()) == fnaddrmap_.end()) {
    fnaddrmap_[ref.name()] = ref.cpu();
    auto llvm_fn = llvm::Function::Create(
        ref.llvm_function_type(*mod_),
        llvm::GlobalValue::ExternalLinkage,
        ref.name(),
        *mod_
    );
    ref.add_attributes(llvm_fn);
  }
  return mod_->getFunction(ref.name());
}

void Function::_check_finalized(bool expected)
{
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

Function::Function(FunctionData data) :
    data_(std::move(data)),
    ctx_(new llvm::LLVMContext()),
    builder_(*ctx_),
    finalized_(false)
{
  // Instantiate LLVM module.
  auto mod_id = std::string("Module_for_") + data_.id_;
  mod_ = std::make_unique<llvm::Module>(mod_id, *ctx_);

  // The function signature will be:
  //
  //   void <function-name>(Tensor* in_tensors, Tensor* out_tensors);
  //
  // This allow us to create a function with a variable number of input
  // and output tensors.
  fn_ = llvm::Function::Create(
      ABILLVMFunctionType<void (*)(at::Tensor *, at::Tensor *)>::get(*mod_),
      llvm::GlobalValue::ExternalLinkage,
      data_.id_,
      *mod_
  );

  // Move builder to the first basic block of the function.
  builder_.SetInsertPoint(llvm::BasicBlock::Create(*ctx_, "entry", fn_));
}

Function::Function(Function &&fn) noexcept :
    data_(std::move(fn.data_)),
    ctx_(fn.ctx_.release()),
    mod_(fn.mod_.release()),
    fn_(fn.fn_),
    builder_(*ctx_),
    symbolmap_(std::move(fn.symbolmap_)),
    fnaddrmap_(std::move(fn.fnaddrmap_)),
    finalized_(false)
{
  builder_.SetInsertPoint(&fn_->getEntryBlock());
}

Value Function::set_placeholder(int i, const std::string &name)
{
  _check_finalized();

  // Get the i-th input tensor.
  symbolmap_[name] = builder_.CreateGEP(fn_->getArg(0), builder_.getInt64(i));
  return {symbolmap_[name]};
}

std::vector<Value> Function::set_outputs(const std::vector<Value> &outputs)
{
  _check_finalized();

  std::vector<Value> real_outputs;

  for (size_t i = 0; i < data_.out_tensors_; i++) {
    auto ptr = outputs[i].val_;
    auto value = builder_.CreateLoad(_get_type<at::Tensor>(), ptr);

    auto out_ptr = builder_.CreateGEP(fn_->getArg(1), builder_.getInt64(i));
    builder_.CreateStore(value, out_ptr);

    real_outputs.push_back({out_ptr});
  }

  return real_outputs;
}

Value Function::set_output(const Value &output)
{
  return set_outputs({output})[0];
}

Value Function::add_call(
    const std::string &symbolname,
    const std::string &opname,
    const std::vector<Value> &args
)
{
  _check_finalized();

  auto opref_ = get_aten_op(opname);
  TORCH_CHECK(opref_.has_value(), "PyTorch operation not registered: ", opname);

  auto opref = opref_.value();
  auto opfn = _add_aten_op_decl(opref);

  auto values = std::vector<llvm::Value *>();
  std::transform(args.begin(), args.end(), std::back_inserter(values), [](Value arg) {
    return arg.val_;
  });

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

Value Function::build_bool(bool b)
{
  _check_finalized();
  return {builder_.getInt1(b)};
}

Value Function::build_optional_tensorlist(const std::vector<Value> &v)
{
  using OptionalTensor = c10::optional<at::Tensor>;

  _check_finalized();

  auto optional_tensorlist_fn = _add_factory_decl<factory::OptionalTensorList>();

  auto size = build_integer(v.size()).val_;
  auto alloca = builder_.CreateAlloca(_get_type<OptionalTensor>(), size);
  auto alloca_ret = builder_.CreateAlloca(_get_type<c10::List<OptionalTensor>>());

  for (size_t i = 0; i < v.size(); i++) {
    auto ptr = v[i].val_;
    auto load = builder_.CreateLoad(_get_type<OptionalTensor>(), ptr);
    auto elem_ptr = builder_.CreateGEP(alloca, builder_.getInt64(i));
    builder_.CreateStore(load, elem_ptr);
  }

  builder_.CreateCall(optional_tensorlist_fn, {alloca_ret, alloca, size});
  return {alloca_ret};
}

Value Function::build_scalar_type(at::ScalarType type)
{
  _check_finalized();
  return {build_integer(static_cast<int8_t>(type))};
}

Value Function::build_scalar(int64_t n)
{
  _check_finalized();
  return _build_scalar<int64_t>({builder_.getInt64(n)});
}

struct VectorAtTensor {
  static std::string name()
  {
    return "vector_at_tensor";
  }

  static const at::Tensor &at(const std::vector<at::Tensor> &v, int64_t i)
  {
    return v[i];
  }
};

Value Function::build_vector_at_tensor(Value val, Value position)
{
  _check_finalized();
  auto at_fn = _add_function_decl(VectorAtTensor::name(), &VectorAtTensor::at);
  return {builder_.CreateCall(at_fn, {val.val_, position.val_})};
}

void Function::dump()
{
  mod_->print(llvm::outs(), nullptr);
}

void Function::finalize()
{
  _check_finalized();

  builder_.CreateRetVoid();
  finalized_ = true;

  bool any_errors = llvm::verifyModule(*mod_, &llvm::errs());
  if (any_errors) {
    mod_->print(llvm::errs(), nullptr);
    TORCH_CHECK(false, "Bad module");
  }
}

JITFunction Function::into_jit()
{
  _check_finalized(true);

  auto id = fn_->getName();
  auto jit = llvm::cantFail(llvm::orc::LLJITBuilder().create());

  llvm::cantFail(jit->addIRModule({std::move(mod_), std::move(ctx_)}));

  auto symbols = llvm::orc::SymbolMap(fnaddrmap_.size());
  for (auto &pair : fnaddrmap_) {
    symbols.insert(std::make_pair(
        jit->mangleAndIntern(pair.first),
        llvm::JITEvaluatedSymbol(pair.second, llvm::JITSymbolFlags::Exported)
    ));
  }
  llvm::cantFail(jit->define(llvm::orc::absoluteSymbols(symbols)));

  return {jit.release(), data_};
}
