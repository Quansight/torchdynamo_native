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

template <typename T> Value Function::__build_scalar(Value val) {
  auto scalar_fn = __add_factory_decl<factory::Scalar<T>>();

  auto alloca = builder_.CreateAlloca(__get_type<at::Scalar>());
  alloca->setAlignment(llvm::Align(alignof(at::Scalar)));

  builder_.CreateCall(scalar_fn, {alloca, val.val_});
  return {alloca};
}

template <typename T> Value Function::__build_optional(Value val) {
  auto fn = __add_factory_decl<factory::Optional<T>>();

  std::vector<llvm::Value *> args;

  if (IsABIMemoryClass<T>::value) {
    auto alloca = builder_.CreateAlloca(__get_type<c10::optional<T>>());
    args.push_back(alloca);
  }

  args.push_back(val.val_);

  auto call = builder_.CreateCall(fn, args);

  if (IsABIMemoryClass<T>::value) {
    return {args[0]};
  } else {
    return {call};
  }
}

template <typename T>
Value Function::__build_arrayref(const std::vector<Value> &vals,
                                 bool from_literal) {
  auto fn = __add_factory_decl<factory::ArrayRef<T>>();

  auto size = build_integer(vals.size()).val_;
  auto alloca = builder_.CreateAlloca(__get_type<T>(), size);

  for (size_t i = 0; i < vals.size(); i++) {
    llvm::Value *value = vals[i].val_;

    if (!from_literal) {
      value = builder_.CreateLoad(value);
    }

    auto gep = builder_.CreateGEP(alloca, builder_.getInt64(i));
    builder_.CreateStore(value, gep);
  }

  return {builder_.CreateCall(fn, {alloca, size})};
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

template <typename T> llvm::Type *Function::__get_type() {
  return LLVMType<T>::get(*mod_);
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

Value Function::set_placeholder(int i, const std::string &name) {
  __check_finalized();

  // Get the i-th input tensor.
  symbolmap_[name] = builder_.CreateGEP(fn_->getArg(0), builder_.getInt64(i));
  return {symbolmap_[name]};
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

template <typename T> Value Function::build_integer(T n) {
  __check_finalized();
  return {builder_.getIntN(sizeof(T) * 8, n)};
}

template Value Function::build_integer<size_t>(size_t);
template Value Function::build_integer<int32_t>(int32_t);
template Value Function::build_integer<int64_t>(int64_t);

template <typename T>
Value Function::build_arrayref(const std::vector<Value> &v) {
  __check_finalized();
  return __build_arrayref<T>(v, /* from_literal= */ false);
}

template Value Function::build_arrayref<at::Tensor>(const std::vector<Value> &);

template <typename T>
Value Function::build_arrayref_lit(const std::vector<Value> &v) {
  __check_finalized();
  return __build_arrayref<T>(v, /* from_literal= */ true);
}

template Value Function::build_arrayref_lit<int64_t>(const std::vector<Value> &);

template <typename T> Value Function::build_optional() {
  __check_finalized();

  auto nullopt_fn = __add_factory_decl<factory::NullOpt<T>>();

  std::vector<llvm::Value *> args;

  if (IsABIMemoryClass<T>::value) {
    auto alloca = builder_.CreateAlloca(__get_type<c10::optional<T>>());
    args.push_back(alloca);
  }

  auto call = builder_.CreateCall(nullopt_fn, args);

  if (IsABIMemoryClass<T>::value) {
    return {args[0]};
  } else {
    return {call};
  }
}

template Value Function::build_optional<at::Tensor>();
template Value Function::build_optional<at::ScalarType>();

template <typename T> Value Function::build_optional(Value val) {
  __check_finalized();
  return __build_optional<T>({val});
}

template Value Function::build_optional<at::Tensor>(Value);

template <typename T> Value Function::build_optional_lit(Value val) {
  __check_finalized();

  auto alloca = builder_.CreateAlloca(__get_type<T>());
  builder_.CreateStore(val.val_, alloca);

  return __build_optional<T>({alloca});
}

template Value Function::build_optional_lit<long>(Value);
template Value Function::build_optional_lit<at::ScalarType>(Value);

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
