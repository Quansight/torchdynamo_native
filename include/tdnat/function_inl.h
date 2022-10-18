namespace tdnat
{

template <typename Return, typename... Args>
llvm::Function *Function::_add_function_decl(const std::string &name, Return (*fn)(Args...))
{
  auto &mod = *module_.getModuleUnlocked();
  if (fnaddrmap_.find(name) == fnaddrmap_.end()) {
    fnaddrmap_[name] = reinterpret_cast<Addr>(fn);
    auto llvm_fn = llvm::Function::Create(
        ABILLVMFunctionType<Return (*)(Args...)>::get(mod),
        llvm::GlobalValue::ExternalLinkage,
        name,
        mod
    );
    add_attributes<Return, Args...>(llvm_fn);
  }
  return mod.getFunction(name);
}

template <typename API>
llvm::Function *Function::_add_api_decl()
{
  return _add_function_decl(API::name(), &API::run);
}

template <typename T>
llvm::Type *Function::_get_type()
{
  return LLVMType<T>::get(*module_.getModuleUnlocked());
}

template <typename T>
Value Function::build_int(T i)
{
  static_assert(std::is_integral<T>::value);
  return {builder_.getIntN(sizeof(T) * 8, i)};
}

template <typename Repr, typename Enum>
Value Function::build_int_from_enum(Enum e)
{
  return {build_int(static_cast<Repr>(e))};
}

template <typename T>
Value Function::build_float(T f)
{
  static_assert(std::is_floating_point<T>::value);
  return {llvm::ConstantFP::get(fn_->getContext(), llvm::APFloat(f))};
}

template <typename T>
Value Function::build_scalar(Value literal)
{
  return {builder_.CreateCall(_add_api_decl<jit::Scalar<T>>(), {*literal})};
}

template <typename T>
Value Function::build_array(const std::vector<Value> &elements)
{
  auto size = *build_int(elements.size());
  auto alloca = builder_.CreateAlloca(_get_type<T>(), size);

  for (size_t i = 0; i < elements.size(); i++) {
    auto gep = builder_.CreateGEP(alloca, builder_.getInt64(i));
    builder_.CreateStore(*elements[i], gep);
  }

  return {alloca};
}

template <typename T>
Value Function::build_nullopt()
{
  auto nullopt_fn = _add_api_decl<jit::NullOpt<T>>();
  return {builder_.CreateCall(nullopt_fn, {})};
}

template <typename T, typename... Args>
Value Function::build_optional(typename replace<Args, Value>::type... args)
{
  auto optional_fn = _add_api_decl<jit::Optional<T, Args...>>();
  return {builder_.CreateCall(optional_fn, {*args...})};
}

template <typename T>
Value Function::build_nullopt_optionalarrayref()
{
  auto nullopt_fn = _add_api_decl<jit::NullOptOptionalArrayRef<T>>();
  return {builder_.CreateCall(nullopt_fn, {})};
}

template <typename T>
Value Function::build_optionalarrayref(const std::vector<Value> &elements)
{
  auto optionalarrayref_fn = _add_api_decl<jit::OptionalArrayRef<T>>();
  auto size = build_int<int64_t>(elements.size());
  auto ptr = build_array<T>(elements);
  return {builder_.CreateCall(optionalarrayref_fn, {*ptr, *size})};
}

template <typename T>
Value Function::build_list(const std::vector<Value> &elements)
{
  auto list_fn = _add_api_decl<jit::List<T>>();
  auto list_ptr = build_array<T>(elements);
  auto list_size = build_int<int64_t>(elements.size());
  return {builder_.CreateCall(list_fn, {*list_ptr, *list_size})};
}

template <typename T>
Value Function::build_vector_index(Value vector, Value position)
{
  auto at_fn = _add_api_decl<jit::VectorIndex<T>>();
  return {builder_.CreateCall(at_fn, {*vector, *position})};
}

} // namespace tdnat
