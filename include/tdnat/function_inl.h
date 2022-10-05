namespace tdnat
{

template <typename Return, typename... Args>
llvm::Function *Function::_add_function_decl(const std::string &name, Return (*fn)(Args...))
{
  if (fnaddrmap_.find(name) == fnaddrmap_.end()) {
    fnaddrmap_[name] = reinterpret_cast<Addr>(fn);
    auto llvm_fn = llvm::Function::Create(
        ABILLVMFunctionType<Return (*)(Args...)>::get(*mod_),
        llvm::GlobalValue::ExternalLinkage,
        name,
        *mod_
    );
    add_attributes<Return, Args...>(llvm_fn);
  }
  return mod_->getFunction(name);
}

template <typename Factory>
llvm::Function *Function::_add_factory_decl()
{
  return _add_function_decl(Factory::name(), &Factory::create);
}

template <typename T>
Value Function::_build_scalar(Value val)
{
  auto scalar_fn = _add_factory_decl<factory::Scalar<T>>();

  auto alloca = builder_.CreateAlloca(_get_type<at::Scalar>());
  alloca->setAlignment(llvm::Align(alignof(at::Scalar)));

  builder_.CreateCall(scalar_fn, {alloca, val.val_});
  return {alloca};
}

template <typename T, typename Factory>
Value Function::_build_optional(c10::optional<Value> val)
{
  auto fn = _add_factory_decl<Factory>();

  std::vector<llvm::Value *> args;

  if (IsABIMemoryClass<T>::value) {
    auto alloca = builder_.CreateAlloca(_get_type<c10::optional<T>>());
    args.push_back(alloca);
  }

  if (val.has_value()) {
    args.push_back(val->val_);
  }

  auto call = builder_.CreateCall(fn, args);

  if (IsABIMemoryClass<T>::value) {
    return {args[0]};
  } else {
    return {call};
  }
}

template <typename T>
Value Function::_build_arrayref(const std::vector<Value> &vals, bool from_literal)
{
  auto fn = _add_factory_decl<factory::ArrayRef<T>>();

  auto size = build_int(vals.size()).val_;
  auto alloca = builder_.CreateAlloca(_get_type<T>(), size);

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

template <typename T>
llvm::Type *Function::_get_type()
{
  return LLVMType<T>::get(*mod_);
}

template <typename T>
Value Function::build_int(T n)
{
  static_assert(std::is_integral<T>::value);
  _check_finalized();
  return {builder_.getIntN(sizeof(T) * 8, n)};
}

template <typename T>
Value Function::build_float(T n)
{
  static_assert(std::is_floating_point<T>::value);
  _check_finalized();
  return {llvm::ConstantFP::get(*ctx_, llvm::APFloat(n))};
}

template <typename T>
Value Function::build_arrayref(const std::vector<Value> &v)
{
  _check_finalized();
  return _build_arrayref<T>(v, /* from_literal= */ false);
}

template <typename T>
Value Function::build_arrayref_lit(const std::vector<Value> &v)
{
  _check_finalized();
  return _build_arrayref<T>(v, /* from_literal= */ true);
}

template <typename T>
Value Function::build_nullopt()
{
  _check_finalized();
  return _build_optional<T, factory::NullOpt<T>>();
}

template <typename T>
Value Function::build_optional(Value val)
{
  _check_finalized();
  return _build_optional<T, factory::Optional<T>>(val);
}

template <typename T>
Value Function::build_optional_lit(Value val)
{
  _check_finalized();

  auto alloca = builder_.CreateAlloca(_get_type<T>());
  builder_.CreateStore(val.val_, alloca);

  return build_optional<T>({alloca});
}

} // namespace tdnat
