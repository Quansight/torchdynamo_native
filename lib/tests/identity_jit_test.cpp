#include <gtest/gtest.h>

#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/Support/TargetSelect.h>

#include <ATen/ops/diag.h>
#include <ATen/ops/ones.h>
#include <ATen/ops/rand.h>
#include <ATen/ops/tensor.h>

#include <iostream>
#include <memory>
#include <type_traits>

#include <tdnat/llvm.h>
#include <tdnat/ops.h>

#include "types.h"

using namespace tdnat;

static const char *ModuleName = "jit_test";
static const char *EntryFn = "entry_function";
static const char *IdentityFn = "identity_function";
static const char *EntryBlock = "entry";

template <typename T> static T identity(T t) { return t; }

class Environment : public ::testing::Environment {
public:
  ~Environment() override {}
  void SetUp() override { initialize_llvm(); }
};

const auto *env = ::testing::AddGlobalTestEnvironment(new Environment());

template <typename T> struct BuildLLVMFunction {};

template <typename... Args> struct BuildLLVMFunction<void (*)(Args...)> {
  static void call(llvm::Function &fn, llvm::Function &idfn) {
    auto arg0 = fn.getArg(0);
    auto arg1 = fn.getArg(1);

    auto bb = llvm::BasicBlock::Create(fn.getContext(), EntryBlock, &fn);
    llvm::IRBuilder<> builder(bb);

    builder.CreateCall(&idfn, {arg0, arg1});
    builder.CreateRetVoid();
  }
};

template <typename Return, typename... Args>
struct BuildLLVMFunction<Return (*)(Args...)> {
  static void call(llvm::Function &fn, llvm::Function &idfn) {
    auto arg = fn.getArg(0);

    auto bb = llvm::BasicBlock::Create(fn.getContext(), EntryBlock, &fn);
    llvm::IRBuilder<> builder(bb);

    auto ret = builder.CreateCall(&idfn, {arg});
    builder.CreateRet(ret);
  }
};

template <typename T> struct ReturnType { using type = T; };
template <typename T> struct ReturnType<at::ArrayRef<T>> {
  using type = std::vector<T>;
};

template <typename T> struct InstancesTrait {};

template <typename T> struct InstancesTrait<c10::optional<T>> {
  static std::vector<typename ReturnType<c10::optional<T>>::type> instances() {
    std::vector<typename ReturnType<c10::optional<T>>::type> inst;
    inst.push_back(c10::nullopt);

    for (auto &arg : InstancesTrait<T>::instances()) {
      inst.push_back(arg);
    }

    return inst;
  }
};

#define IMPL_INSTANCE(TYPE, ...)                                               \
  template <> struct InstancesTrait<TYPE> {                                    \
    static std::vector<typename ReturnType<TYPE>::type> instances() {          \
      return {__VA_ARGS__};                                                    \
    }                                                                          \
  };
IMPL_INSTANCE(int64_t, 42);
IMPL_INSTANCE(double, 3.14159);
IMPL_INSTANCE(at::IntArrayRef, std::vector<long>(20, 0),
              std::vector<long>({1, 2, 3, 4}));
IMPL_INSTANCE(at::Tensor, at::tensor({42}), at::ones({3, 3}),
              at::diag(at::rand({4})));
#undef IMPL_INSTANCE

template <typename T, typename _Tp = void> struct EqualTrait {};

template <typename T>
struct EqualTrait<T, std::enable_if_t<!IsInvisibleReturnType<T>::value>> {
  static bool equal(const T &t1, const T &t2) { return t1 == t2; }
};

template <typename T>
struct EqualTrait<c10::optional<T>,
                  std::enable_if_t<IsInvisibleReturnType<T>::value>> {
  static bool equal(const c10::optional<T> &t1, const c10::optional<T> &t2) {
    if (t1.has_value() != t2.has_value()) {
      return false;
    }
    if (!t1.has_value()) {
      return true;
    }
    return EqualTrait<T>::equal(*t1, *t2);
  }
};

#define IMPL_EQUAL(TYPE, RET)                                                  \
  template <> struct EqualTrait<TYPE> {                                        \
    static bool equal(const TYPE &t1, const TYPE &t2) { return RET; }          \
  };
IMPL_EQUAL(at::Tensor, t1.equal(t2));
#undef IMPL_EQUAL

template <typename T> std::vector<typename ReturnType<T>::type> instances() {
  return InstancesTrait<T>::instances();
}

template <typename T> T equal(const T &t1, const T &t2) {
  return EqualTrait<T>::equal(t1, t2);
}

template <typename T> class JITTest : public testing::Test {
public:
  using FnTypePtr = typename ABITrait<T (*)(T)>::type;

  JITTest() : jit_(llvm::cantFail(llvm::orc::LLJITBuilder().create())) {}

  void SetUp() override {
    auto ctx = std::make_unique<llvm::LLVMContext>();
    auto mod = std::make_unique<llvm::Module>(ModuleName, *ctx);

    auto entry_fn = llvm::Function::Create(
        ToLLVMFunctionType<FnTypePtr>::call(*ctx),
        llvm::Function::ExternalLinkage, EntryFn, mod.get());

    auto identity_fn = llvm::Function::Create(
        ToLLVMFunctionType<FnTypePtr>::call(*ctx),
        llvm::Function::ExternalLinkage, IdentityFn, mod.get());

    BuildLLVMFunction<FnTypePtr>::call(*entry_fn, *identity_fn);

    llvm::cantFail(jit_->addIRModule({std::move(mod), std::move(ctx)}));

    llvm::cantFail(jit_->define(llvm::orc::absoluteSymbols(
        {{jit_->mangleAndIntern(IdentityFn),
          llvm::JITEvaluatedSymbol::fromPointer(&identity<T>)}})));
  }

protected:
  std::unique_ptr<llvm::orc::LLJIT> jit_;
};

TYPED_TEST_SUITE(JITTest, PyTorchTypes);
TYPED_TEST(JITTest, IdentityTest) {
  auto symbol = llvm::cantFail(this->jit_->lookup(EntryFn));
  auto func = (TypeParam(*)(TypeParam))symbol.getAddress();

  for (auto &arg : instances<TypeParam>()) {
    auto result = func(arg);
    ASSERT_TRUE(EqualTrait<TypeParam>::equal(arg, result));
  }
}
