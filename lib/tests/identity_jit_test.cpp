#include <gtest/gtest.h>

#include <tdnat/llvm_function_type.h>
#include <tdnat/ops.h>

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

#include "tdnat/utils.h"
#include "test_utils.h"
#include "llvm/Support/raw_ostream.h"

using namespace tdnat;

static const char *ModuleName = "jit_test";
static const char *EntryFn = "entry_function";
static const char *IdentityFn = "identity_function";
static const char *EntryBlock = "entry";

template <typename T> static T identity(T t) { return t; }
template <> at::Scalar identity<at::Scalar>(at::Scalar s) {
  std::cout << s << std::endl;
  return s;
}

class Environment : public ::testing::Environment {
public:
  ~Environment() override {}
  void SetUp() override { initialize_llvm(); }
};

const auto *env = ::testing::AddGlobalTestEnvironment(new Environment());

template <typename T, typename _Tp = void> struct BuildLLVMFunction {
  static void call(llvm::Function &fn, llvm::Function &idfn) {
    auto arg = fn.getArg(0);

    auto bb = llvm::BasicBlock::Create(fn.getContext(), EntryBlock, &fn);
    llvm::IRBuilder<> builder(bb);

    auto ret = builder.CreateCall(&idfn, {arg});
    builder.CreateRet(ret);
  }
};

template <typename T> struct BuildLLVMFunction<T, std::enable_if_t<ReturnsOnMemory<T>::value>> {
  static void call(llvm::Function &fn, llvm::Function &idfn) {
    auto arg0 = fn.getArg(0);
    auto arg1 = fn.getArg(1);

    auto bb = llvm::BasicBlock::Create(fn.getContext(), EntryBlock, &fn);
    llvm::IRBuilder<> builder(bb);

    builder.CreateCall(&idfn, {arg0, arg1});
    builder.CreateRetVoid();
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

template <typename T, typename _Tp = void> struct EqualTrait {};

template <typename T> struct EqualTrait<c10::optional<T>> {
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

#define IMPL_INSTANCE(TYPE, ...)                                               \
  template <> struct InstancesTrait<TYPE> {                                    \
    static std::vector<typename ReturnType<TYPE>::type> instances() {          \
      return {__VA_ARGS__};                                                    \
    }                                                                          \
  };

#define IMPL_EQUAL(TYPE, RET)                                                  \
  template <> struct EqualTrait<TYPE> {                                        \
    static bool equal(const TYPE &t1, const TYPE &t2) { return RET; }          \
  };

#define IMPL_EQUAL_OPERATOR(TYPE)                                              \
  template <> struct EqualTrait<TYPE> {                                        \
    static bool equal(const TYPE &t1, const TYPE &t2) { return t1 == t2; }     \
  };

IMPL_INSTANCE(bool, true, false);
IMPL_EQUAL_OPERATOR(bool)

IMPL_INSTANCE(int64_t, 42);
IMPL_EQUAL_OPERATOR(int64_t)

IMPL_INSTANCE(double, 3.14159);
IMPL_EQUAL_OPERATOR(double)

IMPL_INSTANCE(at::ArrayRef<int64_t>, std::vector<int64_t>(20, 0),
              std::vector<int64_t>({1, 2, 3, 4}));
IMPL_EQUAL_OPERATOR(at::ArrayRef<int64_t>)

IMPL_INSTANCE(at::Tensor, at::tensor({42}), at::ones({3, 3}),
              at::diag(at::rand({4})));
IMPL_EQUAL(at::Tensor, t1.equal(t2));

IMPL_INSTANCE(at::Scalar, at::Scalar(42), at::Scalar(3.14));
template <> struct EqualTrait<at::Scalar> {
  static bool equal(const at::Scalar &t1, const at::Scalar &t2) {
    if (t1.type() != t2.type()) {
      return false;
    }

    if (t1.isComplex()) {
      return t1.toComplexDouble() == t2.toComplexDouble();
    } else if (t1.isFloatingPoint()) {
      return t1.toDouble() == t2.toDouble();
    } else if (t1.isIntegral(/*includeBool=*/false)) {
      return t1.toInt() == t2.toInt();
    } else if (t1.isBoolean()) {
      return t1.toBool() == t2.toBool();
    } else {
      return false;
    }
  }
};

IMPL_INSTANCE(std::vector<at::Tensor>, {}, {at::tensor({42})},
              {at::ones({3, 3}), at::tensor({42}), at::Tensor()});
template <> struct EqualTrait<std::vector<at::Tensor>> {
  static bool equal(const std::vector<at::Tensor> &t1,
                    const std::vector<at::Tensor> &t2) {
    if (t1.size() != t2.size()) {
      return false;
    }

    for (size_t i = 0; i < t1.size(); i++) {
      if (t1[i].defined() != t2[i].defined()) {
        return false;
      }
      if (t1[i].defined() && !EqualTrait<at::Tensor>::equal(t1[i], t2[i])) {
        return false;
      }
    }
    return true;
  }
};
#undef IMPL_INSTANCE
#undef IMPL_EQUAL
#undef IMPL_EQUAL_OPERATOR

template <typename T> std::vector<typename ReturnType<T>::type> instances() {
  return InstancesTrait<T>::instances();
}

template <typename T> T equal(const T &t1, const T &t2) {
  return EqualTrait<T>::equal(t1, t2);
}

template <typename T> class JITTest : public testing::Test {
public:
  using FnTypePtr = T (*)(T);

  JITTest() : jit_(llvm::cantFail(llvm::orc::LLJITBuilder().create())) {}

  void SetUp() override {
    auto ctx = std::make_unique<llvm::LLVMContext>();
    auto mod = std::make_unique<llvm::Module>(ModuleName, *ctx);

    auto entry_fn = llvm::Function::Create(
        ABILLVMFunctionType<FnTypePtr>::get(*ctx),
        llvm::Function::ExternalLinkage, EntryFn, mod.get());

    auto identity_fn = llvm::Function::Create(
        ABILLVMFunctionType<FnTypePtr>::get(*ctx),
        llvm::Function::ExternalLinkage, IdentityFn, mod.get());

    BuildLLVMFunction<typename ABISignature<FnTypePtr>::type>::call(
        *entry_fn, *identity_fn);

    llvm::cantFail(jit_->addIRModule({std::move(mod), std::move(ctx)}));

    llvm::cantFail(jit_->define(llvm::orc::absoluteSymbols(
        {{jit_->mangleAndIntern(IdentityFn),
          llvm::JITEvaluatedSymbol::fromPointer(&identity<T>)}})));

    entry_fn->print(llvm::outs());
    identity_fn->print(llvm::outs());
  }

protected:
  std::unique_ptr<llvm::orc::LLJIT> jit_;
};

TYPED_TEST_SUITE(JITTest, PyTorchTypes);
TYPED_TEST(JITTest, IdentityTest) {
  auto symbol = llvm::cantFail(this->jit_->lookup(EntryFn));
  auto func = (TypeParam(*)(TypeParam))symbol.getAddress();

  for (auto arg : instances<TypeParam>()) {
    auto result = func(arg);
    ASSERT_TRUE(EqualTrait<TypeParam>::equal(arg, result));
  }
}
