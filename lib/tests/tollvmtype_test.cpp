#include <gtest/gtest.h>

#include <cstdint>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/LLVMContext.h>
#include <memory>

#include <tdnat/llvm.h>

#include "types.h"
#include "llvm/Support/Casting.h"

using namespace tdnat;

template <typename T> class ToLLVMTypeTest : public testing::Test {
public:
  ToLLVMTypeTest() : context_(new llvm::LLVMContext()) {}

protected:
  std::unique_ptr<llvm::LLVMContext> context_;
};

template <typename T>
llvm::Type *llvm_expected_type(llvm::LLVMContext &context) {
  return nullptr;
}

#define LLVM_EXPECTED_PTR_TYPE(TYPE)                                           \
  template <>                                                                  \
  llvm::Type *llvm_expected_type<TYPE>(llvm::LLVMContext & context) {          \
    return llvm::Type::getInt8PtrTy(context);                                  \
  }
SUPPORTED_PTR_TYPES(LLVM_EXPECTED_PTR_TYPE);
#undef LLVM_EXPECTED_PTR_TYPE

#define LLVM_EXPECTED_TYPE(TYPE, EXPR)                                         \
  template <>                                                                  \
  llvm::Type *llvm_expected_type<TYPE>(llvm::LLVMContext & context) {          \
    return EXPR;                                                               \
  }
LLVM_EXPECTED_TYPE(bool, llvm::Type::getInt1Ty(context));
LLVM_EXPECTED_TYPE(int8_t *, llvm::Type::getInt8PtrTy(context));
LLVM_EXPECTED_TYPE(size_t, llvm::Type::getIntNTy(context, sizeof(size_t) * 8));
LLVM_EXPECTED_TYPE(int64_t, llvm::Type::getInt64Ty(context));
LLVM_EXPECTED_TYPE(double, llvm::Type::getDoubleTy(context));
LLVM_EXPECTED_TYPE(at::IntArrayRef,
                   llvm::StructType::get(context,
                                         {llvm_expected_type<int8_t *>(context),
                                          llvm_expected_type<size_t>(context)},
                                         false));
LLVM_EXPECTED_TYPE(c10::optional<at::IntArrayRef>,
                   llvm_expected_type<at::IntArrayRef>(context));
#undef LLVM_EXPECTED_TYPE

#define LLVM_EXPECTED_OPT_TYPE(TYPE)                                           \
  template <>                                                                  \
  llvm::Type *llvm_expected_type<c10::optional<TYPE>>(llvm::LLVMContext &      \
                                                      context) {               \
    return llvm::StructType::get(context,                                      \
                                 {llvm_expected_type<bool>(context),           \
                                  llvm_expected_type<TYPE>(context)},          \
                                 false);                                       \
  }
LLVM_EXPECTED_OPT_TYPE(int64_t);
LLVM_EXPECTED_OPT_TYPE(double);
#undef LLVM_EXPECTED_OPT_TYPE

#define IDENTITY_FN_TYPE_VALUE(TYPE)                                           \
  template <>                                                                  \
  llvm::Type *llvm_expected_type<TYPE (*)(TYPE)>(llvm::LLVMContext &           \
                                                 context) {                    \
    return llvm::FunctionType::get(llvm_expected_type<TYPE>(context),          \
                                   {llvm_expected_type<TYPE>(context)},        \
                                   false);                                     \
  }
SUPPORTED_VALUE_TYPES(IDENTITY_FN_TYPE_VALUE);
IDENTITY_FN_TYPE_VALUE(c10::optional<long>);
IDENTITY_FN_TYPE_VALUE(c10::optional<double>);
IDENTITY_FN_TYPE_VALUE(c10::optional<at::ArrayRef<long>>);
#undef IDENTITY_FN_TYPE_VALUE

#define IDENTITY_FN_TYPE_PTR(TYPE)                                             \
  template <>                                                                  \
  llvm::Type *llvm_expected_type<TYPE (*)(TYPE)>(llvm::LLVMContext &           \
                                                 context) {                    \
    return llvm::FunctionType::get(llvm::Type::getVoidTy(context),             \
                                   {llvm_expected_type<TYPE>(context),         \
                                    llvm_expected_type<TYPE>(context)},        \
                                   false);                                     \
  }
SUPPORTED_PTR_TYPES(IDENTITY_FN_TYPE_PTR);
#undef IDENTITY_FN_TYPE_PTR

TYPED_TEST_SUITE(ToLLVMTypeTest, PyTorchTypes);
TYPED_TEST(ToLLVMTypeTest, IdentityTest) {
  auto &context = *this->context_;

  auto ty = ToLLVMType<TypeParam>::call(context);
  auto ex_ty = llvm_expected_type<TypeParam>(context);

  ASSERT_EQ(ty, ex_ty);
}

TYPED_TEST(ToLLVMTypeTest, IdentityFunctionTest) {
  auto &context = *this->context_;

  auto fn_ty = ToLLVMFunctionType<typename ABITrait<TypeParam (*)(TypeParam)>::type>::call(context);
  auto ex_ty = llvm_expected_type<TypeParam (*)(TypeParam)>(context);

  ASSERT_TRUE(llvm::isa<llvm::FunctionType>(*ex_ty));
  auto exfn_ty = llvm::dyn_cast<llvm::FunctionType>(ex_ty);

  ASSERT_EQ(fn_ty->isVarArg(), exfn_ty->isVarArg());
  ASSERT_EQ(fn_ty->getNumParams(), exfn_ty->getNumParams());
  ASSERT_EQ(fn_ty, exfn_ty);
}
