#include <gtest/gtest.h>

#include <sstream>
#include <tdnat/llvm_function_type.h>

#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/Support/raw_os_ostream.h>

#include <cstdint>
#include <memory>

#include "test_utils.h"

using namespace tdnat;

template <typename T> class LLVMTypeTest : public testing::Test {
public:
  LLVMTypeTest() : context_(new llvm::LLVMContext()) {}

protected:
  std::unique_ptr<llvm::LLVMContext> context_;
};

// Wraps around llvm::Type* for adding functionality.
// In fact, we only need to re-define operator<<, for improved debugging.
struct LLVMTypeWrapper {
public:
  LLVMTypeWrapper() : type_(nullptr) {}
  LLVMTypeWrapper(llvm::Type *type) : type_(type) {}

  llvm::Type *get() const { return type_; }

  void print(std::ostream &os) const {
    if (type_ == nullptr) {
      os << "nullptr";
    } else {
      llvm::raw_os_ostream raw_os(os);
      type_->print(raw_os);
    }
  }

  bool operator==(const LLVMTypeWrapper &rhs) const {
    return type_ == rhs.type_;
  }

private:
  llvm::Type *type_;
};

std::ostream &operator<<(std::ostream &os, const LLVMTypeWrapper &type) {
  type.print(os);
  return os;
}

// These can be functions, since we are fully specializing them for
// each tested type.
template <typename T>
llvm::Type *llvm_expected_type(llvm::LLVMContext &context) {
  return {};
}

template <typename T>
llvm::Type *llvm_expected_arg_type(llvm::LLVMContext &context) {
  return {};
}

#define LLVM_EXPECTED_TYPE(TYPE, EXPR)                                         \
  template <>                                                                  \
  llvm::Type *llvm_expected_type<TYPE>(llvm::LLVMContext & context) {          \
    return EXPR;                                                               \
  }

#define LLVM_EXPECTED_ARG_TYPE(TYPE, EXPR)                                     \
  template <>                                                                  \
  llvm::Type *llvm_expected_arg_type<TYPE>(llvm::LLVMContext & context) {      \
    return EXPR;                                                               \
  }

#define LLVM_EXPECTED_ARG_TYPE_ALIAS(TYPE)                                     \
  template <>                                                                  \
  llvm::Type *llvm_expected_arg_type<TYPE>(llvm::LLVMContext & context) {      \
    return llvm_expected_type<TYPE>(context);                                  \
  }

// =============================================================================
// bool
// =============================================================================
LLVM_EXPECTED_TYPE(bool, llvm::Type::getInt1Ty(context));
LLVM_EXPECTED_ARG_TYPE_ALIAS(bool);

// c10::optional<bool>
LLVM_EXPECTED_TYPE(c10::optional<bool>,
                   llvm::StructType::get(context,
                                         {llvm_expected_type<bool>(context),
                                          llvm::Type::getInt8Ty(context)}));
LLVM_EXPECTED_ARG_TYPE_ALIAS(c10::optional<bool>);

// bool (*)(bool)
LLVM_EXPECTED_TYPE(bool (*)(bool),
                   llvm::FunctionType::get(llvm_expected_type<bool>(context),
                                           {llvm_expected_type<bool>(context)},
                                           false));

// =============================================================================
// int64_t
// =============================================================================
LLVM_EXPECTED_TYPE(int64_t, llvm::Type::getInt64Ty(context));
LLVM_EXPECTED_ARG_TYPE_ALIAS(int64_t);

// c10::optional<int64_t>
LLVM_EXPECTED_TYPE(
    c10::optional<int64_t>,
    llvm::StructType::get(context, {llvm_expected_type<bool>(context),
                                    llvm_expected_type<int64_t>(context)}));
LLVM_EXPECTED_ARG_TYPE_ALIAS(c10::optional<int64_t>);

// int64_t (*)(int64_t)
LLVM_EXPECTED_TYPE(
    int64_t (*)(int64_t),
    llvm::FunctionType::get(llvm_expected_type<int64_t>(context),
                            {llvm_expected_type<int64_t>(context)}, false));

// =============================================================================
// at::ArrayRef<int64_t>
// =============================================================================
LLVM_EXPECTED_TYPE(at::ArrayRef<int64_t>,
                   llvm::StructType::get(
                       context,
                       {llvm::Type::getInt8PtrTy(context),
                        llvm::Type::getIntNTy(context, sizeof(size_t) * 8)}));
LLVM_EXPECTED_ARG_TYPE_ALIAS(at::ArrayRef<int64_t>);

// c10::optional<at::ArrayRef<int64_t>>
LLVM_EXPECTED_TYPE(c10::optional<at::ArrayRef<int64_t>>,
                   llvm_expected_type<at::ArrayRef<int64_t>>(context));
LLVM_EXPECTED_ARG_TYPE_ALIAS(c10::optional<at::ArrayRef<int64_t>>);

// at::ArrayRef<int64_t> (*)(at::ArrayRef<int64_t>)
LLVM_EXPECTED_TYPE(at::ArrayRef<int64_t> (*)(at::ArrayRef<int64_t>),
                   llvm::FunctionType::get(
                       llvm_expected_type<at::ArrayRef<int64_t>>(context),
                       {llvm_expected_type<at::ArrayRef<int64_t>>(context)},
                       false));

// =============================================================================
// at::Tensor
// =============================================================================
LLVM_EXPECTED_TYPE(at::Tensor, llvm::Type::getInt8PtrTy(context));
LLVM_EXPECTED_ARG_TYPE_ALIAS(at::Tensor);

// c10::optional<at::Tensor>
LLVM_EXPECTED_TYPE(
    c10::optional<at::Tensor>,
    llvm::StructType::get(context, {llvm_expected_type<bool>(context),
                                    llvm_expected_type<at::Tensor>(context)}));
LLVM_EXPECTED_ARG_TYPE(c10::optional<at::Tensor>,
                       llvm::Type::getInt8PtrTy(context));

// at::Tensor (*)(at::Tensor)
LLVM_EXPECTED_TYPE(
    at::Tensor (*)(at::Tensor),
    llvm::FunctionType::get(llvm::Type::getVoidTy(context),
                            {llvm::Type::getInt8PtrTy(context),
                             llvm_expected_type<at::Tensor>(context)},
                            false));

// =============================================================================
// at::Scalar
// =============================================================================
LLVM_EXPECTED_TYPE(
    at::Scalar,
    llvm::StructType::get(
        context,
        {llvm::Type::getIntNTy(context, sizeof(int) * 8),
         llvm::StructType::get(context, {llvm::Type::getDoubleTy(context),
                                         llvm::Type::getDoubleTy(context)})}));
LLVM_EXPECTED_ARG_TYPE(at::Scalar, llvm::Type::getInt8PtrTy(context));

// c10::optional<at::Scalar>
LLVM_EXPECTED_TYPE(
    c10::optional<at::Scalar>,
    llvm::StructType::get(context, {llvm_expected_type<bool>(context),
                                    llvm_expected_type<at::Scalar>(context)}));
LLVM_EXPECTED_ARG_TYPE(c10::optional<at::Scalar>,
                       llvm::Type::getInt8PtrTy(context));

// at::Scalar (*)(at::Scalar)
LLVM_EXPECTED_TYPE(at::Scalar (*)(at::Scalar),
                   llvm::FunctionType::get(llvm::Type::getVoidTy(context),
                                           {llvm::Type::getInt8PtrTy(context),
                                            llvm::Type::getInt8PtrTy(context)},
                                           false));

// =============================================================================
// std::vector<at::Tensor>
// =============================================================================
LLVM_EXPECTED_TYPE(std::vector<at::Tensor>,
                   llvm::StructType::get(context,
                                         {llvm::Type::getInt8PtrTy(context),
                                          llvm::Type::getInt8PtrTy(context),
                                          llvm::Type::getInt8PtrTy(context)}));
LLVM_EXPECTED_ARG_TYPE(std::vector<at::Tensor>,
                       llvm::Type::getInt8PtrTy(context));

// c10::optional<std::vector<at::Tensor>>
LLVM_EXPECTED_TYPE(
    c10::optional<std::vector<at::Tensor>>,
    llvm::StructType::get(
        context,
        {llvm_expected_type<bool>(context),
         llvm::StructType::get(context, {llvm::Type::getInt8PtrTy(context),
                                         llvm::Type::getInt8PtrTy(context),
                                         llvm::Type::getInt8PtrTy(context)})}));
LLVM_EXPECTED_ARG_TYPE(c10::optional<std::vector<at::Tensor>>,
                       llvm::Type::getInt8PtrTy(context));

// std::vector<at::Tensor> (*)(std::vector<at::Tensor>)
LLVM_EXPECTED_TYPE(std::vector<at::Tensor> (*)(std::vector<at::Tensor>),
                   llvm::FunctionType::get(llvm::Type::getVoidTy(context),
                                           {llvm::Type::getInt8PtrTy(context),
                                            llvm::Type::getInt8PtrTy(context)},
                                           false));

#undef LLVM_EXPECTED_TYPE
#undef LLVM_EXPECTED_ARG_TYPE
#undef LLVM_EXPECTED_ARG_TYPE_ALIAS

TYPED_TEST_SUITE(LLVMTypeTest, PyTorchTypes);
TYPED_TEST(LLVMTypeTest, SimpleTypes) {
  auto &context = *this->context_;

  LLVMTypeWrapper ty = LLVMType<TypeParam>::get(context);
  LLVMTypeWrapper ex_ty = llvm_expected_type<TypeParam>(context);
  ASSERT_EQ(ty, ex_ty);

  LLVMTypeWrapper opt_ty = LLVMType<c10::optional<TypeParam>>::get(context);
  LLVMTypeWrapper opt_ex_ty =
      llvm_expected_type<c10::optional<TypeParam>>(context);
  ASSERT_EQ(opt_ty, opt_ex_ty);
}

TYPED_TEST(LLVMTypeTest, ArgumentTypes) {
  auto &context = *this->context_;

  LLVMTypeWrapper ty = LLVMArgType<TypeParam>::get(context);
  LLVMTypeWrapper ex_ty = llvm_expected_arg_type<TypeParam>(context);
  ASSERT_EQ(ty, ex_ty);

  LLVMTypeWrapper opt_ty = LLVMArgType<c10::optional<TypeParam>>::get(context);
  LLVMTypeWrapper opt_ex_ty =
      llvm_expected_arg_type<c10::optional<TypeParam>>(context);
  ASSERT_EQ(opt_ty, opt_ex_ty);
}

TYPED_TEST(LLVMTypeTest, IdentityFunctionTest) {
  auto &context = *this->context_;

  auto fn_ty = ABILLVMFunctionType<TypeParam (*)(TypeParam)>::get(context);
  auto ex_ty = llvm_expected_type<TypeParam (*)(TypeParam)>(context);

  ASSERT_TRUE(llvm::isa<llvm::FunctionType>(ex_ty));
  auto exfn_ty = llvm::dyn_cast<llvm::FunctionType>(ex_ty);

  ASSERT_EQ(fn_ty->isVarArg(), exfn_ty->isVarArg());
  ASSERT_EQ(fn_ty->getNumParams(), exfn_ty->getNumParams());

  ASSERT_EQ(LLVMTypeWrapper{fn_ty}, LLVMTypeWrapper{exfn_ty});
}
