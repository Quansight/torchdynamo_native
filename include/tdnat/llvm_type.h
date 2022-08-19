#pragma once

#include <tdnat/utils.h>

#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/LLVMContext.h>

#include <ATen/core/Generator.h>
#include <ATen/core/List.h>
#include <c10/core/Layout.h>
#include <c10/core/MemoryFormat.h>
#include <c10/core/Scalar.h>
#include <c10/core/ScalarType.h>
#include <c10/core/Storage.h>
#include <c10/core/Stream.h>
#include <c10/util/Optional.h>
#include <c10/util/OptionalArrayRef.h>

#include <type_traits>
#include <iostream>

namespace tdnat {

template <typename T, typename _Tp = void> struct LLVMType {};

template <> struct LLVMType<bool> {
  static llvm::Type *get(llvm::LLVMContext &context) {
    return llvm::Type::getInt1Ty(context);
  }
};

// Types covered:
//   - int64_t
//   - double
template <typename T>
struct LLVMType<T, std::enable_if_t<std::is_arithmetic<T>::value>> {
  static llvm::Type *get(llvm::LLVMContext &context) {
    return llvm::Type::getScalarTy<T>(context);
  }
};

// Types covered:
//   - References: T&
//   - Pointers: T*
template <typename T>
struct LLVMType<T, std::enable_if_t<std::is_pointer<T>::value ||
                                    std::is_reference<T>::value>> {
  static llvm::Type *get(llvm::LLVMContext &context) {
    return llvm::Type::getInt8PtrTy(context);
  }
};

#define LLVMTYPE_SPECIALIZE(TYPE, EXPR)                                        \
  template <> struct LLVMType<TYPE> {                                          \
    static llvm::Type *get(llvm::LLVMContext &context) { return EXPR; }        \
  };
LLVMTYPE_SPECIALIZE(void, llvm::Type::getVoidTy(context));

// Enums that inherit from int8_t.
LLVMTYPE_SPECIALIZE(at::ScalarType, LLVMType<int8_t>::get(context));
LLVMTYPE_SPECIALIZE(at::MemoryFormat, LLVMType<int8_t>::get(context));
LLVMTYPE_SPECIALIZE(at::Layout, LLVMType<int8_t>::get(context));

// Wrappers to intrusive_ptr.
LLVMTYPE_SPECIALIZE(at::Tensor, LLVMType<int8_t *>::get(context));
LLVMTYPE_SPECIALIZE(at::Generator, LLVMType<int8_t *>::get(context));
LLVMTYPE_SPECIALIZE(at::Storage, LLVMType<int8_t *>::get(context));

LLVMTYPE_SPECIALIZE(c10::Device,
                    llvm::StructType::get(context,
                                          {llvm::Type::getInt8Ty(context),
                                           llvm::Type::getInt8Ty(context)}));

LLVMTYPE_SPECIALIZE(c10::Stream,
                    llvm::StructType::get(context,
                                          {LLVMType<c10::Device>::get(context),
                                           LLVMType<int64_t>::get(context)}));

LLVMTYPE_SPECIALIZE(c10::string_view,
                    llvm::StructType::get(context,
                                          {LLVMType<char *>::get(context),
                                           LLVMType<size_t>::get(context)}));

LLVMTYPE_SPECIALIZE(
    c10::Scalar,
    llvm::StructType::get(
        context,
        {LLVMType<int32_t>::get(context),
         llvm::StructType::get(context, {LLVMType<double>::get(context),
                                         LLVMType<double>::get(context)})}));
#undef LLVMTYPE_SPECIALIZE

template <typename T> struct LLVMType<at::ArrayRef<T>> {
  static llvm::Type *get(llvm::LLVMContext &context) {
    return llvm::StructType::get(
        context, {LLVMType<T *>::get(context), LLVMType<size_t>::get(context)},
        false);
  }
};

template <typename T> struct LLVMType<c10::optional<at::ArrayRef<T>>> {
  static llvm::Type *get(llvm::LLVMContext &context) {
    return LLVMType<at::ArrayRef<T>>::get(context);
  }
};

template <typename T> struct LLVMType<c10::optional<T>> {
  static llvm::Type *get(llvm::LLVMContext &context) {
    llvm::Type *storage_type = nullptr;
    if (sizeof(T) > 1) {
      storage_type = LLVMType<T>::get(context);
    } else {
      storage_type = LLVMType<unsigned char>::get(context);
    }
    return llvm::StructType::get(context,
                                 {LLVMType<bool>::get(context), storage_type});
  }
};

template <typename T> struct LLVMType<c10::OptionalArrayRef<T>> {
  static llvm::Type *get(llvm::LLVMContext &context) {
    return LLVMType<c10::optional<at::ArrayRef<T>>>::get(context);
  }
};

template <typename T> struct LLVMType<std::vector<T>> {
  static llvm::Type *get(llvm::LLVMContext &context) {
    return llvm::StructType::get(context, {LLVMType<T *>::get(context),
                                           LLVMType<T *>::get(context),
                                           LLVMType<T *>::get(context)});
  }
};

template <typename T, typename _Tp = void> struct LLVMArgType {};

template <typename T>
struct LLVMArgType<T, std::enable_if_t<ReturnsOnMemory<T>::value>> {
  static llvm::Type *get(llvm::LLVMContext &context) {
    return LLVMType<T *>::get(context);
  }
};

template <typename T>
struct LLVMArgType<T, std::enable_if_t<!ReturnsOnMemory<T>::value>> {
  static llvm::Type *get(llvm::LLVMContext &context) {
    return LLVMType<T>::get(context);
  }
};
/*
template <typename T, typename _Tp = void> struct LLVMTrait {};

template <typename T> struct ArgTypeCallsType {
  static llvm::Type *arg_type(llvm::LLVMContext &context) {
    return LLVMTrait<T>::type(context);
  }
};

template <typename T> struct ArgTypeReturnsPointer {
  static llvm::Type *arg_type(llvm::LLVMContext &context) {
    return LLVMTrait<T *>::type(context);
  }
};

template <typename T>
struct LLVMTrait<T, std::enable_if_t<std::is_arithmetic<T>::value>>
    : public ArgTypeCallsType<T> {
  static llvm::Type *type(llvm::LLVMContext &context) {
    return llvm::Type::getScalarTy<T>(context);
  }
};

template <typename T>
struct LLVMTrait<T, std::enable_if_t<std::is_pointer<T>::value ||
                                     std::is_reference<T>::value>>
    : public ArgTypeCallsType<T> {
  static llvm::Type *type(llvm::LLVMContext &context) {
    return llvm::Type::getInt8PtrTy(context);
  }
};

#define LLVM_TRAIT_ARGTYPE_IS_ALIAS(TYPE, EXPR)                                \
  template <> struct LLVMTrait<TYPE> : public ArgTypeCallsType<TYPE> {         \
    static llvm::Type *type(llvm::LLVMContext &context) { return EXPR; }       \
  };
LLVM_TRAIT_ARGTYPE_IS_ALIAS(bool, llvm::Type::getInt1Ty(context));
LLVM_TRAIT_ARGTYPE_IS_ALIAS(void, llvm::Type::getVoidTy(context));
LLVM_TRAIT_ARGTYPE_IS_ALIAS(at::ScalarType, LLVMTrait<int8_t>::type(context));
LLVM_TRAIT_ARGTYPE_IS_ALIAS(at::MemoryFormat, LLVMTrait<int8_t>::type(context));
LLVM_TRAIT_ARGTYPE_IS_ALIAS(c10::Layout, LLVMTrait<int8_t>::type(context));
LLVM_TRAIT_ARGTYPE_IS_ALIAS(at::Tensor, LLVMTrait<int8_t *>::type(context));
LLVM_TRAIT_ARGTYPE_IS_ALIAS(at::Generator, LLVMTrait<int8_t *>::type(context));
LLVM_TRAIT_ARGTYPE_IS_ALIAS(c10::Storage, LLVMTrait<int8_t *>::type(context));
LLVM_TRAIT_ARGTYPE_IS_ALIAS(
    c10::Device,
    llvm::StructType::get(context, {llvm::Type::getInt8Ty(context),
                                    llvm::Type::getInt8Ty(context)}));
LLVM_TRAIT_ARGTYPE_IS_ALIAS(
    c10::Stream,
    llvm::StructType::get(context, {LLVMTrait<c10::Device>::type(context),
                                    LLVMTrait<int64_t>::type(context)}));
LLVM_TRAIT_ARGTYPE_IS_ALIAS(
    c10::string_view,
    llvm::StructType::get(context, {LLVMTrait<char *>::type(context),
                                    LLVMTrait<size_t>::type(context)}));
#undef LLVM_TRAIT_ARGTYPE_IS_ALIAS

template <typename T>
struct LLVMTrait<c10::OptionalArrayRef<T>>
    : public ArgTypeCallsType<c10::OptionalArrayRef<T>> {
  static llvm::Type *type(llvm::LLVMContext &context) {
    return LLVMTrait<c10::optional<at::ArrayRef<T>>>::type(context);
  }
};

template <typename T>
struct LLVMTrait<at::ArrayRef<T>> : public ArgTypeCallsType<at::ArrayRef<T>> {
  static llvm::Type *type(llvm::LLVMContext &context) {
    return llvm::StructType::get(
        context,
        {LLVMTrait<T *>::type(context), LLVMTrait<size_t>::type(context)},
        false);
  }
};

template <typename T>
struct LLVMTrait<c10::optional<at::ArrayRef<T>>>
    : public ArgTypeCallsType<c10::optional<at::ArrayRef<T>>> {
  static llvm::Type *type(llvm::LLVMContext &context) {
    return LLVMTrait<at::ArrayRef<T>>::type(context);
  }
};

template <typename T>
struct LLVMTrait<c10::optional<T>,
                 std::enable_if_t<!IsInvisibleReturnType<T>::value &&
                                  !c10::detail_::is_arrayref<T>::value>>
    : public OptionalTypeLLVMTraitBase<T>,
      public ArgTypeCallsType<c10::optional<T>> {};

template <typename T>
struct LLVMTrait<c10::optional<T>,
                 std::enable_if_t<IsInvisibleReturnType<T>::value &&
                                  !c10::detail_::is_arrayref<T>::value>>
    : public OptionalTypeLLVMTraitBase<T>,
      public ArgTypeReturnsPointer<c10::optional<T>> {};

template <typename T>
struct LLVMTrait<std::vector<T>>
    : public ArgTypeReturnsPointer<std::vector<T>> {
  static llvm::Type *type(llvm::LLVMContext &context) {
    return llvm::StructType::get(context, {LLVMTrait<T *>::type(context),
                                           LLVMTrait<T *>::type(context),
                                           LLVMTrait<T *>::type(context)});
  }
};

template <>
struct LLVMTrait<c10::Scalar> : public ArgTypeReturnsPointer<c10::Scalar> {
  static llvm::Type *type(llvm::LLVMContext &context) {
    return llvm::StructType::get(
        context,
        {LLVMTrait<int32_t>::type(context),
         llvm::StructType::get(context, {LLVMTrait<double>::type(context),
                                         LLVMTrait<double>::type(context)})});
  }
};

template <typename T> struct OptionalTypeLLVMTraitBase {
  static llvm::Type *type(llvm::LLVMContext &context) {
    llvm::Type *storage_type = nullptr;
    if (sizeof(T) > 1) {
      storage_type = LLVMTrait<T>::type(context);
    } else {
      storage_type = LLVMTrait<unsigned char>::type(context);
    }
    return llvm::StructType::get(
        context, {LLVMTrait<bool>::type(context), storage_type}, false);
  }
};
*/
} // namespace tdnat
