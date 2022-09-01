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

#include <iostream>
#include <type_traits>

namespace tdnat {

// Main class for translating PyTorch types into LLVM IR types.

template <typename T, typename _Tp = void> struct LLVMType {};

// Types covered:
// References (T&) and pointers (T*) for the following T:
//   - Tensor
//   - Scalar
template <typename T>
struct LLVMType<T, std::enable_if_t<std::is_pointer<T>::value ||
                                    std::is_reference<T>::value>> {
  static llvm::Type *get(llvm::LLVMContext &context) {
    return llvm::Type::getInt8PtrTy(context);
  }
};

// Types covered: ArrayRef<T>
template <typename T> struct LLVMType<at::ArrayRef<T>> {
  static llvm::Type *get(llvm::LLVMContext &context) {
    return llvm::StructType::get(context,
                                 {llvm::Type::getInt8PtrTy(context),
                                  llvm::Type::getScalarTy<size_t>(context)},
                                 false);
  }
};

#define LLVMTYPE_SPECIALIZE(TYPE, EXPR)                                        \
  template <> struct LLVMType<TYPE> {                                          \
    static llvm::Type *get(llvm::LLVMContext &context) { return EXPR; }        \
  };

LLVMTYPE_SPECIALIZE(void, llvm::Type::getVoidTy(context));

LLVMTYPE_SPECIALIZE(bool, llvm::Type::getInt1Ty(context));

LLVMTYPE_SPECIALIZE(double, llvm::Type::getScalarTy<double>(context));

LLVMTYPE_SPECIALIZE(long, llvm::Type::getScalarTy<long>(context));

LLVMTYPE_SPECIALIZE(at::ScalarType, llvm::Type::getInt8Ty(context));

LLVMTYPE_SPECIALIZE(at::MemoryFormat, llvm::Type::getInt8Ty(context));

LLVMTYPE_SPECIALIZE(at::Layout, llvm::Type::getInt8Ty(context));

LLVMTYPE_SPECIALIZE(at::Tensor,
                    llvm::StructType::get(context,
                                          {llvm::Type::getInt8PtrTy(context)},
                                          false);)

LLVMTYPE_SPECIALIZE(c10::Device,
                    llvm::StructType::get(context,
                                          {llvm::Type::getInt8Ty(context),
                                           llvm::Type::getInt8Ty(context)}));

LLVMTYPE_SPECIALIZE(c10::Stream,
                    llvm::StructType::get(context,
                                          {LLVMType<c10::Device>::get(context),
                                           LLVMType<int64_t>::get(context)}));

LLVMTYPE_SPECIALIZE(
    c10::string_view,
    llvm::StructType::get(context, {llvm::Type::getInt8PtrTy(context),
                                    llvm::Type::getScalarTy<size_t>(context)}));

LLVMTYPE_SPECIALIZE(c10::OptionalArrayRef<long>,
                    LLVMType<at::ArrayRef<long>>::get(context));

LLVMTYPE_SPECIALIZE(std::vector<at::Tensor>,
                    llvm::StructType::get(context,
                                          {llvm::Type::getInt8PtrTy(context),
                                           llvm::Type::getInt8PtrTy(context),
                                           llvm::Type::getInt8PtrTy(context)}));
#undef LLVMTYPE_SPECIALIZE

// Types covered: optional<ArrayRef<T>>
//
// Rationale: optional<ArrayRef<T>> has a storage optimization,
// where it uses the same storage as ArrayRef<T> to represent
// a 'nullopt' value.
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
      storage_type = llvm::Type::getInt8Ty(context);
    }
    return llvm::StructType::get(context,
                                 {LLVMType<bool>::get(context), storage_type});
  }
};

// C++ types classifies as MEMORY by the SysV ABI classifier are
// passed on memory.
//
// In other words, they have to be copied to memory. Not only that
// but, their new memory address must be passed as argument.

template <typename T, typename _Tp = void> struct LLVMArgType {};

template <typename T>
struct LLVMArgType<T, std::enable_if_t<IsABIMemoryClass<T>::value>> {
  static llvm::Type *get(llvm::LLVMContext &context) {
    return LLVMType<T *>::get(context);
  }
};

template <typename T>
struct LLVMArgType<T, std::enable_if_t<!IsABIMemoryClass<T>::value>> {
  static llvm::Type *get(llvm::LLVMContext &context) {
    return LLVMType<T>::get(context);
  }
};

} // namespace tdnat
