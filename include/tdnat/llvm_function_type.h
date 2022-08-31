#pragma once

#include <tdnat/llvm_type.h>
#include <tdnat/utils.h>

#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/LLVMContext.h>

#include <ATen/core/Generator.h>
#include <c10/core/Storage.h>
#include <c10/core/Stream.h>
#include <c10/util/Optional.h>
#include <c10/util/OptionalArrayRef.h>

#include <type_traits>

namespace tdnat {

template <typename T> struct LLVMFunctionType {};

template <typename Return, typename... Args>
struct LLVMFunctionType<Return (*)(Args...)> {
  static llvm::FunctionType *get(llvm::LLVMContext &context) {
    return llvm::FunctionType::get(LLVMType<Return>::get(context),
                                   {LLVMArgType<Args>::get(context)...}, false);
  }
};

template <typename T> struct ABILLVMFunctionType {
  static llvm::FunctionType *get(llvm::LLVMContext &context) {
    return LLVMFunctionType<typename ABIFunctionType<T>::type>::get(context);
  }
};

} // namespace tdnat
