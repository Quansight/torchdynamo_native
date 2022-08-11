#pragma once

#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/LLVMContext.h>

#include <ATen/core/Generator.h>
#include <c10/core/Storage.h>
#include <c10/core/Stream.h>
#include <c10/util/Optional.h>
#include <c10/util/OptionalArrayRef.h>

#include <type_traits>

namespace tdnat {

template <typename T, typename _Tp = void>
struct IsInvisibleReturnType : public std::false_type {};

template <typename T>
struct IsInvisibleReturnType<
    T, std::enable_if_t<!std::is_same<T, void>::value &&
                        (!std::is_trivially_copy_constructible<T>::value ||
                         !std::is_trivially_destructible<T>::value)>>
    : public std::true_type {};

template <typename T, typename U = void> struct ToLLVMType {};

template <> struct ToLLVMType<void> {
  static llvm::Type *call(llvm::LLVMContext &context) {
    return llvm::Type::getVoidTy(context);
  }
};

template <> struct ToLLVMType<bool> {
  static llvm::Type *call(llvm::LLVMContext &context) {
    return llvm::Type::getInt1Ty(context);
  }
};

template <typename T>
struct ToLLVMType<T, std::enable_if_t<std::is_pointer<T>::value ||
                                      std::is_reference<T>::value>> {
  static llvm::Type *call(llvm::LLVMContext &context) {
    return llvm::Type::getInt8PtrTy(context);
  }
};

template <typename T>
struct ToLLVMType<T, std::enable_if_t<std::is_integral<T>::value>> {
  static llvm::Type *call(llvm::LLVMContext &context) {
    switch (sizeof(T)) {
    case 1:
      return llvm::Type::getInt8Ty(context);
    case 2:
      return llvm::Type::getInt16Ty(context);
    case 4:
      return llvm::Type::getInt32Ty(context);
    case 8:
      return llvm::Type::getInt64Ty(context);
    default:
      TORCH_INTERNAL_ASSERT(false, "no support for integer of more than ",
                            sizeof(T), " bytes")
    }
  }
};

template <typename T>
struct ToLLVMType<T, std::enable_if_t<std::is_floating_point<T>::value>> {
  static llvm::Type *call(llvm::LLVMContext &context) {
    switch (sizeof(T)) {
    case 2:
      return llvm::Type::getHalfTy(context);
    case 4:
      return llvm::Type::getFloatTy(context);
    case 8:
      return llvm::Type::getDoubleTy(context);
    default:
      TORCH_INTERNAL_ASSERT(false,
                            "no support for floating types of more than ",
                            sizeof(T), "bytes");
    }
  }
};

template <typename T>
struct ToLLVMType<T, std::enable_if_t<IsInvisibleReturnType<T>::value ||
                                      std::is_same<T, c10::Scalar>::value>> {
  static llvm::Type *call(llvm::LLVMContext &context) {
    return ToLLVMType<uint8_t *>::call(context);
  }
};

#define TOLLVMTYPE_INT8(TYPE)                                                  \
  template <> struct ToLLVMType<TYPE> {                                        \
    static llvm::Type *call(llvm::LLVMContext &context) {                      \
      return ToLLVMType<int8_t>::call(context);                                \
    }                                                                          \
  };
TOLLVMTYPE_INT8(at::ScalarType);
TOLLVMTYPE_INT8(c10::Layout);
TOLLVMTYPE_INT8(c10::MemoryFormat);
#undef TOLLVMTYPE_INT8

template <> struct ToLLVMType<c10::Device> {
  static llvm::Type *call(llvm::LLVMContext &context) {
    return llvm::StructType::get(
        context,
        {ToLLVMType<int8_t>::call(context), ToLLVMType<int8_t>::call(context)},
        false);
  }
};

template <> struct ToLLVMType<c10::Stream> {
  static llvm::Type *call(llvm::LLVMContext &context) {
    return llvm::StructType::get(context,
                                 {ToLLVMType<c10::Device>::call(context),
                                  ToLLVMType<int64_t>::call(context)},
                                 false);
  }
};

template <> struct ToLLVMType<c10::basic_string_view<char>> {
  static llvm::Type *call(llvm::LLVMContext &context) {
    return llvm::StructType::get(
        context,
        {ToLLVMType<char *>::call(context), ToLLVMType<size_t>::call(context)},
        false);
  }
};

template <typename T> struct ToLLVMType<at::ArrayRef<T>> {
  static llvm::Type *call(llvm::LLVMContext &context) {
    return llvm::StructType::get(
        context,
        {ToLLVMType<T *>::call(context), ToLLVMType<size_t>::call(context)},
        false);
  }
};

template <typename T> struct ToLLVMType<c10::optional<at::ArrayRef<T>>> {
  static llvm::Type *call(llvm::LLVMContext &context) {
    return ToLLVMType<at::ArrayRef<T>>::call(context);
  }
};

template <typename T>
struct ToLLVMType<
    c10::optional<T>,
    std::enable_if_t<!IsInvisibleReturnType<T>::value &&
                     !c10::detail_::is_arrayref<T>::value &&
                     !std::is_same<T, c10::basic_string_view<char>>::value>> {
  static llvm::Type *call(llvm::LLVMContext &context) {
    llvm::Type *storage_type = nullptr;
    if (sizeof(T) > 1) {
      storage_type = ToLLVMType<T>::call(context);
    } else {
      storage_type = ToLLVMType<unsigned char>::call(context);
    }
    return llvm::StructType::get(
        context, {ToLLVMType<bool>::call(context), storage_type}, false);
  }
};

template <typename T> struct ToLLVMType<c10::OptionalArrayRef<T>> {
  static llvm::Type *call(llvm::LLVMContext &context) {
    return ToLLVMType<c10::optional<at::ArrayRef<T>>>::call(context);
  }
};

template <typename T, typename X = void> struct ABITrait {};

template <typename Return, typename... Args>
struct ABITrait<
    Return (*)(Args...),
    typename std::enable_if<IsInvisibleReturnType<Return>::value, void>::type> {
  using type = void (*)(Return, Args...);
};

template <typename Return, typename... Args>
struct ABITrait<Return (*)(Args...),
                typename std::enable_if<!IsInvisibleReturnType<Return>::value,
                                        void>::type> {
  using type = Return (*)(Args...);
};

template <typename T> struct ToLLVMFunctionType {};

template <typename Return, typename... Args>
struct ToLLVMFunctionType<Return (*)(Args...)> {
  static llvm::FunctionType *call(llvm::LLVMContext &context) {
    return llvm::FunctionType::get(ToLLVMType<Return>::call(context),
                                   {ToLLVMType<Args>::call(context)...}, false);
  }
};

} // namespace tdnat
