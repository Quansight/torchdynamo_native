#pragma once

#include <tdnat/utils.h>

#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>

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

template <typename T>
llvm::Type *get_or_create_type_by_name(llvm::Module &module);

// Main class for translating PyTorch types into LLVM IR types.
//
// Each of the class specializations listed below have 2 static
// methods and 1 optional static method:
//
//   1. std::string name()
//     Retrieves the string name of the type. This is useful for
//     keeping track of the structs by a unique name.
//
//   2. llvm::Type *get(llvm::Module &module)
//     Retrieves the type to be used in general. This is, for
//     named structs, their name.
//
//   3. (optional) llvm::StructType *create(llvm::Module &module)
//     Creates the struct with its name. This method should be
//     called before get().

template <typename T, typename _Tp = void> struct LLVMType {};

// Types covered: references (T&) and pointers (T*).
template <typename T>
struct LLVMType<T, std::enable_if_t<std::is_pointer<T>::value ||
                                    std::is_reference<T>::value>> {
  using Tval =
      std::remove_cv_t<std::remove_pointer_t<std::remove_reference_t<T>>>;

  static std::string name() { return LLVMType<Tval>::name() + "_ptr"; }

  static llvm::Type *get(llvm::Module &module) {
    return llvm::PointerType::get(LLVMType<Tval>::get(module), 0);
  }
};

// Types covered: integer types without bool
//   - (unsigned) char
//   - (unsigned) short
//   - (unsigned) int
//   - (unsigned) long
//   - (unsigned) long long
template <typename T>
struct LLVMType<T, std::enable_if_t<std::numeric_limits<T>::is_integer &&
                                    !std::is_same<T, bool>::value>> {
  static std::string name() {
    size_t bits = sizeof(T) * 8;
    if (std::is_integral<T>::value) {
      return std::string("i") + std::to_string(bits);
    } else {
      return std::string("u") + std::to_string(bits);
    }
  }

  static llvm::Type *get(llvm::Module &module) {
    return llvm::Type::getIntNTy(module.getContext(), sizeof(T) * 8);
  }
};

template <> struct LLVMType<void> {
  static std::string name() { return "void"; }

  static llvm::Type *get(llvm::Module &module) {
    return llvm::Type::getVoidTy(module.getContext());
  }
};

template <> struct LLVMType<bool> {
  static std::string name() { return "bool"; }

  static llvm::Type *get(llvm::Module &module) {
    return llvm::Type::getInt1Ty(module.getContext());
  }
};

template <> struct LLVMType<double> {
  static std::string name() { return "double"; }

  static llvm::Type *get(llvm::Module &module) {
    return llvm::Type::getDoubleTy(module.getContext());
  }
};

template <> struct LLVMType<at::ScalarType> {
  static std::string name() { return "at::ScalarType"; }

  static llvm::Type *get(llvm::Module &module) {
    return llvm::Type::getInt8Ty(module.getContext());
  }
};

template <> struct LLVMType<at::MemoryFormat> {
  static std::string name() { return "at::MemoryFormat"; }

  static llvm::Type *get(llvm::Module &module) {
    return llvm::Type::getInt8Ty(module.getContext());
  }
};

template <> struct LLVMType<at::Layout> {
  static std::string name() { return "at::Layout"; }

  static llvm::Type *get(llvm::Module &module) {
    return llvm::Type::getInt8Ty(module.getContext());
  }
};

template <> struct LLVMType<at::Tensor> {
  static std::string name() { return "at::Tensor"; }

  static llvm::Type *get(llvm::Module &module) {
    return get_or_create_type_by_name<at::Tensor>(module);
  }

  static llvm::StructType *create(llvm::Module &module) {
    return llvm::StructType::create(
        {llvm::Type::getInt8PtrTy(module.getContext())}, name(), false);
  }
};

template <> struct LLVMType<at::Generator> {
  static std::string name() { return "at::Generator"; }

  static llvm::Type *get(llvm::Module &module) {
    return get_or_create_type_by_name<at::Generator>(module);
  }

  static llvm::StructType *create(llvm::Module &module) {
    return llvm::StructType::create(
        {llvm::Type::getInt8PtrTy(module.getContext())}, name(), false);
  }
};

template <> struct LLVMType<at::Storage> {
  static std::string name() { return "at::Storage"; }

  static llvm::Type *get(llvm::Module &module) {
    return get_or_create_type_by_name<at::Storage>(module);
  }

  static llvm::StructType *create(llvm::Module &module) {
    return llvm::StructType::create(
        {llvm::Type::getInt8PtrTy(module.getContext())}, name(), false);
  }
};

template <> struct LLVMType<at::Scalar> {
  static std::string name() { return "at::Scalar"; }

  static llvm::Type *get(llvm::Module &module) {
    return get_or_create_type_by_name<at::Scalar>(module);
  }

  static llvm::StructType *create(llvm::Module &module) {
    auto &context = module.getContext();
    return llvm::StructType::create(
        {llvm::Type::getInt32Ty(context),
         llvm::ArrayType::get(llvm::Type::getInt8Ty(context),
                              sizeof(double) * 2 - 4),
         llvm::StructType::get(context, {llvm::Type::getDoubleTy(context),
                                         llvm::Type::getDoubleTy(context)})},
        name(), false);
  }
};

template <> struct LLVMType<at::Device> {
  static std::string name() { return "at::Device"; }

  static llvm::Type *get(llvm::Module &module) {
    return get_or_create_type_by_name<at::Device>(module);
  }

  static llvm::StructType *create(llvm::Module &module) {
    auto &context = module.getContext();
    return llvm::StructType::create(
        {llvm::Type::getInt8Ty(context), llvm::Type::getInt8Ty(context)},
        name(), false);
  }
};

template <> struct LLVMType<c10::Stream> {
  static std::string name() { return "c10::Stream"; }

  static llvm::Type *get(llvm::Module &module) {
    return get_or_create_type_by_name<c10::Stream>(module);
  }

  static llvm::StructType *create(llvm::Module &module) {
    auto &context = module.getContext();
    return llvm::StructType::create(
        {LLVMType<c10::Device>::get(module), LLVMType<int64_t>::get(module)},
        name(), false);
  }
};

template <> struct LLVMType<c10::string_view> {
  static std::string name() { return "c10::string_view"; }

  static llvm::Type *get(llvm::Module &module) {
    return get_or_create_type_by_name<c10::string_view>(module);
  }

  static llvm::StructType *create(llvm::Module &module) {
    auto &context = module.getContext();
    return llvm::StructType::create({llvm::Type::getInt8PtrTy(context),
                                     llvm::Type::getScalarTy<size_t>(context)},
                                    name(), false);
  }
};

template <> struct LLVMType<std::vector<at::Tensor>> {
  static std::string name() { return "std::vector<at::Tensor>"; }

  static llvm::Type *get(llvm::Module &module) {
    return get_or_create_type_by_name<std::vector<at::Tensor>>(module);
  }

  static llvm::StructType *create(llvm::Module &module) {
    auto &context = module.getContext();
    return llvm::StructType::create({LLVMType<at::Tensor *>::get(module),
                                     LLVMType<at::Tensor *>::get(module),
                                     LLVMType<at::Tensor *>::get(module)},
                                    name(), false);
  }
};

// Types covered: List<T>
template <typename T> struct LLVMType<c10::List<T>> {
  static std::string name() { return "c10::List<" + LLVMType<T>::name() + ">"; }

  static llvm::Type *get(llvm::Module &module) {
    return get_or_create_type_by_name<c10::List<T>>(module);
  }

  static llvm::StructType *create(llvm::Module &module) {
    return llvm::StructType::create(
        {llvm::Type::getInt8PtrTy(module.getContext())}, name(), false);
  }
};

// Types covered: ArrayRef<T>
template <typename T> struct LLVMType<at::ArrayRef<T>> {
  static std::string name() {
    return std::string("at::ArrayRef<") + LLVMType<T>::name() + ">";
  }

  static llvm::Type *get(llvm::Module &module) {
    return get_or_create_type_by_name<at::ArrayRef<T>>(module);
  }

  static llvm::StructType *create(llvm::Module &module) {
    return llvm::StructType::create(
        {LLVMType<T *>::get(module),
         llvm::Type::getScalarTy<size_t>(module.getContext())},
        name(), false);
  }
};

// Types covered: optional<ArrayRef<T>>
//
// Rationale: optional<ArrayRef<T>> has a storage optimization,
// where it uses the same storage as ArrayRef<T> to represent
// a 'nullopt' value.
template <typename T> struct LLVMType<c10::optional<at::ArrayRef<T>>> {
  static std::string name() {
    return std::string("c10::optional<at::ArrayRef<") + LLVMType<T>::name() +
           ">>";
  }

  static llvm::Type *get(llvm::Module &module) {
    return get_or_create_type_by_name<c10::optional<at::ArrayRef<T>>>(module);
  }

  static llvm::StructType *create(llvm::Module &module) {
    return llvm::StructType::create(
        {llvm::PointerType::get(LLVMType<T>::get(module), 0),
         llvm::Type::getScalarTy<size_t>(module.getContext())},
        name(), false);
  }
};

template <typename T> struct LLVMType<c10::optional<T>> {
  static std::string name() {
    return std::string("c10::optional<") + LLVMType<T>::name() + ">>";
  }

  static llvm::Type *get(llvm::Module &module) {
    return get_or_create_type_by_name<c10::optional<T>>(module);
  }

  static llvm::StructType *create(llvm::Module &module) {
    auto &context = module.getContext();

    llvm::Type *storage_type = nullptr;
    if (sizeof(T) > 1) {
      storage_type = LLVMType<T>::get(module);
    } else {
      storage_type = llvm::Type::getInt8Ty(context);
    }
    return llvm::StructType::create({LLVMType<bool>::get(module), storage_type},
                                    name(), false);
  }
};

template <> struct LLVMType<c10::OptionalArrayRef<long>> {
  static std::string name() { return "c10::OptionalArrayRef<long>"; }

  static llvm::Type *get(llvm::Module &module) {
    return get_or_create_type_by_name<c10::OptionalArrayRef<long>>(module);
  }

  static llvm::Type *create(llvm::Module &module) {
    return llvm::StructType::create(
        {LLVMType<c10::optional<at::ArrayRef<long>>>::get(module)}, name(),
        false);
  }
};

template <typename T>
llvm::Type *get_or_create_type_by_name(llvm::Module &module) {
  auto name = LLVMType<T>::name();
  auto type = module.getTypeByName(name);
  if (type == nullptr) {
    return LLVMType<T>::create(module);
  } else {
    return type;
  }
}

// 1. C++ types classifies as MEMORY by the SysV ABI classifier are
// passed on memory.
//
// In other words, they have to be copied to memory. Not only that
// but, their new memory address must be passed as argument.
//
// 2. POD types smaller than 16-bytes may be coerced into up to 2
// registers. In those cases, the type is treated as an N-bit integer.

template <typename T, typename _Tp = void> struct LLVMArgType {
  static llvm::Type *get(llvm::Module &module) {
    return LLVMType<T>::get(module);
  }
};

template <typename T>
struct LLVMArgType<T, std::enable_if_t<IsABIMemoryClass<T>::value>> {
  static llvm::Type *get(llvm::Module &module) {
    return LLVMType<T *>::get(module);
  }
};

// Covers optional values for 'int8_t' enumerations:
//   - optional<ScalarType>
//   - optional<MemoryFormat>
//   - optional<Layout>
//
// Return type should be i16:
//   - 8 bits for the 'bool' flag.
//   - 8 bits for the enumeration.
template <typename T>
struct LLVMArgType<c10::optional<T>,
                   std::enable_if_t<IsInt8EnumType<T>::value>> {
  static llvm::Type *get(llvm::Module &module) {
    return LLVMType<int16_t>::get(module);
  }
};

// See above (2).

template <typename T, typename _Tp = void> struct LLVMRetType {
  static llvm::Type *get(llvm::Module &module) {
    return LLVMType<T>::get(module);
  }
};

template <typename T>
struct LLVMRetType<c10::optional<T>,
                   std::enable_if_t<IsInt8EnumType<T>::value>> {
  static llvm::Type *get(llvm::Module &module) {
    return LLVMType<int16_t>::get(module);
  }
};

} // namespace tdnat
