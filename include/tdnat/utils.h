#pragma once

#include <ATen/core/TensorBody.h>
#include <c10/core/Layout.h>
#include <c10/core/MemoryFormat.h>
#include <c10/core/Scalar.h>
#include <c10/util/Optional.h>

#include <type_traits>
#include <vector>

namespace tdnat
{

template <typename T>
struct IsOptionalType : public std::false_type {
};
template <typename T>
struct IsOptionalType<c10::optional<T>> : public std::true_type {
};

template <typename T>
struct IsInt8EnumType : public std::false_type {
};
template <>
struct IsInt8EnumType<at::ScalarType> : public std::true_type {
};
template <>
struct IsInt8EnumType<at::MemoryFormat> : public std::true_type {
};
template <>
struct IsInt8EnumType<at::Layout> : public std::true_type {
};

// C++ types that are classified as MEMORY class in the SystemV ABI.
// Some of the necessary conditions are:
//   - Different than void
//   - Trivially copy constructible
//   - Trivially destructible
//
// c10::Scalar, for example, is also classified as MEMORY.
// However, that is because of its size.

template <typename T, typename Tp = void>
struct IsABIMemoryClass : public std::false_type {
};

template <>
struct IsABIMemoryClass<at::Scalar> : public std::true_type {
};

template <>
struct IsABIMemoryClass<c10::optional<c10::string_view>> : public std::true_type {
};

template <typename T>
struct IsABIMemoryClass<c10::optional<T>, std::enable_if_t<IsABIMemoryClass<T>::value>>
    : public std::true_type {
};

template <typename T>
struct IsABIMemoryClass<
    T,
    std::enable_if_t<
        !std::is_same<T, void>::value && !IsOptionalType<T>::value &&
        (!std::is_trivially_copy_constructible<T>::value ||
         !std::is_trivially_destructible<T>::value)>> : public std::true_type {
};

// Types in the MEMORY class can be passed as argument in 2 different ways:
//
//   1. Passing the pointer address in a register. Observed when the aggregate
//      type has either a non-trivial constructor or destructor.
//
//   2. Copying the whole data to the stack. Observed when the aggregate type
//      is bigger than 2 registers (on System V ABI).
//
// Not entirely sure why that is the case, but that's how GCC and LLVM
// appear to work.

template <typename T>
struct IsCopiedToStack : std::false_type {
};

// at::Scalar is the only type used in PyTorch operations, that is bigger
// than 2 registers ("eightbytes").
template <>
struct IsCopiedToStack<at::Scalar> : std::true_type {
};

// Some function types are translated slightly different for the
// SystemV ABI.
//
// The type of functions whose return value is classified as MEMORY has
// the following changes:
//
//   - Adds an invisible 1st parameter: a memory address from the
//     caller stack frame
//
//   - Returns the address of the return value (i.e. same as the 1st
//     argument)

template <typename T, typename Tp = void>
struct ABIFunctionType {
};

template <typename Return, typename... Args>
struct ABIFunctionType<Return (*)(Args...), std::enable_if_t<IsABIMemoryClass<Return>::value>> {
  using type = void (*)(Return *, Args...);
};

template <typename Return, typename... Args>
struct ABIFunctionType<Return (*)(Args...), std::enable_if_t<!IsABIMemoryClass<Return>::value>> {
  using type = Return (*)(Args...);
};


// Convenient struct for replacing a type for another.
// This is useful, mainly, when expand template parameter packs.

template <typename T, typename U>
struct replace {
  using type = U;
};

} // namespace tdnat
