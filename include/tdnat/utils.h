#pragma once

#include <ATen/core/Generator.h>
#include <ATen/core/TensorBody.h>
#include <c10/core/Scalar.h>
#include <c10/core/Storage.h>
#include <c10/util/Optional.h>

#include <type_traits>
#include <vector>

namespace tdnat {

template <typename T> struct IsOptionalType : public std::false_type {};
template <typename T>
struct IsOptionalType<c10::optional<T>> : public std::true_type {};

// C++ types that are classified as MEMORY class in the SystemV ABI.
// Some of the necessary conditions are:
//   - Different than void
//   - Trivially copy constructible
//   - Trivially destructible
//
// c10::Scalar, for example, is also classified as MEMORY.
// However, that is because of its size.

template <typename T, typename _Tp = void>
struct IsABIMemoryClass : public std::false_type {};

template <typename T>
struct IsABIMemoryClass<c10::optional<T>,
                       std::enable_if_t<IsABIMemoryClass<T>::value>>
    : public std::true_type {};

template <typename T>
struct IsABIMemoryClass<
    T, std::enable_if_t<!std::is_same<T, void>::value &&
                        !IsOptionalType<T>::value &&
                        (!std::is_trivially_copy_constructible<T>::value ||
                         !std::is_trivially_destructible<T>::value)>>
    : public std::true_type {};

template <> struct IsABIMemoryClass<at::Scalar> : public std::true_type {};

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

template <typename T, typename _Tp = void> struct ABIFunctionType {};

template <typename Return, typename... Args>
struct ABIFunctionType<Return (*)(Args...),
                    std::enable_if_t<IsABIMemoryClass<Return>::value>> {
  using type = Return *(*)(Return *, Args...);
};

template <typename Return, typename... Args>
struct ABIFunctionType<Return (*)(Args...),
                    std::enable_if_t<!IsABIMemoryClass<Return>::value>> {
  using type = Return (*)(Args...);
};

} // namespace tdnat
