#pragma once

#include <c10/core/Scalar.h>
#include <c10/util/Optional.h>

#include <type_traits>
#include <vector>

namespace tdnat {

template <typename T> struct IsOptionalType : public std::false_type {};
template <typename T>
struct IsOptionalType<c10::optional<T>> : public std::true_type {};

// C++ types that are returned on memory as 1st argument.
// Some of the necessary conditions are:
//   - Different than void
//   - Trivially copy constructible
//   - Trivially destructible
//
// c10::Scalar, for example, is also returned on memory.
// However, that is because of its size.

template <typename T, typename _Tp = void>
struct ReturnsOnMemory : public std::false_type {};

template <typename T>
struct ReturnsOnMemory<c10::optional<T>,
                       std::enable_if_t<ReturnsOnMemory<T>::value>>
    : public std::true_type {};

template <typename T>
struct ReturnsOnMemory<
    T, std::enable_if_t<!std::is_same<T, void>::value &&
                        !IsOptionalType<T>::value &&
                        (!std::is_trivially_copy_constructible<T>::value ||
                         !std::is_trivially_destructible<T>::value)>>
    : public std::true_type {};

template <> struct ReturnsOnMemory<at::Scalar> : public std::true_type {};

} // namespace tdnat
