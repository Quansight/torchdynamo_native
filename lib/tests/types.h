#pragma once

#include <ATen/core/TensorBody.h>
#include <c10/util/Optional.h>

#include <gtest/gtest.h>

#define SUPPORTED_VALUE_TYPES(_)                                               \
  _(bool)                                                                      \
  _(int8_t)                                                                    \
  _(int16_t)                                                                   \
  _(int32_t)                                                                   \
  _(int64_t)                                                                   \
  _(float)                                                                     \
  _(double)                                                                    \
  _(c10::ScalarType)                                                           \
  _(c10::Layout)                                                               \
  _(c10::MemoryFormat)                                                         \
  _(c10::Device)                                                               \
  _(c10::Stream)                                                               \
  _(c10::basic_string_view<char>)                                              \
  _(at::ArrayRef<long>)                                                        \
  _(c10::optional<at::ArrayRef<at::Tensor>>)                                   \
  _(c10::OptionalArrayRef<at::Tensor>)

#define SUPPORTED_PTR_TYPES(_)                                                 \
  _(at::Tensor)                                                                \
  _(at::Generator)                                                             \
  _(c10::Storage)                                                              \
  _(std::vector<at::Tensor>)                                                   \
  _(at::Scalar)                                                                \
  _(c10::optional<at::Tensor>)

#define SUPPORTED_TYPES(_)                                                     \
  SUPPORTED_VALUE_TYPES(_)                                                     \
  SUPPORTED_PTR_TYPES(_)

namespace tdnat {

template <typename First, typename... Args> struct testing_skip_first {
  using type = testing::Types<Args...>;
};

#if defined(TEST_ALL_TYPES)
#define COMMA(TYPE) , TYPE
using PyTorchTypes =
    typename testing_skip_first<void SUPPORTED_TYPES(COMMA)>::type;
#undef COMMA
#else
using PyTorchTypes =
    testing::Types<int64_t, double, at::IntArrayRef, at::Tensor,
                   c10::optional<int64_t>, c10::optional<double>,
                   c10::optional<at::IntArrayRef>, c10::optional<at::Tensor>>;
#endif

} // namespace tdnat
