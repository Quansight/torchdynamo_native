#pragma once

#include <ATen/core/TensorBody.h>
#include <c10/util/Optional.h>

#include <gtest/gtest.h>

namespace tdnat {

using PyTorchTypes =
    testing::Types<bool, int64_t, at::ArrayRef<int64_t>, at::Tensor,
                   c10::Scalar, std::vector<at::Tensor>>;


} // namespace tdnat
