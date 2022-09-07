#pragma once

#include <tdnat/function.h>

#include <llvm/IR/IRBuilder.h>

#include <c10/core/Scalar.h>

namespace tdnat {
namespace factory {

#define DECLARE_FACTORY(NAME) \
    static const char* __##NAME##_id = "__jit_factory_" #NAME;


DECLARE_FACTORY(scalar);
template <typename T> at::Scalar scalar(T val) { return {val}; }

#undef DECLARE_FACTORY

} // namespace factory
} // namespace tdnat
