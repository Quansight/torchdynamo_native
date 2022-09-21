#pragma once

#include <tdnat/llvm_type.h>

#include <llvm/IR/IRBuilder.h>

#include <ATen/core/TensorBody.h>
#include <c10/core/Scalar.h>
#include <c10/util/ArrayRef.h>
#include <c10/util/Optional.h>

namespace tdnat
{
namespace factory
{

// Factories are just plain functions coupled with an
// unique identifier, i.e. its name.
//
// They are used for creating data structures out of
// scalars by implicitly calling their constructors.
//
// There are no requirements for something to be a
// factory. But, as a rule of thumb, we make each factory
// a structure with, at least, 2 static methods inside:
//
//   1. std::string name()
//     Returns the unique identifier for this factory.
//
//   2. T create(...)
//     The actual factory function.

template <typename T>
struct Scalar {
  static std::string name()
  {
    return "Scalar<" + LLVMType<T>::name() + ">";
  }

  static at::Scalar create(T val)
  {
    return {val};
  }
};

template <typename T>
struct ArrayRef {
  static std::string name()
  {
    return "ArrayRef<" + LLVMType<T>::name() + ">";
  }

  static at::ArrayRef<T> create(T *list, size_t size)
  {
    return {list, size};
  }
};

template <typename T>
struct Optional {
  static std::string name()
  {
    return "Optional<" + LLVMType<T>::name() + ">";
  }

  static c10::optional<T> create(const T &val)
  {
    return {val};
  }
};

template <typename T>
struct NullOpt {
  static std::string name()
  {
    return "NullOpt<" + LLVMType<T>::name() + ">";
  }

  static c10::optional<T> create()
  {
    return {c10::nullopt};
  }
};

struct OptionalTensorList {
  static std::string name()
  {
    return "List<Optional<Tensor>>";
  }

  static c10::List<c10::optional<at::Tensor>> create(c10::optional<at::Tensor> *list, size_t size)
  {
    return c10::List<c10::optional<at::Tensor>>{{list, size}};
  }
};

} // namespace factory
} // namespace tdnat
