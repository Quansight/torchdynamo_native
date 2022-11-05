#pragma once

#include <tdnat/llvm_type.h>

#include <llvm/IR/IRBuilder.h>

#include <ATen/core/TensorBody.h>
#include <c10/core/Scalar.h>
#include <c10/util/ArrayRef.h>
#include <c10/util/Optional.h>
#include <c10/util/OptionalArrayRef.h>

namespace tdnat
{
namespace jit
{

// Implementation of functions used by the JIT for creation,
// deletion, and manipulation of PyTorch types.
//
// They are used for creating data structures out of
// scalars by implicitly calling their constructors.
//
// There are no requirements for something to be part of the
// JIT API. But, as a rule of thumb, we make each function
// a structure with, at least, 2 static methods inside:
//
//   1. std::string name()
//     Returns a unique identifier.
//
//   2. T run(...)
//     The actual function.

template <typename T, typename... Args>
struct Scalar {
  static std::string name()
  {
    return "Scalar(" + LLVMType<T>::name() + ")<" + concat_type_names<Args...>() + ">";
  }

  static at::Scalar *run(Args... args)
  {
    return new at::Scalar(T(args...));
  }
};

template <typename T, typename... Args>
struct Optional {
  static std::string name()
  {
    return "Optional(" + LLVMType<T>::name() + ")<" + concat_type_names<Args...>() + ">";
  }

  static c10::optional<T> *run(Args... args)
  {
    return new c10::optional<T>(T(args...));
  }
};

template <typename T>
struct NullOpt {
  static std::string name()
  {
    return "NullOpt<" + LLVMType<T>::name() + ">";
  }

  static c10::optional<T> *run()
  {
    return new c10::optional<T>(c10::nullopt);
  }
};

template <typename T>
struct OptionalArrayRef {
  static std::string name()
  {
    return "OptionalArrayRef<" + LLVMType<T>::name() + ">";
  }

  static c10::OptionalArrayRef<T> *run(const T *ptr, int64_t size)
  {
    return new c10::OptionalArrayRef<T>({ptr, static_cast<size_t>(size)});
  }
};

template <typename T>
struct NullOptOptionalArrayRef {
  static std::string name()
  {
    return "NullOptOptionalArrayRef<" + LLVMType<T>::name() + ">";
  }

  static c10::OptionalArrayRef<T> *run()
  {
    return new c10::OptionalArrayRef<T>(c10::nullopt);
  }
};

template <typename T>
struct List {
  static std::string name()
  {
    return "List<" + LLVMType<T>::name() + ">";
  }

  static c10::List<T> *run(T *list, int64_t size)
  {
    // Constructor is marked as explicit.
    return new c10::List<T>({list, static_cast<size_t>(size)});
  }
};

template <typename T>
struct VectorIndex {
  static std::string name()
  {
    return "vector_index";
  }

  static const at::Tensor &run(const std::vector<at::Tensor> &vector, int64_t i)
  {
    return vector[i];
  }
};

} // namespace jit
} // namespace tdnat
