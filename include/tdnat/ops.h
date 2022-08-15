#pragma once

#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/LLVMContext.h>

#include <c10/util/Optional.h>

#include <cstdint>
#include <string>
#include <unordered_map>

namespace tdnat {

using Addr = uint64_t;

struct ATenOpRef {
  using LLVMFunctionTypeFn = llvm::FunctionType *(*)(llvm::LLVMContext &);

  const char *opname_;
  Addr cpufn_;
  LLVMFunctionTypeFn llvm_function_type_fn_;

  ATenOpRef() {}

  ATenOpRef(const char *opname, Addr cpufn,
            LLVMFunctionTypeFn llvm_function_type_fn)
      : opname_(opname), cpufn_(cpufn),
        llvm_function_type_fn_(llvm_function_type_fn) {}

  llvm::FunctionType *llvm_function_type(llvm::LLVMContext &context) {
    return (*llvm_function_type_fn_)(context);
  }
};

using ATenOpRegistry = std::unordered_map<std::string, ATenOpRef>;
using ATenOpRegistryEntry = std::pair<std::string, ATenOpRef>;

// Gets an ATenOpRef, given a full ATen operation name.
// This API hides the actual registry implementation.
c10::optional<ATenOpRef> get_aten_op(const std::string &opname);

// Initialize a registry with PyTorch operations.
void initialize_registry(ATenOpRegistry& registry);

// Initialize the necessary LLVM components.
void initialize_llvm();

// Initialize tdnat library.
void initialize(ATenOpRegistry& registry);
void initialize();

} // namespace tdnat
