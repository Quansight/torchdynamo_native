#pragma once

#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/LLVMContext.h>

#include <c10/util/Optional.h>

#include <cstdint>
#include <string>
#include <unordered_map>

namespace tdnat {

using Addr = uint64_t;
using LLVMFunctionTypeFactory = llvm::FunctionType *(*)(llvm::LLVMContext &);

class ATenOpRef {
public:
  ATenOpRef() {}

  ATenOpRef(const char *opname, Addr cpufn, bool returns_on_memory,
            LLVMFunctionTypeFactory llvm_function_type_fn)
      : opname_(opname), cpufn_(cpufn), returns_on_memory_(returns_on_memory),
        llvm_function_type_fn_(llvm_function_type_fn) {}

  std::string name() { return std::string("__jit_") + opname_; }

  Addr cpu() { return cpufn_; }

  bool returns_on_memory() { return returns_on_memory_; }

  llvm::FunctionType *llvm_function_type(llvm::LLVMContext &context) {
    return llvm_function_type_fn_(context);
  }

private:
  // Key used in the registry.
  // Corresponds to the overloaded name of the operation.
  const char *opname_;

  // Actual address of the function.
  Addr cpufn_;

  // Flag indicating whether this operation's return value is passed
  // on the stack, whose address is stored in the first parameter.
  bool returns_on_memory_;

  // Pointer to the function that returns the LLVMFunctionType
  // instance that corresponds to this function's signature.
  LLVMFunctionTypeFactory llvm_function_type_fn_;
};

using ATenOpRegistry = std::unordered_map<std::string, ATenOpRef>;
using ATenOpRegistryEntry = std::pair<std::string, ATenOpRef>;

// Gets an ATenOpRef, given a full ATen operation name.
// This API hides the actual registry implementation.
c10::optional<ATenOpRef> get_aten_op(const std::string &opname);

// Initialize a registry with PyTorch operations.
void initialize_registry(ATenOpRegistry &registry);

// Initialize the necessary LLVM components.
void initialize_llvm();

// Initialize tdnat library.
void initialize(ATenOpRegistry &registry);
void initialize();

} // namespace tdnat
