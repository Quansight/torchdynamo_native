#pragma once

#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/LLVMContext.h>

#include <c10/util/Optional.h>

#include <cstdint>
#include <string>
#include <unordered_map>

namespace tdnat {

using Addr = uint64_t;

struct ATenOpRefVTable {
  using LLVMFunctionTypeFn = llvm::FunctionType *(*)(llvm::LLVMContext &);
  LLVMFunctionTypeFn llvm_function_type_fn_;
};

class ATenOpRef {
public:
  ATenOpRef() {}

  ATenOpRef(const char *opname, Addr cpufn, ATenOpRefVTable vtable)
      : opname_(opname), cpufn_(cpufn), vtable_(vtable) {}

  std::string name() { return std::string("__jit_") + opname_; }

  Addr cpu() { return cpufn_; }

  llvm::FunctionType *llvm_function_type(llvm::LLVMContext &context) {
    return (*vtable_.llvm_function_type_fn_)(context);
  }

private:
  const char *opname_;
  Addr cpufn_;
  ATenOpRefVTable vtable_;
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
