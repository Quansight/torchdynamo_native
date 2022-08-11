#pragma once

#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Module.h>

#include <c10/util/Optional.h>

#include <cstdint>
#include <string>
#include <unordered_map>

namespace tdnat {

using Addr = uint64_t;

struct ATenOpVTable {
  using LLVMFunctionTypeFn = llvm::FunctionType *(*)(llvm::Module &);
  using AddAttributesFn = void (*)(llvm::Function *);
  using AllocateMemForRetFn = llvm::Value *(*)(llvm::IRBuilder<> &);

  ATenOpVTable(LLVMFunctionTypeFn llvm_function_type_fn,
               AddAttributesFn add_attributes_fn,
               AllocateMemForRetFn allocate_mem_for_ret_fn)
      : llvm_function_type_fn_(llvm_function_type_fn),
        add_attributes_fn_(add_attributes_fn),
        allocate_mem_for_ret_fn_(allocate_mem_for_ret_fn) {}

  // Pointer to the function that returns the LLVMFunctionType
  // instance that corresponds to this function's signature.
  LLVMFunctionTypeFn llvm_function_type_fn_;

  // Pointer to the function that adds the right attributes to a
  // llvm::Function instance of this operation.
  AddAttributesFn add_attributes_fn_;

  // Pointer to the function that allocates space on the stack
  // for this operation return value.
  AllocateMemForRetFn allocate_mem_for_ret_fn_;
};

class ATenOpRef {
public:
  ATenOpRef(const char *opname, Addr cpufn, bool returns_on_memory,
            ATenOpVTable vtable)
      : opname_(opname), cpufn_(cpufn), returns_on_memory_(returns_on_memory),
        vtable_(vtable) {}

  std::string name() { return std::string("__jit_") + opname_; }

  Addr cpu() { return cpufn_; }

  bool returns_on_memory() { return returns_on_memory_; }

  llvm::FunctionType *llvm_function_type(llvm::Module &module) {
    return vtable_.llvm_function_type_fn_(module);
  }

  void add_attributes(llvm::Function *fn) { vtable_.add_attributes_fn_(fn); }

  llvm::Value *allocate_mem_for_ret(llvm::IRBuilder<> &builder) {
    return vtable_.allocate_mem_for_ret_fn_(builder);
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

  // ATen type-specific operations.
  ATenOpVTable vtable_;
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
