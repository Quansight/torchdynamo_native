#pragma once

#include <tdnat/llvm_function_type.h>
#include <tdnat/ops.h>

#include <llvm/IR/IRBuilder.h>

namespace tdnat {

// Struct to be specialized for constructing an ATenOpRegistryEntry.
// The code generation will use it for building entries for the
// global registry.
template <typename T, T t> struct MakeRef {};

// Construct the pair, given a void function pointer.
template <typename Return, typename... Args, Return (*CPUFn)(Args...)>
struct MakeRef<Return (*)(Args...), CPUFn> {
  using type_ptr = Return (*)(Args...);

  static ATenOpRegistryEntry get(const char *opname) {
    return {opname,
            {opname, (Addr)CPUFn, IsABIMemoryClass<Return>::value, vtable()}};
  }

  static ATenOpVTable vtable() {
    return {&ABILLVMFunctionType<type_ptr>::get,
            &add_attributes<Return, Args...>, &alloc};
  }

  static llvm::Value *alloc(llvm::IRBuilder<> &builder) {
    if (IsABIMemoryClass<Return>::value) {
      auto &module = *builder.GetInsertBlock()->getModule();
      return builder.CreateAlloca(LLVMType<Return>::get(module));
    }
    return nullptr;
  }
};

} // namespace tdnat
