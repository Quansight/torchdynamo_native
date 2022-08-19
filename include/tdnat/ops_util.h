#pragma once

#include <tdnat/llvm_function_type.h>
#include <tdnat/ops.h>

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
    return {opname, {opname, (Addr)CPUFn, MakeRef<type_ptr, CPUFn>::vtable()}};
  }

  static ATenOpRefVTable vtable() {
    return {&ABILLVMFunctionType<type_ptr>::get};
  }
};

} // namespace tdnat
