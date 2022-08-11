#pragma once

#include <tdnat/ops.h>
#include <tdnat/llvm.h>

namespace tdnat {

using Addr = uint64_t;

// Struct to be specialized for constructing an ATenOpRegistryEntry.
// The code generation will use it for building entries for the
// global registry.
template <typename T, T t> struct MakeRef {};

// Construct the pair, given a void function pointer.
template <typename Return, typename... Args, Return (*CPUFn)(Args...)>
struct MakeRef<Return (*)(Args...), CPUFn> {
  using type_ptr = Return (*)(Args...);
  using abi_type_ptr = typename ABITrait<type_ptr>::type;

  static ATenOpRegistryEntry get(const char *opname) {
    return {opname,
            {opname, (Addr)CPUFn, &ToLLVMFunctionType<abi_type_ptr>::call}};
  }
};

} // namespace tdnat
