#pragma once

#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/LLVMContext.h>

#include <ATen/core/boxing/impl/make_boxed_from_unboxed_functor.h>
#include <ATen/core/ivalue.h>

#include <cstdint>
#include <string>
#include <unordered_map>

namespace at {

class Tensor;

} // namespace at

namespace tdnat {

using Addr = uint64_t;

struct ATenOpRef {
  const char *opname_;
  Addr cpufn_;
  size_t input_number_;
  bool returns_void_;

  ATenOpRef() {}

  ATenOpRef(const char *opname, Addr cpufn, size_t input_number,
            bool returns_void)
      : opname_(opname), cpufn_(cpufn), input_number_(input_number),
        returns_void_(returns_void) {}

  llvm::FunctionType *llvm_function_type(llvm::LLVMContext &context) const {
    // For each input, we will have one IValue.
    std::vector<llvm::Type *> parameters{input_number_,
                                         llvm::Type::getInt8PtrTy(context)};
    // If the operation returns something, it will be an IValue.
    // Returning IValue means that there will be an extra IValue parameter.
    if (!returns_void_) {
      parameters.push_back(llvm::Type::getInt8PtrTy(context));
    }
    // The return type will always be 'void'.
    return llvm::FunctionType::get(llvm::Type::getVoidTy(context), parameters,
                                   false);
  }
};

using ATenOpRegistry = std::unordered_map<std::string, ATenOpRef>;
using ATenOpRegistryEntry = std::pair<std::string, ATenOpRef>;

// Gets an ATenOpRef, given a full ATen operation name.
// This API hides the actual registry implementation.
c10::optional<ATenOpRef> get_aten_op(const std::string &opname);

// Struct to be specialized for constructing an ATenOpRegistryEntry.
// The code generation will use it for building entries for the
// global registry.
template <typename T, T t> struct MakeRef {};

// Construct the pair, given a void function pointer.
template <typename... Args, void (*CPUFn)(Args...)>
struct MakeRef<void (*)(Args...), CPUFn> {
  using TypePtr = void (*)(Args...);
  using Self = MakeRef<TypePtr, CPUFn>;

  static ATenOpRegistryEntry get(const char *opname) {
    return {opname, {opname, (Addr)&Self::call, sizeof...(Args), true}};
  }

  static void call(c10::IValue *stack) {
    Self::call_impl(stack, std::make_index_sequence<sizeof...(Args)>());
  }

  template <size_t... ArgsIndex>
  static void call_impl(c10::IValue *stack, std::index_sequence<ArgsIndex...>) {
    (*CPUFn)(c10::impl::ivalue_to_arg<Args, false>::call(stack[ArgsIndex])...);
  }
};

/*
// Construct the pair, given a function pointer.
template <typename... OutputTypes, typename... Args,
          std::tuple<OutputTypes...> CPUFn>
struct MakeRef<std::tuple<OutputTypes...> (*)(Args...), CPUFn> {
  using TypePtr = std::tuple<OutputTypes...> (*)(Args...);
  using Self = MakeRef<TypePtr, CPUFn>;

  static ATenOpRegistryEntry get(const char *opname) {
    return {opname, {opname, (Addr)&Self::call, sizeof...(Args), true}};
  }

  static c10::IValue call(c10::IValue *stack, c10::IValue *output) {
    Self::call_impl(stack, output, std::make_index_sequence<sizeof...(Args)>());
  }

    static void push(c10::IValue* output, c10::IValue&& value, size_t i) {
        output[i] = value;
    }

  template <size_t... OutputIndex>
  static void ret(c10::IValue *output, std::tuple<OutputTypes...>&& values,
                               std::index_sequence<OutputIndex...>) {
      Self::push(output, c10::impl::return_to_ivalue<OutputTypes, false>::call(std::get<OutputIndex>(values)), OutputIndex)...
  }

  template <size_t... ArgsIndex>
  static void call_impl(c10::IValue *stack,
                               c10::IValue *output,
                               std::index_sequence<ArgsIndex...>) {
    return c10::impl::return_to_ivalue<Return, false>::call(
        (*CPUFn)(
            c10::impl::ivalue_to_arg<Args, false>::call(stack[ArgsIndex])...),
        true);
  }
};
*/

// Construct the pair, given a function pointer.
template <typename Return, typename... Args, Return (*CPUFn)(Args...)>
struct MakeRef<Return (*)(Args...), CPUFn> {
  using TypePtr = Return (*)(Args...);
  using Self = MakeRef<TypePtr, CPUFn>;

  static ATenOpRegistryEntry get(const char *opname) {
    return {opname, {opname, (Addr)&Self::call, sizeof...(Args), false}};
  }

  static c10::IValue call(c10::IValue *stack) {
    return Self::call_impl(stack, std::make_index_sequence<sizeof...(Args)>());
  }

  template <size_t... ArgsIndex>
  static c10::IValue call_impl(c10::IValue *stack,
                               std::index_sequence<ArgsIndex...>) {
    return c10::impl::return_to_ivalue<Return, false>::call(
        (*CPUFn)(
            c10::impl::ivalue_to_arg<Args, false>::call(stack[ArgsIndex])...),
        true);
  }
};

} // namespace tdnat
