#pragma once

#include <tdnat/llvm_type.h>
#include <tdnat/utils.h>

#include <llvm/IR/Attributes.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/LLVMContext.h>

#include <type_traits>

namespace tdnat {

template <typename T> struct LLVMFunctionType {};

template <typename Return, typename... Args>
struct LLVMFunctionType<Return (*)(Args...)> {
  static llvm::FunctionType *get(llvm::Module &module) {
    return llvm::FunctionType::get(LLVMType<Return>::get(module),
                                   {LLVMArgType<Args>::get(module)...}, false);
  }
};

template <typename T> struct ABILLVMFunctionType {
  static llvm::FunctionType *get(llvm::Module &module) {
    return LLVMFunctionType<typename ABIFunctionType<T>::type>::get(module);
  }
};

// LLVM needs return type and parameter type attributes
// so as to correctly generate code for a backend.
//
// In summary, types classified as MEMORY class must be
// passed (returned) on the stack. So as to meet that
// requirement, we must add these attributes.

template <typename Return, typename... Args>
void add_attributes(llvm::Function *fn) {
  using AttrKind = llvm::Attribute::AttrKind;

  size_t offset = 0;

  // Add the 'sret' attribute for return types that should be
  // returned on the stack.
  auto &mod = *fn->getParent();
  auto &ctx = fn->getContext();
  if (IsABIMemoryClass<Return>::value) {
    fn->getArg(0)->addAttr(llvm::Attribute::get(ctx, AttrKind::StructRet,
                                                LLVMType<Return>::get(mod)));
    offset = 1;
  }

  // Add the 'byval' attribute for parameters that should be
  // passed on the stack.
  auto is_byval_type = std::vector<bool>{IsABIMemoryClass<Args>::value...};
  auto types = std::vector<llvm::Type *>{LLVMType<Args>::get(mod)...};
  for (size_t i = 0; i < is_byval_type.size(); i++) {
    if (is_byval_type[i]) {
      fn->getArg(i + offset)
          ->addAttr(llvm::Attribute::getWithByValType(ctx, types[i]));
    }
  }
}

} // namespace tdnat
