#pragma once

#include <tdnat/llvm_type.h>
#include <tdnat/ops.h>

#include <llvm/IR/IRBuilder.h>

#include <ATen/core/TensorBody.h>

namespace tdnat {

struct Value {
  llvm::Value *val_;
};

struct LLVMFunctionBuilder {
  LLVMFunctionBuilder(const std::string &id, size_t in_tensors,
                      size_t out_tensors);

  void add_tensor_placeholder(int i, const std::string &name);

  void add_call(const std::string &symbolname, const std::string &opname,
                const std::vector<Value> &args);

  void add_outputs(const std::vector<std::string>& outputs);
  void add_output(const std::string& output);

  llvm::Function *get_or_declare_op(ATenOpRef ref);

  Value build_int(int64_t n);

  Value build_intarray(const std::vector<int64_t> &v);

  void dump();

private:
  size_t in_tensors_;
  size_t out_tensors_;

  std::unique_ptr<llvm::LLVMContext> ctx_;
  std::unique_ptr<llvm::Module> mod_;

  llvm::Function *fn_;
  llvm::IRBuilder<> builder_;

  // Map of symbol names to llvm::Value*.
  // Useful when one expression depends on some other expression
  // stored on a specific symbol.
  std::unordered_map<std::string, llvm::Value *> symbolmap_;

  // Map of ATen operation names to their actual address.
  // Used for registering those operations in the JIT.
  std::unordered_map<std::string, Addr> opaddrmap_;
};

} // namespace tdnat
