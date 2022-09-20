#pragma once

#include <tdnat/factories.h>
#include <tdnat/llvm_function_type.h>
#include <tdnat/llvm_type.h>
#include <tdnat/ops.h>

#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <llvm/IR/IRBuilder.h>

#include <ATen/core/TensorBody.h>

#include <memory>
#include <type_traits>

namespace tdnat {
class JITFunction;

struct Value {
  llvm::Value *val_;
};

struct FunctionData {
  std::string id_;
  size_t in_tensors_;
  size_t out_tensors_;
};

class Function {
private:
  // Declares a given extern function to the module, so that it
  // can be called by the JIT function.
  template <typename Return, typename... Args>
  llvm::Function *__add_function_decl(const std::string &name,
                                      Return (*fn)(Args...));

  template <typename Factory> llvm::Function *__add_factory_decl();

  // For ATenOpRef, we don't have type information.
  llvm::Function *__add_aten_op_decl(ATenOpRef ref);

  template <typename T> Value __build_scalar(Value val);

  template <typename T, typename Factory>
  Value __build_optional(c10::optional<Value> val = c10::nullopt);

  template <typename T>
  Value __build_arrayref(const std::vector<Value> &vals, bool from_literal);

  // Checks whether this function was (not) finalized if
  // 'expected' is true (false).
  void __check_finalized(bool expected = false);

  template <typename T> llvm::Type *__get_type();

public:
  Function(const FunctionData &data);
  Function(Function &&fn);

  Value set_placeholder(int i, const std::string &name);

  std::vector<Value> set_outputs(const std::vector<Value> &outputs);
  Value set_output(const Value &output);

  Value add_call(const std::string &symbolname, const std::string &opname,
                 const std::vector<Value> &args);

  void dump();
  void finalize();

  JITFunction into_jit();

  Value build_bool(bool b);

  Value build_optional_tensorlist(const std::vector<Value> &v);

  Value build_scalar_type(at::ScalarType type);

  Value build_scalar(int64_t n);

  Value build_vector_at_tensor(Value val, Value position);

  template <typename T> Value build_integer(T n);

  template <typename T> Value build_arrayref(const std::vector<Value> &v);
  template <typename T> Value build_arrayref_lit(const std::vector<Value> &v);

  template <typename T> Value build_nullopt();
  template <typename T> Value build_optional(Value val);
  template <typename T> Value build_optional_lit(Value val);

private:
  FunctionData data_;

  std::unique_ptr<llvm::LLVMContext> ctx_;
  std::unique_ptr<llvm::Module> mod_;

  llvm::Function *fn_;
  llvm::IRBuilder<> builder_;

  // Map of symbol names to llvm::Value*.
  // Useful when one expression depends on some other expression
  // stored on a specific symbol.
  std::unordered_map<std::string, llvm::Value *> symbolmap_;

  // Map of function names to their actual address.
  // Used for registering those functions in the JIT.
  std::unordered_map<std::string, Addr> fnaddrmap_;

  // Flags whether the function is still being built.
  bool finalized_;
};

class JITFunction {
  JITFunction(llvm::orc::LLJIT *jit, const FunctionData &data);

  void run_out(at::ArrayRef<at::Tensor> in_tensors,
               at::ArrayRef<at::Tensor> out_tensors);

  std::vector<at::Tensor> run(at::ArrayRef<at::Tensor> in_tensors);

private:
  std::unique_ptr<llvm::orc::LLJIT> jit_;

  FunctionData data_;

  using RunFnType = void (*)(const at::Tensor *, const at::Tensor *);
  RunFnType cache_ = nullptr;
};

} // namespace tdnat

// tdnat::Function member function implementations.
#include <tdnat/function_inl.h>
