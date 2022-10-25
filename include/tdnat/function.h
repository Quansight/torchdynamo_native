#pragma once

#include <tdnat/jit_api.h>
#include <tdnat/llvm_function_type.h>
#include <tdnat/llvm_type.h>
#include <tdnat/ops.h>

#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <llvm/ExecutionEngine/Orc/ThreadSafeModule.h>
#include <llvm/IR/IRBuilder.h>

#include <ATen/core/TensorBody.h>

#include <memory>
#include <type_traits>

namespace tdnat
{
class JITFunction;

struct Value {
  llvm::Value *val_;

  llvm::Value *operator*() const
  {
    return val_;
  }
};

struct FunctionData {
  std::string id_;
  size_t in_tensors_;
  size_t out_tensors_;
};

class Function
{
private:
  // Declares a given extern function to the module, so that it
  // can be called by the JIT function.
  template <typename Return, typename... Args>
  llvm::Function *_add_function_decl(const std::string &name, Return (*fn)(Args...));

  template <typename API>
  llvm::Function *_add_api_decl();

  // For ATenOpRef, we don't have type information.
  llvm::Function *_add_aten_op_decl(ATenOpRef ref);

  template <typename T>
  llvm::Type *_get_type();

public:
  Function(
      std::unique_ptr<llvm::Module> mod,
      std::unique_ptr<llvm::LLVMContext> ctx,
      FunctionData data
  );

  Value set_placeholder(int i, const std::string &name);

  std::vector<Value> set_output_from_refs(const std::vector<Value> &outputs);
  Value set_output_from_ref(const Value &output);

  Value add_call(
      const std::string &symbolname,
      const std::string &opname,
      const std::vector<Value> &args
  );

  void dump();

  JITFunction into_jit();

  Value build_bool(bool b);

  Value build_str(const std::string &s);

  Value build_load(Value val);

  template <typename T>
  Value build_load_for(Value val);

  template <typename T>
  Value build_int(T i);

  template <typename Repr, typename Enum>
  Value build_int_from_enum(Enum e);

  template <typename T>
  Value build_float(T f);

  template <typename T>
  Value build_scalar(Value literal);

  template <typename T>
  Value build_array(const std::vector<Value> &elements);

  template <typename T>
  Value build_nullopt();

  template <typename T, typename... Args>
  Value build_optional(typename replace<Args, Value>::type... args);

  template <typename T>
  Value build_nullopt_optionalarrayref();

  template <typename T>
  Value build_optionalarrayref(const std::vector<Value> &elements);

  template <typename T>
  Value build_list(const std::vector<Value> &elements);

  template <typename T>
  Value build_vector_index(Value vector, Value position);

  static std::unique_ptr<Function> from_data(FunctionData &data)
  {
    auto id = std::string("Module_for_") + data.id_;
    auto ctx = std::make_unique<llvm::LLVMContext>();
    auto mod = std::make_unique<llvm::Module>(id, *ctx);
    return std::make_unique<Function>(std::move(mod), std::move(ctx), data);
  }

private:
  FunctionData data_;

  llvm::orc::ThreadSafeModule module_;

  llvm::Function *fn_;
  llvm::IRBuilder<> builder_;

  // Map of symbol names to llvm::Value*.
  // Useful when one expression depends on some other expression
  // stored on a specific symbol.
  std::unordered_map<std::string, llvm::Value *> symbolmap_;

  // Map of function names to their actual address.
  // Used for registering those functions in the JIT.
  std::unordered_map<std::string, Addr> fnaddrmap_;
};

class JITFunction
{
public:
  JITFunction(llvm::orc::LLJIT *jit, FunctionData data);

  std::vector<at::Tensor *> run(at::ArrayRef<at::Tensor> in_tensors);
  void run_out(at::ArrayRef<at::Tensor> in_tensors, at::ArrayRef<at::Tensor *> out_tensors);

private:
  std::unique_ptr<llvm::orc::LLJIT> jit_;

  FunctionData data_;

  using RunFnType = void (*)(const at::Tensor *, at::Tensor *const *);
  RunFnType cache_ = nullptr;
};

} // namespace tdnat

// tdnat::Function member function implementations.
#include <tdnat/function_inl.h>
