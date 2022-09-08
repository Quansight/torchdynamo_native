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
struct JITFunction;

struct Value {
  llvm::Value *val_;
};

struct FunctionData {
  std::string id_;
  size_t in_tensors_;
  size_t out_tensors_;
};

struct Function {
private:
  // Declares a given extern function to the module, so that it
  // can be called by the JIT function.
  template <typename Return, typename... Args>
  llvm::Function *__add_function_decl(const std::string &name,
                                      Return (*fn)(Args...)) {
    if (fnaddrmap_.find(name) == fnaddrmap_.end()) {
      fnaddrmap_[name] = reinterpret_cast<Addr>(fn);
      auto llvm_fn = llvm::Function::Create(
          ABILLVMFunctionType<Return (*)(Args...)>::get(*mod_),
          llvm::GlobalValue::ExternalLinkage, name, *mod_);
      add_attributes<Return, Args...>(llvm_fn);
    }
    return mod_->getFunction(name);
  }

  // For ATenOpRef, we don't have type information.
  llvm::Function *__add_aten_op_decl(ATenOpRef ref);

  template <typename T> Value __build_scalar_impl(llvm::Value *val) {
    auto scalar_fn =
        __add_function_decl(factory::__scalar_id, &factory::scalar<T>);

    auto alloca = builder_.CreateAlloca(LLVMType<at::Scalar>::get(*mod_));
    alloca->setAlignment(llvm::Align(alignof(at::Scalar)));

    builder_.CreateCall(scalar_fn, {alloca, val});
    return {alloca};
  }

  void __check_finalized(bool expected = false);

public:
  Function(const FunctionData &data);

  Value set_placeholder(int i, const std::string &name);

  Value add_call(const std::string &symbolname, const std::string &opname,
                 const std::vector<Value> &args);

  std::vector<Value> set_outputs(const std::vector<Value> &outputs);
  Value set_output(const Value &output);

  Value build_int(int64_t n);
  Value build_intarray(const std::vector<int64_t> &v);
  Value build_scalar(int64_t n);

  void dump();
  void finalize();

  JITFunction into_jit();

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

struct JITFunction {
  JITFunction(llvm::orc::LLJIT *jit, const FunctionData &data);

  void run(at::ArrayRef<at::Tensor> in_tensors,
           at::ArrayRef<at::Tensor> out_tensors);

  std::vector<at::Tensor> run(at::ArrayRef<at::Tensor> in_tensors);

private:
  std::unique_ptr<llvm::orc::LLJIT> jit_;

  FunctionData data_;

  using RunFnType = void (*)(const at::Tensor *, const at::Tensor *);
  RunFnType cache_ = nullptr;
};

} // namespace tdnat
