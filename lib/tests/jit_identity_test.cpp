#include <algorithm>
#include <bits/utility.h>
#include <gtest/gtest.h>

#include <iterator>
#include <tdnat/llvm_function_type.h>
#include <tdnat/ops.h>
#include <tdnat/utils.h>

#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/Support/TargetSelect.h>

#include <ATen/ops/add.h>
#include <ATen/ops/diag.h>
#include <ATen/ops/ones.h>
#include <ATen/ops/randint.h>
#include <ATen/ops/tensor.h>

#include <iostream>
#include <memory>
#include <type_traits>
#include <utility>

#include "tdnat/llvm_type.h"
#include "tdnat/ops_util.h"
#include "test_utils.h"
#include "llvm/ExecutionEngine/JITSymbol.h"
#include "llvm/IR/Argument.h"
#include "llvm/IR/Attributes.h"
#include "llvm/Support/Error.h"

using namespace tdnat;

static const char *ModuleName = "jit_test";
static const char *EntryFn = "entry_function";
static const char *IdentityFn = "identity_function";
static const char *EntryBlock = "entry";

template <typename T> T identity(T t) { return t; }

class Environment : public testing::Environment {
public:
  ~Environment() override {}
  void SetUp() override { initialize_llvm(); }
};

const auto *env = testing::AddGlobalTestEnvironment(new Environment());

template <typename T>
void add_byval_attr(llvm::LLVMContext &ctx, llvm::Value *arg) {
  if (IsABIMemoryClass<T>::value) {
    std::cout << "AddAttr(byval) to arg" << std::endl;
    assert(llvm::isa<llvm::Argument>(arg));
    auto cast_arg = llvm::dyn_cast<llvm::Argument>(arg);
    cast_arg->addAttr(
        llvm::Attribute::getWithByValType(ctx, LLVMType<T>::get(ctx)));
  }
}

template <typename First = void, typename... Rest>
void add_byval_attrs(size_t index, llvm::LLVMContext &ctx,
                     const std::vector<llvm::Value *> &args) {
  add_byval_attr<First>(ctx, args[index]);
  if (sizeof...(Rest) >= 1) {
    add_byval_attrs<Rest...>(index + 1, ctx, args);
  }
}

template <typename Return, typename... Args>
std::unique_ptr<llvm::orc::LLJIT> create_jit(Return (*fn)(Args...)) {
  using AttrKind = llvm::Attribute::AttrKind;
  using type_ptr = Return (*)(Args...);
  using abi_type_ptr = typename ABIFunctionType<type_ptr>::type;

  auto ctx = std::make_unique<llvm::LLVMContext>();
  auto mod = std::make_unique<llvm::Module>(ModuleName, *ctx);
  auto jit = llvm::cantFail(llvm::orc::LLJITBuilder().create());

  auto entry_fn = llvm::Function::Create(
      LLVMFunctionType<abi_type_ptr>::get(*ctx),
      llvm::Function::ExternalLinkage, EntryFn, mod.get());

  auto identity_fn = llvm::Function::Create(
      LLVMFunctionType<abi_type_ptr>::get(*ctx),
      llvm::Function::ExternalLinkage, IdentityFn, mod.get());

  auto bb =
      llvm::BasicBlock::Create(entry_fn->getContext(), EntryBlock, entry_fn);
  llvm::IRBuilder<> builder(bb);

  std::vector<llvm::Value *> arguments;
  std::transform(entry_fn->arg_begin(), entry_fn->arg_end(),
                 std::back_inserter(arguments),
                 [](llvm::Argument &arg) { return (llvm::Value *)&arg; });
  add_byval_attrs<Args...>(0, *ctx, arguments);

  auto call = builder.CreateCall(identity_fn, arguments);

  if (IsABIMemoryClass<Return>::value) {
    builder.CreateRet(call);
    auto retptr = entry_fn->getArg(0);
    retptr->addAttr(llvm::Attribute::get(*ctx, AttrKind::StructRet,
                                         LLVMType<Return>::get(*ctx)));
  }

  llvm::cantFail(jit->addIRModule({std::move(mod), std::move(ctx)}));

  llvm::cantFail(jit->define(llvm::orc::absoluteSymbols(
      {{jit->mangleAndIntern(IdentityFn),
        llvm::JITEvaluatedSymbol::fromPointer(fn)}})));

  entry_fn->print(llvm::outs());
  identity_fn->print(llvm::outs());

  return jit;
}

TEST(JITTest, AddTest) {
  using AddFnType = at::Tensor (*)(const at::Tensor &, const at::Tensor &,
                                   const at::Scalar &);

  auto jit = create_jit((AddFnType)at::add);
  auto symbol = llvm::cantFail(jit->lookup(IdentityFn));
  auto func = (void *)symbol.getAddress();

  auto lhs = at::randint(10, {2, 2});
  auto rhs = at::randint(10, {2, 2});
  auto alpha = 5;

  auto result = ((AddFnType)func)(lhs, rhs, alpha);
  auto expect = at::add(lhs, rhs, alpha);

  ASSERT_TRUE(expect.equal(result));
}
