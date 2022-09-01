#include "jit_utils.h"

#include <tdnat/llvm_function_type.h>
#include <tdnat/ops.h>
#include <tdnat/utils.h>

#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/Support/TargetSelect.h>

#include <ATen/ops/add.h>
#include <ATen/ops/argmin.h>
#include <ATen/ops/cat.h>
#include <ATen/ops/chunk.h>
#include <ATen/ops/index.h>
#include <ATen/ops/randint.h>
#include <ATen/ops/sum.h>

#include <algorithm>
#include <bits/utility.h>
#include <iostream>
#include <iterator>
#include <memory>
#include <type_traits>
#include <utility>

using namespace tdnat;

static const char *ModuleName = "jit_test";
static const char *EntryFn = "entry_function";
static const char *WrappedFn = "wrapped_function";
static const char *EntryBlock = "entry";

template <typename T>
void add_byval_attr(llvm::LLVMContext &ctx, llvm::Value *arg) {
  if (IsABIMemoryClass<T>::value) {
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

  auto wrapped_fn = llvm::Function::Create(
      LLVMFunctionType<abi_type_ptr>::get(*ctx),
      llvm::Function::ExternalLinkage, WrappedFn, mod.get());

  auto bb =
      llvm::BasicBlock::Create(entry_fn->getContext(), EntryBlock, entry_fn);
  llvm::IRBuilder<> builder(bb);

  std::vector<llvm::Value *> arguments;
  std::transform(entry_fn->arg_begin(), entry_fn->arg_end(),
                 std::back_inserter(arguments),
                 [](llvm::Argument &arg) { return (llvm::Value *)&arg; });
  add_byval_attrs<Args...>(0, *ctx, arguments);

  auto call = builder.CreateCall(wrapped_fn, arguments);

  if (IsABIMemoryClass<Return>::value) {
    builder.CreateRet(call);
    auto retptr = entry_fn->getArg(0);
    retptr->addAttr(llvm::Attribute::get(*ctx, AttrKind::StructRet,
                                         LLVMType<Return>::get(*ctx)));
  }

  llvm::cantFail(jit->addIRModule({std::move(mod), std::move(ctx)}));

  llvm::cantFail(jit->define(llvm::orc::absoluteSymbols(
      {{jit->mangleAndIntern(WrappedFn),
        llvm::JITEvaluatedSymbol::fromPointer(fn)}})));

  return jit;
}

TEST(JITTest, AddTest) {
  using FnType = at::Tensor (*)(const at::Tensor &, const at::Tensor &,
                                const at::Scalar &);

  auto jit = create_jit((FnType)at::add);
  auto symbol = llvm::cantFail(jit->lookup(WrappedFn));
  auto func = (void *)symbol.getAddress();

  auto lhs = at::randint(10, {2, 2});
  auto rhs = at::randint(10, {2, 2});
  auto alpha = 5;

  auto result = ((FnType)func)(lhs, rhs, alpha);
  auto expect = at::add(lhs, rhs, alpha);

  ASSERT_TRUE(expect.equal(result));
}

TEST(JITTest, CatTest) {
  using FnType = at::Tensor (*)(at::ArrayRef<at::Tensor>, int64_t);

  auto jit = create_jit((FnType)at::cat);
  auto symbol = llvm::cantFail(jit->lookup(WrappedFn));
  auto func = (void *)symbol.getAddress();

  auto t1 = at::randint(10, {1, 2, 2});
  auto t2 = at::randint(10, {1, 2, 2});

  auto result = ((FnType)func)({t1, t2}, 0);
  auto expect = at::cat({t1, t2}, 0);

  ASSERT_TRUE(expect.equal(result));
}

TEST(JITTest, IndexTest) {
  using FnType = at::Tensor (*)(const at::Tensor &,
                                const c10::List<c10::optional<at::Tensor>> &);

  auto jit = create_jit((FnType)at::index);
  auto symbol = llvm::cantFail(jit->lookup(WrappedFn));
  auto func = (void *)symbol.getAddress();

  auto tensor = at::randint(10, {1, 10, 10});
  auto i0 = at::randint(10, {4, 4}, at::TensorOptions{}.dtype(at::kLong));
  auto i1 = at::randint(10, {4, 4}, at::TensorOptions{}.dtype(at::kLong));
  c10::List<c10::optional<at::Tensor>> indices{c10::nullopt, i0, i1};

  auto result = ((FnType)func)(tensor, indices);
  auto expect = at::index(tensor, indices);

  ASSERT_TRUE(expect.equal(result));
}

TEST(JITTest, ArgMinTest) {
  using FnType =
      at::Tensor (*)(const at::Tensor &, c10::optional<int64_t>, bool);

  auto jit = create_jit((FnType)at::argmin);
  auto symbol = llvm::cantFail(jit->lookup(WrappedFn));
  auto func = (void *)symbol.getAddress();

  auto tensor = at::randint(10, {1, 10, 10});
  auto dim = 0;
  auto keepdim = true;

  auto result = ((FnType)func)(tensor, dim, keepdim);
  auto expect = at::argmin(tensor, dim, keepdim);

  ASSERT_TRUE(expect.equal(result));
}

TEST(JITTest, SumTest) {
  using FnType = at::Tensor (*)(const at::Tensor &, at::IntArrayRef, bool,
                                c10::optional<at::ScalarType>);

  auto jit = create_jit((FnType)at::sum);
  auto symbol = llvm::cantFail(jit->lookup(WrappedFn));
  auto func = (void *)symbol.getAddress();

  auto tensor = at::randint(10, {1, 10, 10});
  auto dim = std::vector<long>{0, 1};
  auto keepdim = true;
  auto dtype = at::ScalarType::Float;

  auto result = ((FnType)func)(tensor, dim, keepdim, dtype);
  auto expect = at::sum(tensor, dim, keepdim, dtype);

  ASSERT_TRUE(expect.equal(result));
}

TEST(JITTest, ChunkTest) {
  using FnType =
      std::vector<at::Tensor> (*)(const at::Tensor &, int64_t, int64_t);

  auto jit = create_jit((FnType)at::chunk);
  auto symbol = llvm::cantFail(jit->lookup(WrappedFn));
  auto func = (void *)symbol.getAddress();

  auto tensor = at::randint(10, {10, 16});
  auto chunks = 4;
  auto dim = 1;

  auto result = ((FnType)func)(tensor, chunks, dim);
  auto expect = at::chunk(tensor, chunks, dim);

  ASSERT_EQ(expect.size(), result.size());

  auto size = expect.size();
  for (size_t i = 0; i < size; i++) {
    ASSERT_TRUE(expect[i].equal(result[i]));
  }
}
