#include "jit_utils.h"

#include <tdnat/llvm_function_type.h>
#include <tdnat/ops.h>
#include <tdnat/utils.h>

#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Verifier.h>
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

static const char *const ModuleName = "jit_test";
static const char *const EntryFn = "entry_function";
static const char *const WrappedFn = "wrapped_function";
static const char *const EntryBlock = "entry";

template <typename Return, typename... Args>
std::unique_ptr<llvm::orc::LLJIT> create_jit(Return (*fn)(Args...))
{
  using AttrKind = llvm::Attribute::AttrKind;
  using type_ptr = Return (*)(Args...);
  using abi_type_ptr = typename ABIFunctionType<type_ptr>::type;

  auto ctx = std::make_unique<llvm::LLVMContext>();
  auto mod = std::make_unique<llvm::Module>(ModuleName, *ctx);
  auto jit = llvm::cantFail(llvm::orc::LLJITBuilder().create());

  auto entry_fn = llvm::Function::Create(
      LLVMFunctionType<abi_type_ptr>::get(*mod),
      llvm::Function::ExternalLinkage,
      EntryFn,
      mod.get()
  );

  auto wrapped_fn = llvm::Function::Create(
      LLVMFunctionType<abi_type_ptr>::get(*mod),
      llvm::Function::ExternalLinkage,
      WrappedFn,
      mod.get()
  );

  add_attributes<Return, Args...>(entry_fn);
  add_attributes<Return, Args...>(wrapped_fn);

  auto bb = llvm::BasicBlock::Create(entry_fn->getContext(), EntryBlock, entry_fn);
  llvm::IRBuilder<> builder(bb);

  std::vector<llvm::Value *> arguments;
  std::transform(
      entry_fn->arg_begin(),
      entry_fn->arg_end(),
      std::back_inserter(arguments),
      [](llvm::Argument &arg) { return (llvm::Value *)&arg; }
  );

  builder.CreateCall(wrapped_fn, arguments);
  builder.CreateRetVoid();

  llvm::verifyModule(*mod, &llvm::errs());
  llvm::cantFail(jit->addIRModule({std::move(mod), std::move(ctx)}));
  llvm::cantFail(jit->define(llvm::orc::absoluteSymbols(
      {{jit->mangleAndIntern(WrappedFn), llvm::JITEvaluatedSymbol::fromPointer(fn)}}
  )));

  return jit;
}

// NOLINTNEXTLINE
TEST(JITTest, AddTest)
{
  using FnType = at::Tensor (*)(const at::Tensor &, const at::Tensor &, const at::Scalar &);

  auto jit = create_jit(static_cast<FnType>(at::add));
  auto symbol = llvm::cantFail(jit->lookup(WrappedFn));

  // NOLINTNEXTLINE(performance-no-int-to-ptr)
  auto func = reinterpret_cast<void *>(symbol.getAddress());

  auto lhs = at::randint(10, {2, 2}); // NOLINT(cppcoreguidelines-avoid-magic-numbers)
  auto rhs = at::randint(10, {2, 2}); // NOLINT(cppcoreguidelines-avoid-magic-numbers)
  auto alpha = 5;                     // NOLINT(cppcoreguidelines-avoid-magic-numbers)

  auto result = (reinterpret_cast<FnType>(func))(lhs, rhs, alpha);
  auto expect = at::add(lhs, rhs, alpha);

  ASSERT_TRUE(expect.equal(result));
}

// NOLINTNEXTLINE
TEST(JITTest, CatTest)
{
  using FnType = at::Tensor (*)(at::ArrayRef<at::Tensor>, int64_t);

  auto jit = create_jit(static_cast<FnType>(at::cat));
  auto symbol = llvm::cantFail(jit->lookup(WrappedFn));

  // NOLINTNEXTLINE(performance-no-int-to-ptr)
  auto func = reinterpret_cast<void *>(symbol.getAddress());

  auto t1 = at::randint(10, {1, 2, 2}); // NOLINT(cppcoreguidelines-avoid-magic-numbers)
  auto t2 = at::randint(10, {1, 2, 2}); // NOLINT(cppcoreguidelines-avoid-magic-numbers)

  auto result = (reinterpret_cast<FnType>(func))({t1, t2}, 0);
  auto expect = at::cat({t1, t2}, 0);

  ASSERT_TRUE(expect.equal(result));
}

// NOLINTNEXTLINE
TEST(JITTest, IndexTest)
{
  using FnType = at::Tensor (*)(const at::Tensor &, const c10::List<c10::optional<at::Tensor>> &);

  auto jit = create_jit(static_cast<FnType>(at::index));
  auto symbol = llvm::cantFail(jit->lookup(WrappedFn));

  // NOLINTNEXTLINE(performance-no-int-to-ptr)
  auto func = reinterpret_cast<void *>(symbol.getAddress());

  auto tensor = at::randint(10, {1, 10, 10}); // NOLINT(cppcoreguidelines-avoid-magic-numbers)
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  auto i0 = at::randint(10, {4, 4}, at::TensorOptions{}.dtype(at::kLong));
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  auto i1 = at::randint(10, {4, 4}, at::TensorOptions{}.dtype(at::kLong));
  c10::List<c10::optional<at::Tensor>> indices{c10::nullopt, i0, i1};

  auto result = (reinterpret_cast<FnType>(func))(tensor, indices);
  auto expect = at::index(tensor, indices);

  ASSERT_TRUE(expect.equal(result));
}

// NOLINTNEXTLINE
TEST(JITTest, ArgMinTest)
{
  using FnType = at::Tensor (*)(const at::Tensor &, c10::optional<int64_t>, bool);

  auto jit = create_jit(static_cast<FnType>(at::argmin));
  auto symbol = llvm::cantFail(jit->lookup(WrappedFn));

  // NOLINTNEXTLINE(performance-no-int-to-ptr)
  auto func = reinterpret_cast<void *>(symbol.getAddress());

  auto tensor = at::randint(10, {1, 10, 10}); // NOLINT(cppcoreguidelines-avoid-magic-numbers)
  auto dim = 0;
  auto keepdim = true;

  auto result = (reinterpret_cast<FnType>(func))(tensor, dim, keepdim);
  auto expect = at::argmin(tensor, dim, keepdim);

  ASSERT_TRUE(expect.equal(result));
}

// NOLINTNEXTLINE
TEST(JITTest, SumTest)
{
  using FnType =
      at::Tensor (*)(const at::Tensor &, at::IntArrayRef, bool, c10::optional<at::ScalarType>);

  auto jit = create_jit(static_cast<FnType>(at::sum));
  auto symbol = llvm::cantFail(jit->lookup(WrappedFn));

  // NOLINTNEXTLINE(performance-no-int-to-ptr)
  auto func = reinterpret_cast<void *>(symbol.getAddress());

  auto tensor = at::randint(10, {1, 10, 10}); // NOLINT(cppcoreguidelines-avoid-magic-numbers)
  auto dim = std::vector<long>{0, 1};
  auto keepdim = true;
  auto dtype = at::ScalarType::Float;

  auto result = (reinterpret_cast<FnType>(func))(tensor, dim, keepdim, dtype);
  auto expect = at::sum(tensor, dim, keepdim, dtype);

  ASSERT_TRUE(expect.equal(result));
}

// NOLINTNEXTLINE
TEST(JITTest, ChunkTest)
{
  using FnType = std::vector<at::Tensor> (*)(const at::Tensor &, int64_t, int64_t);

  auto jit = create_jit(static_cast<FnType>(at::chunk));
  auto symbol = llvm::cantFail(jit->lookup(WrappedFn));

  // NOLINTNEXTLINE(performance-no-int-to-ptr)
  auto func = reinterpret_cast<void *>(symbol.getAddress());

  auto tensor = at::randint(10, {10, 16}); // NOLINT(cppcoreguidelines-avoid-magic-numbers)
  auto chunks = 4;
  auto dim = 1;

  auto result = (reinterpret_cast<FnType>(func))(tensor, chunks, dim);
  auto expect = at::chunk(tensor, chunks, dim);

  ASSERT_EQ(expect.size(), result.size());

  auto size = expect.size();
  for (size_t i = 0; i < size; i++) {
    ASSERT_TRUE(expect[i].equal(result[i]));
  }
}
// NOLINTEND(cppcoreguidelines-avoid-magic-numbers)
