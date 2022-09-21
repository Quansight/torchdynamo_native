#define TDNAT_LOAD_REGISTRY
#include "jit_utils.h"

#include <tdnat/function.h>

#include <ATen/core/Formatting.h>
#include <ATen/ops/add.h>
#include <ATen/ops/argmin.h>
#include <ATen/ops/cat.h>
#include <ATen/ops/chunk.h>
#include <ATen/ops/index.h>
#include <ATen/ops/randint.h>
#include <ATen/ops/sum.h>

#include <algorithm>
#include <iterator>

TEST(FunctionTest, AddTest)
{
  auto data = tdnat::FunctionData{"aten_add", 2, 1};
  tdnat::Function fn(data);

  auto lhs = at::randint(10, {2, 2});
  auto rhs = at::randint(10, {2, 2});
  auto alpha = 5;
  auto expect = at::add(lhs, rhs, alpha);

  {
    auto lhs = fn.set_placeholder(0, "lhs");
    auto rhs = fn.set_placeholder(1, "rhs");

    auto alpha_ = fn.build_scalar(alpha);
    auto add = fn.add_call("add", "add.Tensor", {lhs, rhs, alpha_});

    fn.set_output(add);
    fn.finalize();
  }

  auto jitfn = fn.into_jit();
  auto result = jitfn.run({lhs, rhs});

  ASSERT_EQ(result.size(), 1);
  ASSERT_TRUE(expect.equal(result[0]));
}

TEST(FunctionTest, CatTest)
{
  auto data = tdnat::FunctionData{"aten_cat", 2, 1};
  tdnat::Function fn(data);

  auto t1 = at::randint(10, {1, 2, 2});
  auto t2 = at::randint(10, {1, 2, 2});
  auto dim = 0;
  auto expect = at::cat({t1, t2}, dim);

  {
    auto t1 = fn.set_placeholder(0, "t1");
    auto t2 = fn.set_placeholder(1, "t2");

    auto dim_ = fn.build_integer<int64_t>(dim);
    auto tensorlist = fn.build_arrayref<at::Tensor>({t1, t2});
    auto cat = fn.add_call("cat", "cat", {tensorlist, dim_});

    fn.set_output(cat);
    fn.finalize();
  }

  auto jitfn = fn.into_jit();
  auto result = jitfn.run({t1, t2});

  ASSERT_EQ(result.size(), 1);
  ASSERT_TRUE(expect.equal(result[0]));
}

TEST(FunctionTest, IndexTest)
{
  auto data = tdnat::FunctionData{"aten_index", 3, 1};
  tdnat::Function fn(data);

  auto tensor = at::randint(10, {1, 10, 10});
  auto i0 = at::randint(10, {4, 4}, at::TensorOptions{}.dtype(at::kLong));
  auto i1 = at::randint(10, {4, 4}, at::TensorOptions{}.dtype(at::kLong));
  c10::List<c10::optional<at::Tensor>> indices{c10::nullopt, i0, i1};
  auto expect = at::index(tensor, indices);

  {
    auto tensor = fn.set_placeholder(0, "tensor");
    auto i1 = fn.set_placeholder(1, "i1");
    auto i2 = fn.set_placeholder(2, "i2");

    auto nullopt = fn.build_nullopt<at::Tensor>();
    auto i1_opt = fn.build_optional<at::Tensor>(i1);
    auto i2_opt = fn.build_optional<at::Tensor>(i2);
    auto indices = fn.build_optional_tensorlist({nullopt, i1_opt, i2_opt});

    auto index = fn.add_call("index", "index.Tensor", {tensor, indices});

    fn.set_output(index);
    fn.finalize();
  }

  auto jitfn = fn.into_jit();
  auto result = jitfn.run({tensor, i0, i1});

  ASSERT_EQ(result.size(), 1);
  ASSERT_TRUE(expect.equal(result[0]));
}

TEST(FunctionTest, ArgMinTest)
{
  auto data = tdnat::FunctionData{"aten_argmin", 1, 1};
  tdnat::Function fn(data);

  auto tensor = at::randint(10, {1, 10, 10});
  auto dim = 0;
  auto keepdim = true;
  auto expect = at::argmin(tensor, dim, keepdim);

  {
    auto tensor = fn.set_placeholder(0, "tensor");

    auto dim_ = fn.build_integer<int64_t>(dim);
    auto dim_opt = fn.build_optional_lit<int64_t>(dim_);
    auto keepdim_ = fn.build_bool(keepdim);

    auto argmin = fn.add_call("argmin", "argmin", {tensor, dim_opt, keepdim_});

    fn.set_output(argmin);
    fn.finalize();
  }

  auto jitfn = fn.into_jit();
  auto result = jitfn.run({tensor});

  ASSERT_EQ(result.size(), 1);
  ASSERT_TRUE(expect.equal(result[0]));
}

TEST(FunctionTest, SumTest)
{
  auto data = tdnat::FunctionData{"aten_sum", 1, 1};
  tdnat::Function fn(data);

  auto tensor = at::randint(10, {1, 10, 10});
  auto dim = std::vector<long>{0, 1};
  auto keepdim = true;
  auto type = at::ScalarType::Float;
  auto expect = at::sum(tensor, dim, keepdim, type);

  {
    auto tensor = fn.set_placeholder(0, "tensor");

    auto dim_ = std::vector<tdnat::Value>{};
    std::transform(dim.begin(), dim.end(), std::back_inserter(dim_), [&](long i) {
      return fn.build_integer<int64_t>(i);
    });
    auto dim_array = fn.build_arrayref_lit<int64_t>(dim_);

    auto type_ = fn.build_scalar_type(type);
    auto type_opt = fn.build_optional_lit<at::ScalarType>(type_);

    auto keepdim = fn.build_bool(true);

    auto sum = fn.add_call("sum", "sum.dim_IntList", {tensor, dim_array, keepdim, type_opt});

    fn.set_output(sum);
    fn.finalize();
  }

  auto jitfn = fn.into_jit();
  auto result = jitfn.run({tensor});

  ASSERT_EQ(result.size(), 1);
  ASSERT_TRUE(expect.equal(result[0]));
}

TEST(FunctionTest, ChunkTest)
{
  auto data = tdnat::FunctionData{"aten_chunk", 1, 4};
  tdnat::Function fn(data);

  auto tensor = at::randint(10, {10, 16});
  auto chunks = 4;
  auto dim = 1;
  auto expect = at::chunk(tensor, chunks, dim);

  {
    auto tensor = fn.set_placeholder(0, "tensor");

    auto chunks_ = fn.build_integer<int64_t>(chunks);
    auto dim_ = fn.build_integer<int64_t>(dim);

    auto chunk = fn.add_call("chunk", "chunk", {tensor, chunks_, dim_});

    std::vector<tdnat::Value> values;
    for (size_t i = 0; i < chunks; i++) {
      values.push_back(fn.build_vector_at_tensor(chunk, fn.build_integer(i)));
    }

    fn.set_outputs(values);
    fn.finalize();
  }

  auto jitfn = fn.into_jit();
  auto result = jitfn.run({tensor});

  ASSERT_EQ(result.size(), chunks);
  for (size_t i = 0; i < chunks; i++) {
    ASSERT_TRUE(expect[i].equal(result[i]));
  }
}
