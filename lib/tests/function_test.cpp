#define TDNAT_LOAD_REGISTRY
#include "jit_utils.h"

#include <tdnat/function.h>

#include <ATen/core/Formatting.h>
#include <ATen/ops/add.h>
#include <ATen/ops/argmin.h>
#include <ATen/ops/cat.h>
#include <ATen/ops/chunk.h>
#include <ATen/ops/index.h>
#include <ATen/ops/multinomial.h>
#include <ATen/ops/randint.h>
#include <ATen/ops/rand.h>
#include <ATen/ops/sum.h>

#include <algorithm>
#include <iterator>

// NOLINTNEXTLINE
TEST(FunctionTest, AddTest)
{
  auto data = tdnat::FunctionData{"aten_add", 2, 1};
  auto fn = tdnat::Function::from_data(data);

  auto lhs = at::randint(10, {2, 2}); // NOLINT(cppcoreguidelines-avoid-magic-numbers)
  auto rhs = at::randint(10, {2, 2}); // NOLINT(cppcoreguidelines-avoid-magic-numbers)
  auto alpha = 5;                     // NOLINT(cppcoreguidelines-avoid-magic-numbers)
  auto expect = at::add(lhs, rhs, alpha);

  {
    auto lhs = fn->set_placeholder(0, "lhs");
    auto rhs = fn->set_placeholder(1, "rhs");

    auto alpha_int = fn->build_int<int64_t>(alpha);
    auto alpha_ = fn->build_scalar<int64_t, int64_t>(alpha_int);
    auto add = fn->add_call("add", "add.Tensor", {lhs, rhs, alpha_});

    fn->set_output_from_ref(add);
  }

  auto jitfn = fn->into_jit();
  auto result = jitfn.run({lhs, rhs});

  ASSERT_EQ(result.size(), 1);
  ASSERT_TRUE(expect.equal(*result[0]));
}

// NOLINTNEXTLINE
TEST(FunctionTest, CatTest)
{
  auto data = tdnat::FunctionData{"aten_cat", 2, 1};
  auto fn = tdnat::Function::from_data(data);
  auto t1 = at::randint(10, {1, 2, 2}); // NOLINT(cppcoreguidelines-avoid-magic-numbers)
  auto t2 = at::randint(10, {1, 2, 2}); // NOLINT(cppcoreguidelines-avoid-magic-numbers)
  auto dim = 0;
  auto expect = at::cat({t1, t2}, dim);

  {
    auto t1 = fn->set_placeholder(0, "t1");
    auto t2 = fn->set_placeholder(1, "t2");

    auto dim_ = fn->build_int<int64_t>(dim);
    auto tensor_size = fn->build_int<int64_t>(2);
    auto tensor_ptr = fn->build_array<at::Tensor>({
        fn->build_load(t1),
        fn->build_load(t2),
    });
    auto cat = fn->add_call("cat", "cat", {tensor_ptr, tensor_size, dim_});

    fn->set_output_from_ref(cat);
  }

  auto jitfn = fn->into_jit();
  auto result = jitfn.run({t1, t2});

  ASSERT_EQ(result.size(), 1);
  ASSERT_TRUE(expect.equal(*result[0]));
}

// NOLINTNEXTLINE
TEST(FunctionTest, IndexTest)
{
  auto data = tdnat::FunctionData{"aten_index", 3, 1};
  auto fn = tdnat::Function::from_data(data);
  auto tensor = at::randint(10, {1, 10, 10}); // NOLINT(cppcoreguidelines-avoid-magic-numbers)
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  auto i0 = at::randint(10, {4, 4}, at::TensorOptions{}.dtype(at::kLong));
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  auto i1 = at::randint(10, {4, 4}, at::TensorOptions{}.dtype(at::kLong));
  c10::List<c10::optional<at::Tensor>> indices{c10::nullopt, i0, i1};
  auto expect = at::index(tensor, indices);

  {
    auto tensor = fn->set_placeholder(0, "tensor");
    auto i1 = fn->set_placeholder(1, "i1");
    auto i2 = fn->set_placeholder(2, "i2");

    auto indices = fn->build_list<c10::optional<at::Tensor>>({
        fn->build_load(fn->build_nullopt<at::Tensor>()),
        fn->build_load(fn->build_optional<at::Tensor, const at::Tensor &>(i1)),
        fn->build_load(fn->build_optional<at::Tensor, const at::Tensor &>(i2)),
    });

    auto index = fn->add_call("index", "index.Tensor", {tensor, indices});

    fn->set_output_from_ref(index);
  }

  auto jitfn = fn->into_jit();
  auto result = jitfn.run({tensor, i0, i1});

  ASSERT_EQ(result.size(), 1);
  ASSERT_TRUE(expect.equal(*result[0]));
}

// NOLINTNEXTLINE
TEST(FunctionTest, ArgMinTest)
{
  auto data = tdnat::FunctionData{"aten_argmin", 1, 1};
  auto fn = tdnat::Function::from_data(data);
  auto tensor = at::randint(10, {1, 10, 10}); // NOLINT(cppcoreguidelines-avoid-magic-numbers)
  auto dim = 0;
  auto keepdim = true;
  auto expect = at::argmin(tensor, dim, keepdim);

  {
    auto tensor = fn->set_placeholder(0, "tensor");

    auto dim_ = fn->build_int<int64_t>(dim);
    auto dim_opt = fn->build_optional<int64_t, int64_t>(dim_);
    auto keepdim_ = fn->build_bool(keepdim);

    auto argmin = fn->add_call("argmin", "argmin", {tensor, dim_opt, keepdim_});

    fn->set_output_from_ref(argmin);
  }

  auto jitfn = fn->into_jit();
  auto result = jitfn.run({tensor});

  ASSERT_EQ(result.size(), 1);
  ASSERT_TRUE(expect.equal(*result[0]));
}

// NOLINTNEXTLINE
TEST(FunctionTest, SumTest)
{
  auto data = tdnat::FunctionData{"aten_sum", 1, 1};
  auto fn = tdnat::Function::from_data(data);
  auto tensor = at::randint(10, {1, 10, 10}); // NOLINT(cppcoreguidelines-avoid-magic-numbers)
  auto dim = std::vector<long>{0, 1};
  auto keepdim = true;
  auto type = at::ScalarType::Float;
  auto expect = at::sum(tensor, dim, keepdim, type);

  {
    auto tensor = fn->set_placeholder(0, "tensor");

    auto dim_ = fn->build_optionalarrayref<int64_t>({
        fn->build_int<int64_t>(dim[0]),
        fn->build_int<int64_t>(dim[1]),
    });

    auto type_ = fn->build_int_from_enum<int8_t>(type);
    auto type_opt = fn->build_optional<at::ScalarType, int8_t>(type_);

    auto keepdim = fn->build_bool(true);

    auto sum =
        fn->add_call("sum", "sum.dim_IntList", {tensor, dim_, keepdim, type_opt});

    fn->set_output_from_ref(sum);
  }

  auto jitfn = fn->into_jit();
  auto result = jitfn.run({tensor});

  ASSERT_EQ(result.size(), 1);
  ASSERT_TRUE(expect.equal(*result[0]));
}

// NOLINTNEXTLINE
TEST(FunctionTest, ChunkTest)
{
  auto data = tdnat::FunctionData{"aten_chunk", 1, 4};
  auto fn = tdnat::Function::from_data(data);
  auto tensor = at::randint(10, {10, 16}); // NOLINT(cppcoreguidelines-avoid-magic-numbers)
  auto chunks = 4;
  auto dim = 1;
  auto expect = at::chunk(tensor, chunks, dim);

  {
    auto tensor = fn->set_placeholder(0, "tensor");

    auto chunks_ = fn->build_int<int64_t>(chunks);
    auto dim_ = fn->build_int<int64_t>(dim);

    auto chunk = fn->add_call("chunk", "chunk", {tensor, chunks_, dim_});

    std::vector<tdnat::Value> values;

    for (int64_t i = 0; i < chunks; i++) {
      values.push_back(fn->build_vector_index<at::Tensor>(chunk, fn->build_int<int64_t>(i)));
    }

    fn->set_output_from_refs(values);
  }

  auto jitfn = fn->into_jit();
  auto result = jitfn.run({tensor});

  ASSERT_EQ(result.size(), chunks);
  for (size_t i = 0; i < chunks; i++) {
    ASSERT_TRUE(expect[i].equal(*result[i]));
  }
}

// NOLINTNEXTLINE
TEST(FunctionTest, MultinomialTest)
{
  auto data = tdnat::FunctionData{"aten_multinomial", 1, 1};
  auto fn = tdnat::Function::from_data(data);
  auto tensor = at::rand({10}); // NOLINT(cppcoreguidelines-avoid-magic-numbers)
  auto samples = 4;
  auto replacement = false;
  auto generator = c10::optional<at::Generator>(c10::nullopt);
  auto expect = at::multinomial(tensor, samples, replacement, generator);

  {
    auto tensor = fn->set_placeholder(0, "tensor");

    auto samples_ = fn->build_int<int64_t>(samples);
    auto replacement_ = fn->build_bool(replacement);
    auto generator = fn->build_nullopt<at::Generator>();

    auto result =
        fn->add_call("multinomial", "multinomial", {tensor, samples_, replacement_, generator});

    fn->set_output_from_ref(result);
  }

  auto jitfn = fn->into_jit();
  auto result = jitfn.run({tensor});

  ASSERT_EQ(expect.sizes(), result[0]->sizes());
  ASSERT_EQ(expect.scalar_type(), result[0]->scalar_type());
  ASSERT_EQ(expect.device(), result[0]->device());
}
