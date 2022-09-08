#define TDNAT_LOAD_REGISTRY
#include "jit_utils.h"

#include <tdnat/function.h>

#include <ATen/ops/add.h>
#include <ATen/ops/randint.h>

TEST(FunctionTest, AddTest) {
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


  jitfn.run({lhs, rhs}, {&result, 1});
  auto expect = at::add(lhs, rhs, alpha);

  ASSERT_TRUE(expect.equal(result));
}
