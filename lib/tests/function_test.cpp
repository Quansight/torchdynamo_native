#define TDNAT_LOAD_REGISTRY
#include "jit_utils.h"

#include <tdnat/function.h>

#include <ATen/ops/add.h>
#include <ATen/ops/randint.h>

TEST(FunctionTest, AddTest) {
  auto data = tdnat::FunctionData{"aten_add", 2, 1};
  tdnat::Function fn(data);

  {
    auto lhs = fn.set_placeholder(0, "lhs");
    auto rhs = fn.set_placeholder(1, "rhs");

    auto alpha = fn.build_scalar(5);
    auto add = fn.add_call("add", "add.Tensor", {lhs, rhs, alpha});

    fn.set_output(add);
    fn.finalize();
  }

  auto jitfn = fn.into_jit();
  at::Tensor result;

  auto lhs = at::randint(10, {2, 2});
  auto rhs = at::randint(10, {2, 2});
  auto alpha = 5;


  jitfn.run({lhs, rhs}, {&result, 1});
  auto expect = at::add(lhs, rhs, alpha);

  ASSERT_TRUE(expect.equal(result));
}
