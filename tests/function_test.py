import torchdynamo_native as nat

import unittest
import torch

from typing import Callable, List


class Spec:
    def expected(self) -> List[torch.Tensor]: ...
    def jit(self) -> List[torch.Tensor]: ...


class SpecAdd(Spec):
    def __init__(self) -> None:
        self.lhs = torch.randint(10, (2, 2))
        self.rhs = torch.randint(10, (2, 2))
        self.alpha = 5

    def expected(self) -> List[torch.Tensor]:
        return [torch.add(self.lhs, self.rhs, alpha=self.alpha)]

    def jit(self) -> List[torch.Tensor]:
        fn = nat.Function("aten_add", 2, 1)

        lhs = fn.set_placeholder(0, "lhs")
        rhs = fn.set_placeholder(1, "rhs")

        alpha_ = fn.build_int(self.alpha)
        alpha_ = fn.build_scalar_int(alpha_)
        add = fn.add_call("add", "add.Tensor", [lhs, rhs, alpha_])

        fn.set_output_from_ref(add)
        return fn.into_jit()([self.lhs, self.rhs])


class SpecCat(Spec):
    def __init__(self) -> None:
        self.t1 = torch.randint(10, (1, 2, 2))
        self.t2 = torch.randint(10, (1, 2, 2))
        self.dim = 0

    def expected(self) -> List[torch.Tensor]:
        return [torch.cat([self.t1, self.t2], dim=self.dim)]

    def jit(self) -> List[torch.Tensor]:
        fn = nat.Function("aten_cat", 2, 1)

        t1 = fn.set_placeholder(0, "t1")
        t2 = fn.set_placeholder(1, "t2")

        dim = fn.build_int(self.dim)
        tensorlist_ptr = fn.build_array_tensor([
            fn.build_load(t1),
            fn.build_load(t2),
        ])
        tensorlist_size = fn.build_int(2)
        cat = fn.add_call("cat", "cat", [tensorlist_ptr, tensorlist_size, dim])

        fn.set_output_from_ref(cat)
        return fn.into_jit()([self.t1, self.t2])


class SpecIndex(Spec):
    def __init__(self) -> None:
        self.tensor = torch.randint(10, (1, 10, 10))
        self.i0 = torch.randint(10, (4, 4), dtype=torch.long)
        self.i1 = torch.randint(10, (4, 4), dtype=torch.long)
        self.indices = (..., self.i0, self.i1)

    def expected(self) -> List[torch.Tensor]:
        return [self.tensor[self.indices]]

    def jit(self) -> List[torch.Tensor]:
        fn = nat.Function("aten_index", 3, 1)

        tensor = fn.set_placeholder(0, "tensor")
        i1 = fn.set_placeholder(1, "i1")
        i2 = fn.set_placeholder(2, "i2")

        indices = fn.build_list_optional_tensor([
            fn.build_load(fn.build_nullopt_tensor()),
            fn.build_load(fn.build_optional_from_ref_tensor(i1)),
            fn.build_load(fn.build_optional_from_ref_tensor(i2)),
        ])

        index = fn.add_call("index", "index.Tensor", [tensor, indices])

        fn.set_output_from_ref(index)
        return fn.into_jit()([self.tensor, self.i0, self.i1])


class SpecArgMin(Spec):
    def __init__(self) -> None:
        self.tensor = torch.randint(10, (1, 10, 10))
        self.dim = 0
        self.keepdim = True

    def expected(self) -> List[torch.Tensor]:
        return [torch.argmin(self.tensor, self.dim, self.keepdim)]

    def jit(self) -> List[torch.Tensor]:
        fn = nat.Function("aten_argmin", 1, 1)

        tensor = fn.set_placeholder(0, "tensor")

        dim = fn.build_int(self.dim)
        dim_opt = fn.build_optional_int(dim)
        keepdim = fn.build_bool(self.keepdim)

        argmin = fn.add_call("argmin", "argmin", [tensor, dim_opt, keepdim])

        fn.set_output_from_ref(argmin)
        return fn.into_jit()([self.tensor])


class SpecSum(Spec):
    def __init__(self) -> None:
        self.tensor = torch.randint(10, (1, 10, 10))
        self.dim = [0, 1]
        self.keepdim = True
        self.dtype = torch.float

    def expected(self) -> List[torch.Tensor]:
        return [torch.sum(self.tensor, self.dim, self.keepdim, dtype=self.dtype)]

    def jit(self) -> List[torch.Tensor]:
        fn = nat.Function("aten_sum", 1, 1)

        tensor = fn.set_placeholder(0, "tensor")

        dim_ptr = fn.build_array_int([fn.build_int(d) for d in self.dim])
        dim_size = fn.build_int(len(self.dim))

        dtype = fn.build_int_from_scalar_type(self.dtype)
        dtype_opt = fn.build_optional_scalar_type(dtype)

        keepdim = fn.build_bool(self.keepdim)

        sum = fn.add_call("sum", "sum.dim_IntList", [tensor, dim_ptr, dim_size, keepdim, dtype_opt])

        fn.set_output_from_ref(sum)
        return fn.into_jit()([self.tensor])


class SpecChunk(Spec):
    def __init__(self) -> None:
        self.tensor = torch.randint(10, (10, 16))
        self.chunks = 4
        self.dim = 1

    def expected(self) -> List[torch.Tensor]:
        return torch.chunk(self.tensor, self.chunks, self.dim)

    def jit(self) -> List[torch.Tensor]:
        fn = nat.Function("aten_chunk", 1, 4)

        tensor = fn.set_placeholder(0, "tensor")

        chunks = fn.build_int(self.chunks)
        dim = fn.build_int(self.dim)

        chunk = fn.add_call("chunk", "chunk", [tensor, chunks, dim])

        values = [
            fn.build_vector_index(chunk, fn.build_int(i))
            for i in range(self.chunks)
        ]

        fn.set_output_from_refs(values)
        return fn.into_jit()([self.tensor])


class TestBindings(unittest.TestCase):
    def _test_spec(self, spec: Spec) -> None:
        exp = spec.expected()
        jit = spec.jit()

        self.assertEqual(len(exp), len(jit))

        for e, j in zip(exp, jit):
            self.assertTrue(e.equal(j))


if __name__ == "__main__":
    specifications = [
        SpecAdd,
        SpecCat,
        SpecIndex,
        SpecArgMin,
        SpecSum,
        SpecChunk,
    ]

    def create_test_for_spec(spec: type) -> Callable[[TestBindings], None]:
        def test(self: TestBindings):
            self._test_spec(spec())
        return test

    for spec in specifications:
        setattr(TestBindings, f"test_{spec.__name__}", create_test_for_spec(spec))

    unittest.main()
