from collections import defaultdict
import torchdynamo_native as nat

import unittest
import torch
import torch._dynamo as dynamo

from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    onlyCPU,
    ops
)

from torch.testing._internal.common_methods_invocations import (
    OpInfo,
    wrapper_set_seed,
    op_db
)

from torchdynamo_native.testing.utils import NONDETERMINISTIC_OPERATIONS, EagerFailedError, equals, similar

_WRAPPER_SET_SEED_OP = {
    "bernoulli",
    "multinomial",
    "nn.functional.dropout",
    "nn.functional.dropout2d",
    "nn.functional.dropout3d",
    "nn.functional.feature_alpha_dropout",
    "nn.functional.fractional_max_pool2d",
    "nn.functional.fractional_max_pool3d",
    "nn.functional.rrelu",
    "nn.functional._scaled_dot_product_attention",
    "normal",
    "pca_lowrank",
    "rand_like",
    "randint",
    "randint_like",
    "randn",
    "randn_like",
    "svd_lowrank",
    "uniform",
}


class TestDynamo(unittest.TestCase):
    @onlyCPU
    @ops(op_db, dtypes=[torch.float])
    def test(self, device: torch.device, dtype: torch.dtype, op: OpInfo):
        dtype = nat.testing.pick_dtype_for(op, device)

        if op.name in _WRAPPER_SET_SEED_OP:
            # Assertion error at 'torch._subclasses.meta_utils.py'
            raise unittest.SkipTest("wrapper function not supported")

        @dynamo.optimize(nat.llvmjit)  # type: ignore
        def wrapper(*args, **kwargs):
            return op(*args, **kwargs)

        def equals_(lhs: torch.Tensor, rhs: torch.Tensor) -> bool:
            return equals(lhs, rhs, dtype)

        compare = similar if op.name in NONDETERMINISTIC_OPERATIONS else equals_
        exceptions = defaultdict(list)
        success_count = 0

        for sample in op.sample_inputs(device, dtype, requires_grad=False):
            args = [sample.input, *sample.args]

            try:
                expected = op(*args, **sample.kwargs)
                assert compare(expected, op(*args, **sample.kwargs))
                assert compare(expected, op(*args, **sample.kwargs))
                assert compare(expected, op(*args, **sample.kwargs))
            except Exception as e:
                exceptions[type(e)].append(EagerFailedError(e))
                continue

            success_count += 1
            result = wrapper(*args, **sample.kwargs)

            if op.name in NONDETERMINISTIC_OPERATIONS:
                self.assertTrue(similar(expected, result))
            else:
                self.assertEqual(expected, result)

        if success_count == 0 and len(exceptions) > 0:
            exp_list = ", ".join([
                f"{ty.__name__}({v[0]}): {len(v)}"
                for ty, v in exceptions.items()
            ])
            raise unittest.SkipTest(str(exp_list))


instantiate_device_type_tests(TestDynamo, globals())

if __name__ == "__main__":
    unittest.main()
