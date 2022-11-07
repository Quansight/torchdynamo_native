import dataclasses
import torch

from torch.testing._internal.common_methods_invocations import OpInfo
from torch.testing._internal.opinfo.core import SampleInput
from torchgen.model import NativeFunction

DTYPE_SET = [
    torch.float,
    torch.long,
    torch.complex128,
]

NONDETERMINISTIC_OPERATIONS = {
    "multinomial",
    "randn",
    "randn_like",
    "randint",
    "randint_like",
    "rand",
    "rand_like",
    "normal",
    "empty",
    "empty_like",
    "new_empty",
    "bernoulli",
    "rrelu",
    "rrelu_",
}


class NativeFunctionNotFoundError(Exception):
    ...


class OverloadNotRegisteredError(Exception):
    ...


class EagerFailedError(Exception):
    ...


class VoidFunctionError(Exception):
    ...


class UnexpectedReturnValueError(Exception):
    ...


@dataclasses.dataclass
class TestData:
    op: OpInfo
    f: NativeFunction
    sample: SampleInput
    dtype: torch.dtype


def pick_dtype_for(op: OpInfo, device) -> torch.dtype:
    for dtype in DTYPE_SET:
        if op.supports_dtype(dtype, device):
            return dtype
    raise ValueError(f"couldn't find dtype. Available: {op.supported_dtypes(device)}")


def equals(lhs: torch.Tensor, rhs: torch.Tensor, dtype: torch.dtype) -> bool:
    if dtype == torch.int:
        return torch.equal(lhs, rhs)
    return bool(torch.isclose(lhs.to_dense(), rhs.to_dense(), equal_nan=True).all())


def similar(lhs: torch.Tensor, rhs: torch.Tensor) -> bool:
    if lhs.dtype != rhs.dtype:
        return False
    if lhs.shape != rhs.shape:
        return False
    return True
