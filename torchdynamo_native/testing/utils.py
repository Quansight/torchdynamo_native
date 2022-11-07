import torch

from torch.testing._internal.common_methods_invocations import OpInfo

DTYPE_SET = [
    torch.float,
    torch.long,
    torch.complex128,
]


def pick_dtype_for(op: OpInfo, device) -> torch.dtype:
    for dtype in DTYPE_SET:
        if op.supports_dtype(dtype, device):
            return dtype
    raise ValueError(f"couldn't find dtype. Available: {op.supported_dtypes(device)}")
