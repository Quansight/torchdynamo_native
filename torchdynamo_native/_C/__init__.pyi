from typing import List

import torch


def operation_in_registry(opname: str) -> bool: ...


class Value:
    ...


class Function:
    ...
    def __init__(self, id: str, in_tensors: int, out_tensors: int) -> None: ...

    def set_placeholder(self, i: int, name: str) -> Value: ...
    def set_outputs(self, outputs: List[Value]) -> List[Value]: ...
    def set_output(self, output: Value) -> Value: ...

    def add_call(self, symbolname: str, opname: str, args: List[Value]) -> Value: ...

    def dump(self) -> None: ...
    def finalize(self) -> None: ...

    def build_bool(self, b: bool) -> Value: ...
    def build_scalar_type(self, ty: torch.dtype) -> Value: ...
    def build_memory_format(self, mf: torch.memory_format) -> Value: ...
    def build_optional_tensorlist(self, tensors: List[Value]) -> Value: ...
    def build_vector_at_tensor(self, v: Value, position: Value) -> Value: ...

    def build_scalar_int(self, n: int) -> Value: ...
    def build_scalar_float(self, n: int) -> Value: ...

    def build_int(self, n: int) -> Value: ...
    def build_float(self, n: float) -> Value: ...

    def build_arrayref_int(self, ints: List[Value]) -> Value: ...
    def build_arrayref_tensor(self, tensors: List[Value]) -> Value: ...
    def build_arrayref_lit_int(self, ints: List[Value]) -> Value: ...
    def build_arrayref_lit_tensor(self, tensors: List[Value]) -> Value: ...

    def build_nullopt_bool(self) -> Value: ...
    def build_nullopt_int(self) -> Value: ...
    def build_nullopt_float(self) -> Value: ...
    def build_nullopt_str(self) -> Value: ...
    def build_nullopt_scalar_type(self) -> Value: ...
    def build_nullopt_memory_format(self) -> Value: ...
    def build_nullopt_device(self) -> Value: ...
    def build_nullopt_layout(self) -> Value: ...
    def build_nullopt_generator(self) -> Value: ...
    def build_nullopt_tensor(self) -> Value: ...

    def build_optional_tensor(self, tensor: Value) -> Value: ...
    def build_optional_arrayref_int(self, arrayref_int: Value) -> Value: ...

    def build_optional_lit_int(self, val: Value) -> Value: ...
    def build_optional_lit_float(self, val: Value) -> Value: ...
    def build_optional_lit_scalar_type(self, val: Value) -> Value: ...
    def build_optional_lit_memory_format(self, val: Value) -> Value: ...

    def into_jit(self) -> "JITFunction": ...


class JITFunction:
    ...
    def run(self, in_tensors: List[torch.Tensor]) -> List[torch.Tensor]: ...

    def __call__(self, in_tensors: List[torch.Tensor]) -> List[torch.Tensor]: ...
