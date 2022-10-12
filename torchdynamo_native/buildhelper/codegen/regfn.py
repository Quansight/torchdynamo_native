from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Type, Union
from torchgen.api.types import CppSignature, CppSignatureGroup, DispatcherSignature

from torchgen.model import BackendIndex, DispatchKey, NativeFunction, Variant
from torchgen.context import with_native_function_and_indices

Signature = Union[CppSignature, DispatcherSignature]

def prefix() -> str:
    return "register_aten_operations"

def parameter_name() -> str:
    return "registry"

def parameter_type() -> str:
    return "ATenOpRegistry&"

def parameters() -> str:
    return f"{parameter_type()} {parameter_name()}"

@dataclass(frozen=True)
class Kernel(ABC):
    f: NativeFunction

    @classmethod
    def from_function_and_indices(cls: Type, f: NativeFunction, indices: Dict[DispatchKey, BackendIndex]) -> "Kernel":
        for key in cls.DISPATCH_KEY_PRIORITY_LIST:
            index = indices[key]
            if index.has_kernel(f) or f.structured_delegate:
                return DeviceKernel(f, str(key).lower())
        return DispatchKernel(f)

    @abstractmethod
    def namespace(self) -> str: ...

    @abstractmethod
    def sig(self) -> Signature: ...

    @abstractmethod
    def incl(self) -> str: ...

    @abstractmethod
    def name(self) -> str: ...

    DISPATCH_KEY_PRIORITY_LIST = [
        DispatchKey.CPU,
        DispatchKey.CompositeExplicitAutograd,
        DispatchKey.CompositeImplicitAutograd
    ]

@dataclass(frozen=True)
class DeviceKernel(Kernel):
    dev: str

    def namespace(self) -> str:
        return f"at::{self.dev}"

    def sig(self) -> Signature:
        return CppSignatureGroup.from_native_function(
            self.f, method=False, fallback_binding=False
        ).most_faithful_signature()

    def incl(self) -> str:
        return f"{self.f.root_name}_{self.dev}_dispatch"

    def name(self) -> str:
        return self.sig().name()

@dataclass(frozen=True)
class DispatchKernel(Kernel):
    def namespace(self) -> str:
        return f"at::_ops::{self.f.func.name.unambiguous_name()}"

    def sig(self) -> Signature:
        return DispatcherSignature.from_schema(self.f.func)

    def incl(self) -> str:
        return f"{self.f.root_name}_ops"

    def name(self) -> str:
        return "call"

@with_native_function_and_indices
def include(f: NativeFunction, indices: Dict[DispatchKey, BackendIndex]) -> str:
    kernel = Kernel.from_function_and_indices(f, indices)
    return f"#include <ATen/ops/{kernel.incl()}.h>"

@with_native_function_and_indices
def insert_entry(f: NativeFunction, indices: Dict[DispatchKey, BackendIndex]) -> str:
    fullname = str(f.func.name)
    kernel = Kernel.from_function_and_indices(f, indices)

    entry = "".join([
        f"MakeRef<{kernel.sig().ptr_type()}, &{kernel.namespace()}::{kernel.name()}>",
        f"::get(\"{fullname}\")"
    ])
    return f"{parameter_name()}.insert({entry});"
