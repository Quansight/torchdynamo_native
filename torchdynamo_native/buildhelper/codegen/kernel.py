from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Sequence, Type, Union

from torchgen.api import cpp, dispatcher
from torchgen.api.types import (
    ArrayCType,
    ArrayRefCType,
    BaseCType,
    BaseCppType,
    Binding,
    CType,
    ConstRefCType,
    CppSignature,
    CppSignatureGroup,
    DispatcherSignature,
    MutRefCType,
    OptionalCType,
    VectorCType,
    boolT,
    charT,
    deviceT,
    doubleT,
    intArrayRefT,
    iTensorListRefT,
    iOptTensorListRefT,
    layoutT,
    longT,
    memoryFormatT,
    optionalIntArrayRefT,
    optionalSymIntArrayRefT,
    scalarT,
    scalarTypeT,
    stringT,
    SymIntT,
    symIntArrayRefT,
    tensorT,
    tensorListT,
    voidT,
)
from torchgen.model import BackendIndex, DispatchKey, NativeFunction
from torchgen.utils import concatMap

Signature = Union[CppSignature, DispatcherSignature]
CharT = BaseCppType("", "char")
CTypeWithPointer = Union[CType, "ConstPointerCType"]


def is_c_enum_type(type: CType) -> bool:
    return isinstance(type, BaseCType) and type.type in (layoutT, memoryFormatT, scalarTypeT)


def is_c_array_ref_like_type(type: CType) -> bool:
    return (
        isinstance(type, ArrayRefCType)
        or (
            isinstance(type, BaseCType)
            and type.type in (stringT, intArrayRefT, tensorListT)
        )
    )


def is_c_ilist_ref_like_type(type: CType) -> bool:
    if isinstance(type, BaseCType):
        return type.type in (iTensorListRefT, iOptTensorListRefT)
    elif isinstance(type, ConstRefCType):
        return is_c_ilist_ref_like_type(type.elem)
    return False


def is_c_symint_array_ref_type(type: CType) -> bool:
    return isinstance(type, BaseCType) and type.type is symIntArrayRefT


def is_c_optional_symint_type(type: CType) -> bool:
    return type == OptionalCType(BaseCType(SymIntT))


def is_c_optional_symint_array_ref_type(type: CType) -> bool:
    return type == BaseCType(optionalSymIntArrayRefT)


def group_by_binding(arguments: Sequence["CABIArgument"]) -> Dict[Binding, List["CABIArgument"]]:
    groups = defaultdict(list)

    for arg in arguments:
        groups[arg.binding].append(arg)

    return groups


@dataclass(frozen=True)
class ConstPointerCType:
    elem: CType

    def cpp_type(self, *, strip_ref: bool = False) -> str:
        # Do not pass `strip_ref` recursively.
        return f"const {self.elem.cpp_type()}*"

    def cpp_type_registration_declarations(self) -> str:
        return f"const {self.elem.cpp_type_registration_declarations()}*"


@dataclass(frozen=True)
class CABIArgument:
    name: str
    type: CTypeWithPointer
    binding: Binding

    @staticmethod
    def from_binding_with_base_type(binding: Binding, type: BaseCType) -> List["CABIArgument"]:
        if type.type in (boolT, longT, doubleT):
            return [CABIArgument(name=binding.name, type=type, binding=binding)]

        if type.type is SymIntT:
            return [CABIArgument(name=binding.name, type=BaseCType(longT), binding=binding)]

        if is_c_enum_type(type):
            return [
                CABIArgument(name=f"{binding.name}__int", type=BaseCType(charT), binding=binding)
            ]

        if type.type in (deviceT, optionalIntArrayRefT):
            return [CABIArgument(name=binding.name, type=MutRefCType(type), binding=binding)]

        if type.type is optionalSymIntArrayRefT:
            return [
                CABIArgument(
                    name=f"{binding.name}__",
                    type=MutRefCType(BaseCType(optionalIntArrayRefT)),
                    binding=binding
                )
            ]

        if type.type in (
                intArrayRefT,
                stringT,
                tensorListT,
                iTensorListRefT,
                iOptTensorListRefT,
                symIntArrayRefT
        ):
            elem_type_of = {
                intArrayRefT:       longT,
                stringT:            CharT,
                tensorListT:        tensorT,
                iTensorListRefT:    tensorT,
                iOptTensorListRefT: OptionalCType(BaseCType(tensorT)),
                symIntArrayRefT:    longT,
            }

            return CABIArgument.from_binding_with_type(
                binding,
                ArrayRefCType(BaseCType(elem_type_of[type.type]))
            )

        raise ValueError(f"can't convert to C ABI type: {type}")

    @staticmethod
    def from_binding_with_type(binding: Binding, type: CType) -> List["CABIArgument"]:
        if isinstance(type, BaseCType):
            return CABIArgument.from_binding_with_base_type(binding, type)

        elif isinstance(type, (MutRefCType, ConstRefCType)):
            if is_c_ilist_ref_like_type(type.elem):
                return CABIArgument.from_binding_with_type(binding, type.elem)
            else:
                return [CABIArgument(name=binding.name, type=type, binding=binding)]

        elif type == OptionalCType(BaseCType(SymIntT)):
            return [
                CABIArgument(
                    name=f"{binding.name}__",
                    type=MutRefCType(OptionalCType(BaseCType(longT))),
                    binding=binding
                )
            ]

        elif isinstance(type, OptionalCType):
            return [CABIArgument(name=binding.name, type=MutRefCType(type), binding=binding)]

        elif isinstance(type, (ArrayCType, ArrayRefCType)):
            return [
                CABIArgument(
                    name=f"{binding.name}__ptr",
                    type=ConstPointerCType(type.elem),
                    binding=binding
                ),
                CABIArgument(
                    name=f"{binding.name}__size",
                    type=BaseCType(longT),
                    binding=binding
                ),
            ]

        else:
            raise ValueError(f"can't convert to C ABI type: {type}")

    @staticmethod
    def from_binding(binding: Binding) -> List["CABIArgument"]:
        return CABIArgument.from_binding_with_type(binding, binding.nctype.type)


@dataclass(frozen=True)
class Kernel(ABC):
    f: NativeFunction

    DISPATCH_KEY_PRIORITY_LIST = [
        DispatchKey.CPU,
        DispatchKey.CompositeExplicitAutograd,
        DispatchKey.CompositeImplicitAutograd
    ]

    @classmethod
    def from_function_and_indices(
            cls: Type,
            f: NativeFunction,
            indices: Dict[DispatchKey, BackendIndex]
    ) -> "Kernel":
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

    @abstractmethod
    def return_type_impl(self) -> CType: ...

    def return_type(self) -> CTypeWithPointer:
        type = self.return_type_impl()

        if isinstance(type, BaseCType) and type.type in (boolT, doubleT, longT, scalarTypeT, voidT):
            return type

        if isinstance(type, (MutRefCType, ConstRefCType)):
            return type

        if isinstance(type, BaseCType) and type.type in (tensorT, scalarT):
            return ConstPointerCType(type)

        elif type == VectorCType(BaseCType(tensorT)):
            return ConstPointerCType(type)

        else:
            raise ValueError(f"can't infer return type: {type}")


    def c_abi_name(self) -> str:
        return f"c_abi__{self.f.func.name.unambiguous_name()}"

    def c_abi_arguments(self) -> List[CABIArgument]:
        return list(concatMap(CABIArgument.from_binding, self.sig().arguments()))


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

    def return_type_impl(self) -> CType:
        return cpp.returns_type(self.f.func.returns)


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

    def return_type_impl(self) -> CType:
        return dispatcher.returns_type(self.f.func.returns)
