from collections import defaultdict
from typing import Dict, List, Sequence

from torchgen.api.types import (
    BaseCType,
    VectorCType,
    scalarT,
    tensorT,
)

from torchgen.model import BackendIndex, DispatchKey, NativeFunction
from torchgen.context import with_native_function_and_indices

from torchdynamo_native.buildhelper.codegen.kernel import (
    CABIArgument,
    ConstPointerCType,
    Kernel,
    is_c_array_ref_like_type,
    is_c_enum_type
)


def pre_processing(arguments: Sequence[CABIArgument]) -> List[str]:
    body = []
    args_of_binding = defaultdict(list)

    for a in arguments:
        args_of_binding[a.binding].append(a)

    sorted_binding_and_args = sorted(list(args_of_binding.items()), key=lambda pair: pair[0].name)

    for binding, args in sorted_binding_and_args:
        type = binding.nctype.type

        if len(args) == 1 and is_c_enum_type(type):
            body.append(f"auto {binding.name} = static_cast<{type.cpp_type()}>({args[0].name});")

        elif len(args) == 2 and is_c_array_ref_like_type(type):
            args_map = {a.name.split("__")[1]: a for a in args}

            arg_ptr = args_map["ptr"].name
            arg_size = args_map["size"].name

            body.append(f"auto {binding.name} = {binding.type}({arg_ptr}, {arg_size});")

        elif len(args) == 1:
            pass

        else:
            arg_names = [a.name for a in args]
            raise ValueError(f"can't reconstruct argument: {binding.name} with {arg_names}")

    return body


def call_kernel_and_return(kernel: Kernel) -> List[str]:
    wrapped_args = ", ".join(a.name for a in kernel.sig().arguments())
    call = f"{kernel.namespace()}::{kernel.name()}({wrapped_args})"

    type = kernel.return_type()
    body = []

    if not isinstance(type, ConstPointerCType):
        return [f"return {call};"]

    if type.elem == BaseCType(tensorT):
        body = [f"auto out = new at::Tensor();"]

    elif type.elem == BaseCType(scalarT):
        body = [f"auto out = new at::Scalar();"]

    elif type.elem == VectorCType(BaseCType(tensorT)):
        body = [f"auto out = new std::vector<at::Tensor>();"]

    body.append(f"*out = {call};")
    body.append("return out;")
    return body


@with_native_function_and_indices
def c_abi(f: NativeFunction, indices: Dict[DispatchKey, BackendIndex]) -> str:
    kernel = Kernel.from_function_and_indices(f, indices)


    pre_processing_body = "\n".join(
        2 * " " + line
        for line in pre_processing(kernel.c_abi_arguments())
    )

    call_kernel_and_return_body = "\n".join(
        2 * " " + line
        for line in call_kernel_and_return(kernel)
    )

    wrapper_args = ", ".join(f"{a.type.cpp_type()} {a.name}" for a in kernel.c_abi_arguments())

    return f"""
{kernel.return_type().cpp_type()} {kernel.c_abi_name()}({wrapper_args}) {{
{pre_processing_body}
{call_kernel_and_return_body}
}}
"""


def prefix() -> str:
    return "register_aten_operations"


def parameter_name() -> str:
    return "registry"


def parameter_type() -> str:
    return "ATenOpRegistry&"


def parameters() -> str:
    return f"{parameter_type()} {parameter_name()}"


@with_native_function_and_indices
def include(f: NativeFunction, indices: Dict[DispatchKey, BackendIndex]) -> str:
    kernel = Kernel.from_function_and_indices(f, indices)
    return f"#include <ATen/ops/{kernel.incl()}.h>"


@with_native_function_and_indices
def insert_entry(f: NativeFunction, indices: Dict[DispatchKey, BackendIndex]) -> str:
    fullname = str(f.func.name)
    kernel = Kernel.from_function_and_indices(f, indices)

    arg_types = ", ".join(a.type.cpp_type() for a in kernel.c_abi_arguments())

    entry = "".join([
        f"MakeRef<{kernel.return_type().cpp_type()} (*)({arg_types}), &{kernel.c_abi_name()}>",
        f"::get(\"{fullname}\")"
    ])
    return f"{parameter_name()}.insert({entry});"