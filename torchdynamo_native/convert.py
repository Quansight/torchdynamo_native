# Load PyTorch libraries.
import torch

from typing import (
    Any,
)

from torchgen.model import (
    BaseTy,
    BaseType,
    ListType,
    OptionalType,
    Type
)

from torchdynamo_native._C import Function, Value

REDUCTION_MEAN = "Mean"


def str_to_py(thing: str, ty: Type) -> Any:
    """Parses the default string into a Python value.
    """
    if ty == BaseType(BaseTy.int):
        # Special case: at::Reduction.
        # Defer translation to Function.
        if thing == REDUCTION_MEAN:
            return thing

        # Otherwise, we try to parse it into an int.
        try:
            return int(thing)
        except ValueError:
            pass

    elif ty == BaseType(BaseTy.float):
        try:
            return float(thing)
        except ValueError:
            pass

    elif ty == BaseType(BaseTy.str):
        return thing

    elif ty == BaseType(BaseTy.bool):
        if thing == "True":
            return True
        elif thing == "False":
            return False

    elif ty == BaseType(BaseTy.Scalar):
        # Parse as either 'int' or 'float', depending on the
        # presence of a '.'.
        if "." not in thing:
            return str_to_py(thing, BaseType(BaseTy.int))
        else:
            return str_to_py(thing, BaseType(BaseTy.float))

    elif ty == BaseType(BaseTy.MemoryFormat):
        if (thing == "contiguous_format"):
            return torch.contiguous_format

    elif isinstance(ty, OptionalType):
        if thing == "None":
            return None
        else:
            return str_to_py(thing, ty.elem)

    elif isinstance(ty, ListType):
        if ty.elem == BaseType(BaseTy.int):
            if len(thing) == 2:
                return []
            if len(thing) > 2:
                return [int(x) for x in thing[1:-1].split(",")]

    raise ValueError(f"can't build {ty} from str: {thing}")


def nullopt_for_type(ty: Type, fn: Function) -> Value:
    if ty == BaseType(BaseTy.Generator):
        return fn.build_nullopt_generator()

    elif ty == BaseType(BaseTy.ScalarType):
        return fn.build_nullopt_scalar_type()

    elif ty == BaseType(BaseTy.Tensor):
        return fn.build_nullopt_tensor()

    elif ty == BaseType(BaseTy.bool):
        return fn.build_nullopt_bool()

    elif ty == BaseType(BaseTy.int):
        return fn.build_nullopt_int()

    elif ty == BaseType(BaseTy.float):
        return fn.build_nullopt_float()

    elif ty == BaseType(BaseTy.str):
        return fn.build_nullopt_str()

    elif ty == BaseType(BaseTy.MemoryFormat):
        return fn.build_nullopt_memory_format()

    elif ty == BaseType(BaseTy.Device):
        return fn.build_nullopt_device()

    elif ty == BaseType(BaseTy.Layout):
        return fn.build_nullopt_layout()

    raise ValueError(f"can't build nullopt for type: {ty}")


def opt_for_type(ty: Type, value: Value, fn: Function) -> Value:
    if ty == BaseType(BaseTy.int):
        return fn.build_optional_lit_int(value)

    elif ty == BaseType(BaseTy.float):
        return fn.build_optional_lit_float(value)

    elif ty == BaseType(BaseTy.MemoryFormat):
        return fn.build_optional_lit_memory_format(value)

    elif ty == BaseType(BaseTy.str):
        return fn.build_optional_lit_str(value)

    elif ty == BaseType(BaseTy.Scalar):
        return fn.build_optional_scalar(value)

    elif isinstance(ty, ListType):
        elty = ty.elem

        if elty == BaseType(BaseTy.int):
            return fn.build_optional_arrayref_int(value)

    raise ValueError(f"can't build optional for type: {ty}")


def py_to_value(thing: Any, ty: Type, fn: Function) -> Value:
    """Translates a Python value into a Value.
    """
    if ty == BaseType(BaseTy.int):
        # Special case: at::Reduction.
        if thing == REDUCTION_MEAN:
            # TODO return fn.build_int_reduction_mean()
            pass
        else:
            # Otherwise, we try to parse it into an int.
            return fn.build_int(thing)

    elif ty == BaseType(BaseTy.float):
        return fn.build_float(thing)

    elif ty == BaseType(BaseTy.str):
        return fn.build_str(thing)

    elif ty == BaseType(BaseTy.bool):
        return fn.build_bool(thing)

    elif ty == BaseType(BaseTy.Scalar):
        if isinstance(thing, int):
            return fn.build_scalar_int(thing)
        else:
            return fn.build_scalar_float(thing)

    elif ty == BaseType(BaseTy.MemoryFormat):
        return fn.build_memory_format(thing)

    elif isinstance(ty, OptionalType):
        if thing is None:
            return nullopt_for_type(ty.elem, fn)
        else:
            value = py_to_value(thing, ty.elem, fn)
            return opt_for_type(ty.elem, value, fn)

    elif isinstance(ty, ListType):
        if ty.elem == BaseType(BaseTy.int):
            return fn.build_arrayref_lit_int([fn.build_int(x) for x in thing])

    raise ValueError(f"can't build value for {ty} from: {thing}")


def torch_isinstance(thing: Any, ty: Type) -> bool:
    if ty == BaseType(BaseTy.Tensor):
        return isinstance(thing, torch.Tensor)

    elif ty == BaseType(BaseTy.int):
        # Special case: at::Reduction.
        if thing == REDUCTION_MEAN:
            return True
        # Otherwise, we just check if it is an integer.
        return isinstance(thing, int)

    elif ty == BaseType(BaseTy.Dimname):
        return isinstance(thing, str)

    elif ty == BaseType(BaseTy.float):
        return isinstance(thing, float)

    elif ty == BaseType(BaseTy.str):
        return isinstance(thing, str)

    elif ty == BaseType(BaseTy.bool):
        return isinstance(thing, bool)

    elif ty == BaseType(BaseTy.Scalar):
        return (
            torch_isinstance(thing, BaseType(BaseTy.int))
            or torch_isinstance(thing, BaseType(BaseTy.float))
        )

    elif ty == BaseType(BaseTy.MemoryFormat):
        return isinstance(thing, torch.memory_format)

    elif isinstance(ty, OptionalType):
        return (
            thing is None
            or torch_isinstance(thing, ty.elem)
        )

    elif isinstance(ty, ListType):
        return (
            isinstance(thing, (list, tuple))
            and all(torch_isinstance(x, ty.elem) for x in thing)
        )

    raise ValueError(f"couldn't check instance for type: {ty}")
