# Load PyTorch libraries.
import torch

from typing import (
    Any,
    Callable,
    Dict,
    List,
    Sequence,
    Tuple,
)

from torchgen.model import (
    BaseTy,
    BaseType,
    ListType,
    OptionalType,
    Type,
)

from torchgen.api.types import (
    ArrayRefCType,
    BaseCType,
    BaseCppType,
    ConstRefCType,
    ListCType,
    MutRefCType,
    OptionalCType,
    boolT,
    charT,
    deviceT,
    doubleT,
    generatorT,
    layoutT,
    longT,
    memoryFormatT,
    optionalIntArrayRefT,
    scalarT,
    scalarTypeT,
    stringT,
    tensorT
)

from torchdynamo_native._C import Function, Value
from torchdynamo_native.buildhelper.codegen.kernel import CTypeWithPointer, CharT, ConstPointerCType

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
        if thing[0] == thing[-1] == "'":
            return thing[1:-1]
        else:
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


def get_values_for_array(
        thing: Sequence[Any],
        ctype: CTypeWithPointer,
        fn: Function
) -> Tuple[Value, Value]:
    assert isinstance(ctype, BaseCType), f"can only build nullopt for BaseCType. Got: {ctype}"

    def chain(f1: Callable, f2: Callable) -> Callable:
        def chained(*args, **kwargs):
            return f2(f1(*args, **kwargs))
        return chained

    cpp_type_table: Dict[BaseCppType, Tuple[type, Callable, Callable[[List[Value]], Value]]] = {
        longT:   (int,    py_to_value,                                  fn.build_array_int),
        scalarT: (object, chain(py_to_value, fn.build_load_for_scalar), fn.build_array_scalar),
        tensorT: (Value,  chain(py_to_value, fn.build_load),            fn.build_array_tensor),
    }

    if ctype.type in cpp_type_table:
        py_type, map_fn, build_fn = cpp_type_table[ctype.type]

        assert all(isinstance(x, py_type) for x in thing), (
            f"all elements should be {py_type.__name__} objects. "
            f"Got: {thing} ({type(thing)})"
        )

        size = fn.build_int(len(thing))
        array = [map_fn(x, ctype, fn) for x in thing]

        return build_fn(array), size

    raise ValueError(f"can't build array for type: {ctype}")


def get_value_for_nullopt(ctype: CTypeWithPointer, fn: Function) -> Value:
    assert isinstance(ctype, BaseCType), f"can only build nullopt for BaseCType. Got: {ctype}"

    cpp_type_table: Dict[BaseCppType, Callable[[], Value]] = {
        generatorT:    fn.build_nullopt_generator,
        scalarTypeT:   fn.build_nullopt_scalar_type,
        tensorT:       fn.build_nullopt_tensor,
        boolT:         fn.build_nullopt_bool,
        longT:         fn.build_nullopt_int,
        doubleT:       fn.build_nullopt_float,
        stringT:       fn.build_nullopt_str,
        memoryFormatT: fn.build_nullopt_memory_format,
        deviceT:       fn.build_nullopt_device,
        layoutT:       fn.build_nullopt_layout,
        scalarT:       fn.build_nullopt_scalar,
    }

    if ctype.type in cpp_type_table:
        return cpp_type_table[ctype.type]()

    raise ValueError(f"can't build nullopt for type: {ctype}")


def get_value_for_optional(
        ctype: CTypeWithPointer,
        thing: Any,
        fn: Function
) -> Value:
    if thing is None:
        return get_value_for_nullopt(ctype, fn)

    if ctype == BaseCType(scalarT):
        if isinstance(thing, int):
            return fn.build_optional_scalar_int(py_to_value(thing, BaseCType(longT), fn))

        elif isinstance(thing, float):
            return fn.build_optional_scalar_float(py_to_value(thing, BaseCType(doubleT), fn))

    elif ctype == BaseCType(stringT):
        ptr = py_to_value(thing, ConstPointerCType(BaseCType(CharT)), fn)
        size = py_to_value(thing, BaseCType(longT), fn)
        return fn.build_optional_str(ptr, size)

    elif isinstance(ctype, BaseCType):
        cpp_type_table: Dict[BaseCppType, Tuple[type, Callable[[Value], Value]]] = {
            boolT:         (bool,                fn.build_optional_bool),
            longT:         (int,                 fn.build_optional_int),
            doubleT:       (float,               fn.build_optional_float),
            memoryFormatT: (torch.memory_format, fn.build_optional_memory_format),
        }

        if ctype.type in cpp_type_table:
            py_type, build_fn = cpp_type_table[ctype.type]

            assert isinstance(thing, py_type), (
                f"{ctype} can only be constructed from an {py_type.__name__} literal. "
                f"Got: {thing} ({type(thing)})"
            )

            thing_as_value = py_to_value(thing, ctype, fn)
            return build_fn(thing_as_value)

    elif isinstance(ctype, ArrayRefCType):
        ptr, size = get_values_for_array(thing, ctype, fn)

        if ctype.elem == BaseCType(longT):
            return fn.build_optional_arrayref_int(ptr, size)

    raise ValueError(f"can't build optional for type: {ctype}")


def py_to_value(thing: Any, ctype: CTypeWithPointer, fn: Function) -> Value:
    if isinstance(thing, Value):
        return thing

    elif ctype == BaseCType(boolT):
        return fn.build_bool(thing)

    elif ctype == BaseCType(charT):
        if isinstance(thing, torch.memory_format):
            return fn.build_int_from_memory_format(thing)

    elif ctype == BaseCType(longT):
        if isinstance(thing, int):
            return fn.build_int(thing)

        elif isinstance(thing, (tuple, list, str)):
            return fn.build_int(len(thing))

    elif ctype == BaseCType(doubleT):
        return fn.build_float(thing)

    elif ctype == BaseCType(memoryFormatT):
        return fn.build_int_from_memory_format(thing)

    elif ctype == BaseCType(scalarT):
        if isinstance(thing, int):
            return fn.build_scalar_int(py_to_value(thing, BaseCType(longT), fn))

        if isinstance(thing, float):
            return fn.build_scalar_float(py_to_value(thing, BaseCType(doubleT), fn))

    elif ctype == BaseCType(optionalIntArrayRefT):
        if thing is None:
            return fn.build_nullopt_optionalarrayref_int()
        else:
            return fn.build_optionalarrayref_int([fn.build_int(x) for x in thing])

    elif ctype == ListCType(OptionalCType(BaseCType(tensorT))):
        return fn.build_list_optional_tensor([fn.build_load(x) for x in thing])

    elif isinstance(ctype, ConstPointerCType):
        if isinstance(thing, str):
            return fn.build_str(thing)

        elif isinstance(thing, (tuple, list)):
            return get_values_for_array(thing, ctype.elem, fn)[0]

    elif isinstance(ctype, OptionalCType):
        return get_value_for_optional(ctype.elem, thing, fn)

    elif isinstance(ctype, (MutRefCType, ConstRefCType)):
        return py_to_value(thing, ctype.elem, fn)

    raise ValueError(f"can't build value for {ctype.cpp_type()} from: {thing} ({type(thing)})")


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
