import torch
import torch.fx

from dataclasses import dataclass, replace
from typing import Any, Dict, List, Sequence

from torchgen.model import Argument, FunctionSchema, NativeFunction

from torchdynamo_native.convert import torch_isinstance
from torchdynamo_native.utils import NATIVE_FUNCTIONS_OVERLOAD_MAP, ExceptionGroup


@dataclass(frozen=True)
class AlignedArg:
    param: Argument
    value: Any
    default: bool = False

    def with_value(self, value: Any) -> "AlignedArg":
        return replace(self, value=value)


def align_arguments(
        parameters: Sequence[Argument],
        args: Sequence[Any],
        kwargs: Dict[str, Any]
) -> List[AlignedArg]:
    """Aligns the formal parameters with the given arguments.

    Tries to align each formal parameter with its corresponding argument.
    This function may fail if:

        - It didn't find a corresponding positional or keyword argument for
          a parameter without a default value.

        - It found both a positional and keyword argument for the same formal
          parameter.

    Thus, if successfull, this function guarantees that there's at least 1
    positional or keyword argument that corresponds to each formal parameter.
    """
    def align_to_parameter(i: int, param: Argument) -> AlignedArg:
        # The i-th parameter may be found as:
        if i < len(args):
            # - A positional argument.
            #     - Can't have multiple arguments for a parameter.
            if param.name in kwargs:
                raise ValueError(
                    f"positional argument {param.name} also passed as keyword-argument."
                )

            return AlignedArg(param, args[i])

        elif param.name in kwargs:
            # - A keyword argument.
            return AlignedArg(param, kwargs[param.name])
        elif param.default is not None:
            # - Not really found, but its default value is used.
            return AlignedArg(param, param.default, default=True)
        else:
            # Otherwise, it's missing.
            raise ValueError(f"missing {param.type} parameter: {param.name}")

    return [align_to_parameter(i, param) for i, param in enumerate(parameters)]


def check_schema_match(
        func: FunctionSchema,
        args: Sequence[Any],
        kwargs: Dict[str, Any]
) -> None:
    parameters = func.arguments.flat_all

    # Check whether we have enough arguments.
    aligned_arguments = align_arguments(parameters, args, kwargs)

    # Check whether we have too many arguments. Either:
    #   - There are more arguments than formal parameters.
    if len(parameters) < len(args) + len(kwargs):
        raise ValueError(
            "invalid number of parameters. "
            f"Expected {len(parameters)}. Got: {len(args) + len(kwargs)}."
        )

    #   - There are some extra keyword arguments not being used.
    if len(set(kwargs.keys()) - set([param.name for param in parameters])) != 0:
        raise ValueError(
            "unexpected keyword arguments: "
            f"{set(kwargs.keys()) - set([param.name for param in parameters])}"
        )

    # Check whether each parameter type matches the argument type.
    for param, arg in zip(parameters, aligned_arguments):
        # If we are using this parameter default value, we don't have
        # to check for its type.
        if arg.default:
            continue

        if not (isinstance(arg.value, torch.fx.Node) or torch_isinstance(arg.value, param.type)):
            raise ValueError(
                f"argument value not instance of {param.type}: {arg.value} ({type(arg.value)})"
            )


def find_native_function(
        op_name: str,
        args: Sequence[Any],
        kwargs: Dict[str, Any]
) -> NativeFunction:
    if op_name not in NATIVE_FUNCTIONS_OVERLOAD_MAP:
        raise ValueError(f"operation not in 'native_functions.yaml': {op_name}")

    exceptions = []

    for ovl in NATIVE_FUNCTIONS_OVERLOAD_MAP[op_name]:
        try:
            check_schema_match(ovl.f.func, args, kwargs)
            return ovl.f
        except Exception as e:
            exceptions.append(e)

    raise ExceptionGroup(f"could not find matching function overload for: {op_name}", exceptions)
