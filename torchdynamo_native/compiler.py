import logging
import torch
import torch.fx

from typing import Any, Callable, Dict, List, Sequence, Tuple, Union

from torchgen.api.types import Binding
from torchgen.utils import concatMap

from torchdynamo_native._C import Function, Value
from torchdynamo_native.buildhelper.codegen.kernel import CABIArgument
from torchdynamo_native.convert import py_to_value, str_to_py
from torchdynamo_native.schema import AlignedArg, align_arguments, find_native_function
from torchdynamo_native.utils import get_kernel, native_function_overloaded_name

logger = logging.getLogger(__name__)


def resolve_target_name(target: Union[str, Callable]) -> str:
    if isinstance(target, str):
        return target

    module, name = target.__module__, target.__name__

    if module == "torch":
        return name
    elif module == "_operator":
        if name == "truediv":
            return "div"
        else:
            return name
    else:
        raise ValueError(f"couldn't resolve target: {module}.{name}")


def build_debug_target_name(target: Union[str, Callable]) -> str:
    if isinstance(target, str):
        return target
    return f"{target.__module__}.{target.__name__}"


def get_bindings_and_arguments(
        arguments: Sequence[AlignedArg],
        bindings: Sequence[Binding]
) -> List[Tuple[Binding, AlignedArg]]:
    bindings_and_arguments = {}
    binding_for = {b.name: (i, b) for i, b in enumerate(bindings)}

    for arg in arguments:
        name = arg.param.name
        if name in binding_for:
            bindings_and_arguments[binding_for[name]] = arg
        else:
            raise ValueError(f"can't lower {name}: {arg.param}")

    return [(b, arg) for ((_, b), arg) in sorted(bindings_and_arguments.items())]


def gen_values(
        bindings_and_arguments: List[Tuple[Binding, AlignedArg]],
        fn: Function
) -> List[Value]:

    def binding_to_values(pair: Tuple[Binding, AlignedArg]) -> List[Value]:
        binding, arg = pair
        py = str_to_py(arg.value, arg.param.type) if arg.default else arg.value
        return [
            py_to_value(py, c_abi.type, fn)
            for c_abi in CABIArgument.from_binding(binding)
        ]

    return list(concatMap(binding_to_values, bindings_and_arguments))


def replace_nodes_by_values(thing: Any, value_for: Dict[torch.fx.Node, Value]) -> Any:
    if isinstance(thing, torch.fx.Node):
        return value_for[thing]
    elif isinstance(thing, (list, tuple)):
        return type(thing)(replace_nodes_by_values(t, value_for) for t in thing)
    elif isinstance(thing, dict):
        return {k: replace_nodes_by_values(v, value_for) for k, v in thing.items()}
    else:
        return thing


def llvmjit(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]) -> Callable:
    g = gm.graph

    in_tensors = sum(int(n.op == "placeholder") for n in g.nodes)
    out_tensors = [len(n.args[0]) for n in g.nodes if n.op == "output"]
    assert len(out_tensors) == 1, f"expected only 1 output node. Got: {len(out_tensors)}"

    function = Function(f"dynamo_{id(gm)}", in_tensors, out_tensors[0])
    value_for: Dict[torch.fx.Node, Value] = {}

    placeholders_index = 0

    for node in g.nodes:
        if node.op == "placeholder":
            value_for[node] = function.set_placeholder(placeholders_index)
            placeholders_index += 1

        elif node.op in ("call_function", "call_method"):
            op_name = resolve_target_name(node.target)
            f = find_native_function(op_name, node.args, node.kwargs)

            logger.debug(f"[{node.op}] found function for {build_debug_target_name(node.target)}:")
            logger.debug(f"""{" " * 4}{f.func}""")

            kernel = get_kernel(f)
            arguments = [
                arg.with_value(replace_nodes_by_values(arg.value, value_for))
                for arg in align_arguments(f.func.arguments.flat_all, node.args, node.kwargs)
            ]
            bindings_and_arguments = get_bindings_and_arguments(arguments, kernel.sig().arguments())
            arguments_values = gen_values(bindings_and_arguments, function)

            value_for[node] = function.add_call(
                native_function_overloaded_name(f),
                arguments_values
            )

        elif node.op == "output":
            assert isinstance(node.args[0], tuple), f"unexpected type: {type(node.args[0])}"
            outputs = list(node.args[0])

            assert all(isinstance(o, torch.fx.Node) for o in outputs), (
                f"invalid output type: {[type(o) for o in outputs]}"
            )

            output_values = replace_nodes_by_values(outputs, value_for)
            function.set_output_from_refs(output_values)

        else:
            raise ValueError(f"invalid fx.Node operation: {node.op}")

    jit_function = function.into_jit()

    def wrapper(*args):
        return jit_function.run(list(args))  # type: ignore

    return wrapper
