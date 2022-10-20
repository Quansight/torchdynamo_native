import torchdynamo_native as nat

import unittest
import torch

from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    onlyCPU,
    ops
)

from torch.testing._internal.common_methods_invocations import (
    OpInfo,
    SampleInput,
    op_db
)

from torchgen.model import (
    BaseTy,
    BaseType,
    NativeFunction,
    OptionalType,
    ListType,
    Type,
)

from torchgen.api.types import (
    BaseCType,
    Binding,
    VectorCType,
    tensorT,
)
from torchgen.context import native_function_manager
from torchgen.utils import concatMap

from typing import (
    Any,
    Callable,
    Dict,
    List,
    Sequence,
    Tuple,
    Union
)

from torchdynamo_native.buildhelper.codegen.kernel import CABIArgument, ConstPointerCType, Kernel
from torchdynamo_native.utils import native_function_overloaded_name

DTYPE_SET = {
    torch.float,
    torch.long,
    torch.complex64,
}

NATIVE_FUNCTIONS, BACKEND_INDICES = nat.utils.parse_native_functions_yaml()
NATIVE_FUNCTIONS_MAP: Dict[str, NativeFunction] = {str(f.func.name): f for f in NATIVE_FUNCTIONS}

SKIP_LIST = [
    ("tensordot", "needs pre-processing before native kernel"),
    ("stft", "needs pre-processing before native kernel"),

    ("corrcoef", "non-deterministic behavior"),
    ("cov", "non-deterministic behavior"),
    ("where", "input is not the first argument"),

    ("multinomial", "non-deterministic operator"),
    ("randn", "non-deterministic operator"),
    ("randn_like", "non-deterministic operator"),
    ("randint", "non-deterministic operator"),
    ("randint_like", "non-deterministic operator"),
    ("rand", "non-deterministic operator"),
    ("rand_like", "non-deterministic operator"),
    ("normal", "non-deterministic operator"),
    ("empty", "non-deterministic operator"),
    ("empty_like", "non-deterministic operator"),
    ("new_empty", "non-deterministic operator"),
    ("bernoulli", "non-deterministic operator"),
]


def pick_dtype_for(op: OpInfo, device) -> torch.dtype:
    for dtype in DTYPE_SET:
        if op.supports_dtype(dtype, device):
            return dtype
    raise ValueError("couldn't find dtype")


def get_tensors_in(thing: Any) -> List[torch.Tensor]:
    if isinstance(thing, torch.Tensor):
        return [thing]
    if isinstance(thing, (list, tuple)) and not isinstance(thing, str):
        return list(concatMap(get_tensors_in, thing))
    if isinstance(thing, dict):
        return list(concatMap(get_tensors_in, thing.values()))
    return []


def at_least_one_tensor(value: Any) -> bool:
    return len(get_tensors_in(value)) > 0


def get_input_tensors(arguments: Sequence[nat.AlignedArg]) -> List[torch.Tensor]:
    return list(concatMap(lambda arg: get_tensors_in(arg.value), arguments))


def create_value_for(index: int, value: Any, name: str, type: Type, fn: nat.Function) -> nat.Value:
    if type == BaseType(BaseTy.Tensor):
        return fn.set_placeholder(index, name)

    elif type == OptionalType(BaseType(BaseTy.Tensor)):
        if value is None:
            return fn.build_nullopt_tensor()
        else:
            return fn.build_optional_from_ref_tensor(fn.set_placeholder(index, name))

    raise ValueError(f"expected Tensor or Optional[Tensor] type: got {type}")


def replace_value_for_inputs(
        arguments: Sequence[nat.AlignedArg],
        fn: nat.Function
) -> List[nat.AlignedArg]:
    placeholder_index = 0

    def replace_for_argument(arg: nat.AlignedArg) -> nat.AlignedArg:
        nonlocal placeholder_index

        new_value: Union[nat.Value, List[nat.Value]]
        type = arg.param.type

        if isinstance(type, ListType):
            assert isinstance(arg.value, (list, tuple))

            new_value = []
            for tensor in arg.value:
                new_value.append(create_value_for(
                    index=placeholder_index,
                    value=arg.value,
                    name=arg.param.name,
                    type=type.elem,
                    fn=fn
                ))
                placeholder_index += int(tensor is not None)

        else:
            new_value = create_value_for(placeholder_index, arg.value, arg.param.name, type, fn)
            placeholder_index += 1

        return arg.with_value(new_value)

    return [
        replace_for_argument(arg) if at_least_one_tensor(arg.value) else arg
        for arg in arguments
    ]


def get_bindings_and_arguments(
        arguments: Sequence[nat.AlignedArg],
        bindings: Sequence[Binding]
) -> List[Tuple[Binding, nat.AlignedArg]]:
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
        bindings_and_arguments: List[Tuple[Binding, nat.AlignedArg]],
        fn: nat.Function
) -> List[nat.Value]:

    def binding_to_values(pair: Tuple[Binding, nat.AlignedArg]) -> List[nat.Value]:
        binding, arg = pair
        py = nat.str_to_py(arg.value, arg.param.type) if arg.default else arg.value
        return [
            nat.py_to_value(py, c_abi.type, fn)
            for c_abi in CABIArgument.from_binding(binding)
        ]

    return list(concatMap(binding_to_values, bindings_and_arguments))


def build_function_for_native(
        f: NativeFunction,
        op_name: str,
        sample: SampleInput,
        id: str,
        out_tensors: int
) -> Callable[[], List[torch.Tensor]]:
    kernel = Kernel.from_function_and_indices(f, BACKEND_INDICES)

    arguments = nat.align_arguments(
        parameters=f.func.arguments.flat_all,
        args=[sample.input, *sample.args],
        kwargs=sample.kwargs
    )

    input_tensors = get_input_tensors(arguments)
    fn = nat.Function(id, len(input_tensors), out_tensors)

    arguments = replace_value_for_inputs(arguments, fn)
    bindings_and_arguments = get_bindings_and_arguments(
        arguments=arguments,
        bindings=kernel.sig().arguments()
    )

    result = fn.add_call("result", op_name, gen_values(bindings_and_arguments, fn))

    if kernel.return_type() == ConstPointerCType(VectorCType(BaseCType(tensorT))):
        fn.set_output_from_refs([
            fn.build_vector_index(result, fn.build_int(i))
            for i in range(out_tensors)
        ])
    else:
        fn.set_output_from_ref(result)

    fn.dump()
    jitfn = fn.into_jit()

    def run() -> List[torch.Tensor]:
        return jitfn.run(input_tensors)

    return run


def equals(lhs: torch.Tensor, rhs: torch.Tensor, dtype: torch.dtype) -> bool:
    if dtype == torch.int:
        return torch.equal(lhs, rhs)
    return bool(torch.isclose(lhs.to_dense(), rhs.to_dense(), equal_nan=True).all())


class TestOps(unittest.TestCase):
    @onlyCPU
    @ops(op_db, dtypes=[torch.float])
    def test_jit(self, device, dtype, op: OpInfo):
        for op_name, msg in SKIP_LIST:
            if op.name == op_name:
                raise unittest.SkipTest(msg)

        # Overwrite the actual dtype.
        # We only want to check whether the PyTorch operation is
        # running correctly.
        dtype = pick_dtype_for(op, device)

        samples = tuple(
            op.sample_inputs(device, dtype, requires_grad=False)
        )

        for sample in samples:
            try:
                native_function = nat.find_native_function(
                    op_name=op.name,
                    args=[sample.input, *sample.args],
                    kwargs=sample.kwargs
                )
                self.assertIsNotNone(native_function)
            except Exception:
                # If there it is not a NativeFunction, bail.
                continue

            # Satisfying the typing hints.
            assert native_function is not None

            with native_function_manager(native_function):
                op_overloaded_name = native_function_overloaded_name(native_function)

                if not nat.operation_in_registry(op_overloaded_name):
                    continue

                try:
                    raw = op(
                        sample.input,
                        *sample.args,
                        **sample.kwargs
                    )
                except Exception:
                    raise unittest.SkipTest("eager function failed")

                expected = [raw] if not isinstance(raw, (tuple, list)) else raw

                if len(expected) < 1:
                    raise unittest.SkipTest("function should return something")

                if not isinstance(expected[0], torch.Tensor):
                    raise unittest.SkipTest(f"function doesn't return tensor: {type(expected[0])}")

                function_id = f"function_{native_function.func.name.unambiguous_name()}"
                result = build_function_for_native(
                    f=native_function,
                    op_name=op_overloaded_name,
                    sample=sample,
                    id=function_id,
                    out_tensors=len(expected)
                )()

                self.assertEqual(len(expected), len(result))
                for exp, res in zip(expected, result):
                    self.assertTrue(equals(exp, res, dtype))


instantiate_device_type_tests(TestOps, globals())

if __name__ == "__main__":
    unittest.main()
