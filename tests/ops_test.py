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
    SchemaKind,
    Type,
)

from typing import (
    Any,
    Callable,
    Dict,
    List,
    Sequence,
    Union
)

DTYPE_SET = {
    torch.float,
    torch.long,
    torch.complex64,
}

NATIVE_FUNCTIONS, _ = nat.parse_native_functions_yaml()
NATIVE_FUNCTIONS_MAP: Dict[str, NativeFunction] = {str(f.func.name): f for f in NATIVE_FUNCTIONS}

SKIP_LIST = [
    ("tensordot", "needs pre-processing before native kernel"),
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

    ("as_strided_scatter", "segfault"),
    ("searchsorted", "segfault"),
    ("istft", "segfault"),
    ("stft", "segfault"),
    ("nanquantile", "segfault"),
    ("quantile", "segfault"),
]


def pick_dtype_for(op: OpInfo, device) -> torch.dtype:
    for dtype in DTYPE_SET:
        if op.supports_dtype(dtype, device):
            return dtype
    raise ValueError("couldn't find dtype")


def count_number_of_tensors(thing: Any) -> int:
    if isinstance(thing, torch.Tensor):
        return 1
    if isinstance(thing, (list, tuple)) and not isinstance(thing, str):
        return sum(count_number_of_tensors(t) for t in thing)
    if isinstance(thing, dict):
        return sum(count_number_of_tensors(v) for v in thing.values())
    return 0


def at_least_one_tensor(value: Any, ty: Type) -> bool:
    if ty == BaseType(BaseTy.Tensor):
        return isinstance(value, torch.Tensor)
    if ty == OptionalType(BaseType(BaseTy.Tensor)):
        return value is not None and isinstance(value, torch.Tensor)

    assert isinstance(ty, ListType)
    return any(at_least_one_tensor(v, ty.elem) for v in value)


def get_inputargs(aligned_arguments: Sequence[nat.AlignedArg]) -> Dict[int, nat.AlignedArg]:
    return {
        i: arg
        for i, arg in enumerate(aligned_arguments)
        if arg.param.type.is_tensor_like() and at_least_one_tensor(arg.value, arg.param.type)
    }


def into_value(type: Type, name: str, value: Any, index: int, fn: nat.Function) -> nat.Value:
    if type == BaseType(BaseTy.Tensor):
        return fn.set_placeholder(index, name)

    elif type == OptionalType(BaseType(BaseTy.Tensor)):
        if value is None:
            return fn.build_nullopt_tensor()
        return fn.build_optional_tensor(fn.set_placeholder(index, name))

    raise ValueError(f"expected Tensor or Optional[Tensor] type: got {type}")


def alignedargs_into_values(
        aligned_arguments: Sequence[nat.AlignedArg],
        input_args: Dict[int, nat.AlignedArg],
        fn: nat.Function
) -> List[nat.Value]:
    placeholder_idx = 0
    arg_values = []

    for i, arg in enumerate(aligned_arguments):
        param = arg.param

        if i in input_args:
            if isinstance(param.type, ListType):
                list_values = []

                for i, tensor in enumerate(arg.value):
                    list_values.append(into_value(
                        type=param.type.elem,
                        name=f"{param.name}_{i}",
                        value=tensor,
                        index=placeholder_idx,
                        fn=fn
                    ))
                    placeholder_idx += int(tensor is not None)

                if param.type.elem == BaseType(BaseTy.Tensor):
                    arg_values.append(fn.build_arrayref_tensor(list_values))
                elif param.type.elem == OptionalType(BaseType(BaseTy.Tensor)):
                    arg_values.append(fn.build_optional_tensorlist(list_values))
                else:
                    raise ValueError(f"invalid input list type: {param.type}")

            else:
                arg_values.append(into_value(
                    type=param.type,
                    name=param.name,
                    value=arg.value,
                    index=placeholder_idx,
                    fn=fn
                ))
                placeholder_idx += 1
        else:
            arg = aligned_arguments[i]
            py = arg.value \
                if not arg.default \
                else nat.str_to_py(arg.value, param.type)
            arg_values.append(nat.py_to_value(py, param.type, fn))

    return arg_values


def get_input_tensors(input_args: Dict[int, nat.AlignedArg]) -> List[torch.Tensor]:
    def expand_arg(arg: nat.AlignedArg) -> List[torch.Tensor]:
        if isinstance(arg.param.type, ListType):
            return [tensor for tensor in arg.value]
        else:
            return [arg.value]

    return [
        tensor
        for i in sorted(input_args.keys())
        for tensor in expand_arg(input_args[i])
    ]


def build_function_for_native(
        f: NativeFunction,
        op_name: str,
        sample: SampleInput,
        id: str,
        out_tensors: int
) -> Callable[[], List[torch.Tensor]]:
    aligned_args = nat.align_and_flat_arguments(
        parameters=f.func.arguments.flat_all,
        args=[sample.input, *sample.args],
        kwargs=sample.kwargs
    )
    input_args = get_inputargs(aligned_args)
    input_tensors = get_input_tensors(input_args)

    fn = nat.Function(id, len(input_tensors), out_tensors)
    value_args = alignedargs_into_values(aligned_args, input_args, fn)

    result = fn.add_call("result", op_name, value_args)
    fn.set_output(result)
    fn.finalize()

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
                op_overloaded_name = nat.find_operator_name(
                    op_name=op.name,
                    args=[sample.input, *sample.args],
                    kwargs=sample.kwargs
                )
                self.assertIsNotNone(op_overloaded_name)
            except Exception:
                # If there it is not a NativeFunction, bail.
                continue

            # Satisfying the typing hints.
            assert op_overloaded_name is not None

            nativef = NATIVE_FUNCTIONS_MAP[op_overloaded_name]

            if nativef.func.kind() == SchemaKind.inplace:
                raise unittest.SkipTest("inplace functions not supported")

            if not nat.operation_in_registry(op_overloaded_name):
                continue

            try:
                expected: Union[torch.Tensor, List[torch.Tensor]] = op(
                    sample.input,
                    *sample.args,
                    **sample.kwargs
                )
            except Exception:
                raise unittest.SkipTest("eager function failed")

            # Assuming every tensor is given as a parameter.
            if isinstance(expected, torch.Tensor):
                out_tensors = 1
            elif isinstance(expected, (list, tuple)):
                raise unittest.SkipTest("function with multiple return values not supported")
                # out_tensors = len(expected)
            else:
                raise unittest.SkipTest(f"unexpected output type: {type(expected)}")

            function_id = f"function_{nativef.func.name.unambiguous_name()}"
            result = build_function_for_native(
                f=nativef,
                op_name=op_overloaded_name,
                sample=sample,
                id=function_id,
                out_tensors=out_tensors
            )()

            if not isinstance(expected, list):
                expected = [expected]

            self.assertEqual(len(expected), len(result))
            for exp, res in zip(expected, result):
                if not equals(exp, res, dtype):
                    print(f"Function: {nativef.func}")
                    print(f"Input: {sample.input}")
                    print(f"Args: {sample.args}")
                    print(f"KWargs: {sample.kwargs}")
                    print(f"Expected: {exp}")
                    print(f"Result: {res}")
                self.assertTrue(equals(exp, res, dtype))


instantiate_device_type_tests(TestOps, globals())

if __name__ == "__main__":
    unittest.main()
