import torchdynamo_native as nat

import unittest
import torch

from collections import defaultdict

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
    VectorCType,
    tensorT,
)
from torchgen.context import native_function_manager
from torchgen.utils import concatMap

from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Sequence,
    Tuple,
    Union
)

from torchdynamo_native.buildhelper.codegen.kernel import ConstPointerCType
from torchdynamo_native.compiler import (
    gen_values,
    get_bindings_and_arguments
)
from torchdynamo_native.testing.utils import (
    NONDETERMINISTIC_OPERATIONS,
    EagerFailedError,
    NativeFunctionNotFoundError,
    OverloadNotRegisteredError,
    TestData,
    UnexpectedReturnValueError,
    VoidFunctionError,
    equals,
    similar
)
from torchdynamo_native.utils import native_function_overloaded_name

SKIP_OPERATIONS = {
    "tensordot": "needs pre-processing before native kernel",
    "stft": "needs pre-processing before native kernel",
    "gelu": "at::impl::variable_excluded_from_dispatch() INTERNAL ASSERT FAILED",
    "vander": "actual operation: linalg_vander",
    "where": "input is not the first argument",
}


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


def create_value_for(index: int, value: Any, type: Type, fn: nat.Function) -> nat.Value:
    if type == BaseType(BaseTy.Tensor):
        return fn.set_placeholder(index)

    elif type == OptionalType(BaseType(BaseTy.Tensor)):
        if value is None:
            return fn.build_nullopt_tensor()
        else:
            return fn.build_optional_from_ref_tensor(fn.set_placeholder(index))

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
                new_value.append(create_value_for(placeholder_index, arg.value, type.elem, fn))
                placeholder_index += int(tensor is not None)

        else:
            new_value = create_value_for(placeholder_index, arg.value, type, fn)
            placeholder_index += 1

        return arg.with_value(new_value)

    return [
        replace_for_argument(arg) if at_least_one_tensor(arg.value) else arg
        for arg in arguments
    ]


def build_function_for_native(
        f: NativeFunction,
        op_name: str,
        sample: SampleInput,
        id: str,
        out_tensors: int
) -> Callable[[], List[torch.Tensor]]:
    kernel = nat.utils.get_kernel(f)

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

    result = fn.add_call(op_name, gen_values(bindings_and_arguments, fn))

    if kernel.return_type() == ConstPointerCType(VectorCType(BaseCType(tensorT))):
        fn.set_output_from_refs([
            fn.build_vector_index(result, fn.build_int(i))
            for i in range(out_tensors)
        ])
    else:
        fn.set_output_from_ref(result)

    jitfn = fn.into_jit()

    def run() -> List[torch.Tensor]:
        return jitfn.run(input_tensors)

    return run


def clone_input(
        input: Union[torch.Tensor, Tuple[torch.Tensor], List[torch.Tensor]]
) -> Union[torch.Tensor, Tuple[torch.Tensor, ...], List[torch.Tensor]]:
    if isinstance(input, torch.Tensor):
        return input.clone()

    assert isinstance(input[0], torch.Tensor)

    if isinstance(input, list):
        return [t.clone() for t in input]
    elif isinstance(input, tuple):
        return tuple(t.clone() for t in input)

    raise ValueError("input is not a Tensor.")


class TestOps(unittest.TestCase):
    def _name(self, op_name: str, inplace: bool = False) -> str:
        name = op_name.split(".")[-1]
        return f"{name}_" if inplace else name

    def _tests_for(
            self,
            device: torch.device,
            dtype: torch.dtype,
            op: OpInfo,
            skip: Dict[str, str] = {},
            inplace: bool = False
    ) -> Iterator[Union[TestData, Exception]]:
        name = self._name(op.name, inplace)

        if name in skip:
            raise unittest.SkipTest(skip[name])

        # Overwrite the actual dtype.
        # We only want to check whether the PyTorch operation is
        # running correctly.
        dtype = nat.testing.pick_dtype_for(op, device)

        samples = tuple(
            op.sample_inputs(device, dtype, requires_grad=False)
        )

        for sample in samples:
            try:
                native_function = nat.find_native_function(
                    op_name=name,
                    args=[sample.input, *sample.args],
                    kwargs=sample.kwargs
                )

                yield TestData(op, native_function, sample, dtype)
            except Exception as e:
                yield NativeFunctionNotFoundError(e)

    def _test(
            self,
            f: NativeFunction,
            op: Callable[..., Union[torch.Tensor, List[torch.Tensor]]],
            sample: SampleInput,
            dtype: torch.dtype,
            check_equality: bool = True,
    ):
        with native_function_manager(f):
            op_overloaded_name = native_function_overloaded_name(f)

            if not nat.operation_in_registry(op_overloaded_name):
                raise OverloadNotRegisteredError(op_overloaded_name)

            try:
                input_clone = clone_input(sample.input)
                raw = op(input_clone, *sample.args, **sample.kwargs)
            except Exception as e:
                raise EagerFailedError(e)

            expected = [raw] if not isinstance(raw, (tuple, list)) else raw

            if len(expected) < 1:
                raise VoidFunctionError()

            if not isinstance(expected[0], torch.Tensor):
                raise UnexpectedReturnValueError(type(expected[0]))

            result = build_function_for_native(
                f=f,
                op_name=op_overloaded_name,
                sample=sample,
                id=f"function_{f.func.name.unambiguous_name()}",
                out_tensors=len(expected)
            )()

            self.assertEqual(len(expected), len(result))

            for exp, res in zip(expected, result):
                self.assertTrue(similar(exp, res))

                if check_equality:
                    self.assertTrue(equals(exp, res, dtype))

    def _track_errors(self, tests: Iterator[Union[TestData, Exception]], test_it: Callable) -> None:
        success_count = 0
        exceptions = defaultdict(list)

        for test in tests:
            try:
                if isinstance(test, Exception):
                    raise test

                test_it(test)
                success_count += 1
            except (
                    NativeFunctionNotFoundError,
                    OverloadNotRegisteredError,
                    EagerFailedError,
                    VoidFunctionError,
                    UnexpectedReturnValueError
            ) as e:
                exceptions[type(e)].append(e)

        if success_count == 0 and len(exceptions) > 0:
            exp_list = ", ".join([
                f"{ty.__name__}({v[0]}): {len(v)}"
                for ty, v in exceptions.items()
            ])
            raise unittest.SkipTest(str(exp_list))

    @onlyCPU
    @ops(op_db, dtypes=[torch.float])
    def test_jit(self, device: torch.device, dtype: torch.dtype, op: OpInfo):
        def test_it(test: TestData):
            self._test(
                f=test.f,
                op=test.op,
                sample=test.sample,
                dtype=test.dtype,
                check_equality=self._name(op.name, False) not in NONDETERMINISTIC_OPERATIONS
            )

        self._track_errors(
            tests=self._tests_for(device, dtype, op, SKIP_OPERATIONS),
            test_it=test_it,
        )

    @onlyCPU
    @ops(op_db, dtypes=[torch.float])
    def test_jit_inplace(self, device: torch.device, dtype: torch.dtype, op: OpInfo):
        if op.inplace_variant is None:
            raise unittest.SkipTest(f"no in-place variant for: {op.name}")

        def test_it(test: TestData):
            self._test(
                f=test.f,
                op=test.op.inplace_variant,
                sample=test.sample,
                dtype=test.dtype,
                check_equality=self._name(op.name, True) not in NONDETERMINISTIC_OPERATIONS,
            )

        self._track_errors(
            tests=self._tests_for(device, dtype, op, SKIP_OPERATIONS, inplace=True),
            test_it=test_it,
        )


instantiate_device_type_tests(TestOps, globals())

if __name__ == "__main__":
    unittest.main()
