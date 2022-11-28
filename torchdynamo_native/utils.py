import os
import torchgen
import traceback

from collections import defaultdict
from dataclasses import dataclass

from typing import (
    Dict,
    List,
)

from torchgen.model import (
    NativeFunction,
)

from torchgen.gen import ParsedYaml, parse_native_yaml

from torchdynamo_native.buildhelper.codegen.kernel import Kernel


@dataclass(frozen=True)
class ExceptionGroup(Exception):
    message: str
    exceptions: List[Exception]

    def __str__(self) -> str:
        # First, print the message.
        # Then, print the stacktrace of all the inner exceptions.
        return "\n".join([
            self.message,
            ""  # Empty line
        ] + [
            "".join(traceback.format_exception(type(e), e, e.__traceback__)) for e in self.exceptions
        ])


@dataclass(frozen=True)
class OverloadInfo:
    f: NativeFunction

    @property
    def arguments(self) -> int:
        return len(self.f.func.arguments.flat_all)

    @property
    def default_arguments(self) -> int:
        return sum(arg.default is not None for arg in self.f.func.arguments.flat_all)

    @property
    def needed_arguments(self) -> int:
        return self.arguments - self.default_arguments


def native_function_key(f: NativeFunction) -> str:
    return str(f.func.name.name)


def native_function_overloaded_name(f: NativeFunction) -> str:
    return str(f.func.name)


def parse_native_functions_yaml() -> ParsedYaml:
    # Torchgen base file.
    torchgen_init = torchgen.__file__
    torchgen_dir = os.path.dirname(torchgen_init)

    # Packaged files directory.
    packaged_dir = os.path.join(torchgen_dir, "packaged", "ATen", "native")

    # Path to YAML files.
    native_functions_yaml_path = os.path.join(packaged_dir, "native_functions.yaml")
    tags_yaml_path = os.path.join(packaged_dir, "tags.yaml")

    return parse_native_yaml(native_functions_yaml_path, tags_yaml_path)


def group_native_functions_overloads(
        native_functions: List[NativeFunction]
) -> Dict[str, List[OverloadInfo]]:
    map_by_name = defaultdict(list)
    for f in native_functions:
        map_by_name[native_function_key(f)].append(OverloadInfo(f))
    return map_by_name


NATIVE_FUNCTIONS, BACKEND_INDICES = parse_native_functions_yaml()
NATIVE_FUNCTIONS_OVERLOAD_MAP = group_native_functions_overloads(NATIVE_FUNCTIONS)


def get_kernel(f: NativeFunction) -> Kernel:
    return Kernel.from_function_and_indices(f, BACKEND_INDICES)
