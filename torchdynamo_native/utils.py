import os
import torchgen

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


NATIVE_FUNCTIONS, _ = parse_native_functions_yaml()
NATIVE_FUNCTIONS_OVERLOAD_MAP = group_native_functions_overloads(NATIVE_FUNCTIONS)
