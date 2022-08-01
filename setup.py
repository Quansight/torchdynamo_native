from collections import defaultdict
import sys
import os

from typing import List, Optional, Sequence
from setuptools import setup
from setuptools.extension import Extension

from torch.utils.cpp_extension import CppExtension
from torchgen.gen import parse_native_yaml
from torchgen.model import NativeFunction
from torchgen.utils import FileManager, mapMaybe

from codegen import gen

CSRC_DIR = os.path.join("torchdynamo_native", "csrc")
TEMPLATES_DIR = os.path.join(CSRC_DIR, "templates")
GENERATED_DIR = os.path.join(CSRC_DIR, "generated")

ATEN_OPS_FILE = os.path.join(CSRC_DIR, "aten_ops.cpp")

def filter_nativefunctions(native_functions: Sequence[NativeFunction]) -> Sequence[NativeFunction]:
    def predicate(f: NativeFunction) -> bool:
        return not (
            f.root_name[0] == "_"
            or isinstance(f.func.returns, tuple)
        )
    # Ignore internal functions: those that start with '_'.
    return list(filter(predicate, native_functions))

def gen_aten_ops() -> None:
    native_functions, indices = parse_native_yaml(gen.get_native_functions_yaml_path(), gen.get_tags_yaml_path())
    filtered_nativefunctions = filter_nativefunctions(native_functions)

    fm = FileManager(GENERATED_DIR, TEMPLATES_DIR, False)

    def include(f: NativeFunction) -> Optional[str]:
        return gen.include(f, indices)

    def add_commas(lines: List[str]) -> List[str]:
        for i in range(len(lines) - 1):
            lines[i] += ","
        return lines

    fm.write(
        filename="aten_ops.cpp",
        env_callable=lambda: {
            "generator_file": __file__,
            "aten_ops_entry": add_commas([gen.entry(f, indices) for f in filtered_nativefunctions]),
            "aten_ops_include": sorted(list(set(
                mapMaybe(include, filtered_nativefunctions)))),
        },
    )

def get_system_root() -> str:
    return os.path.dirname(os.path.dirname(sys.executable))

def get_extension() -> Extension:
    include_dirs = []
    include_dirs.append(os.path.join(get_system_root(), "include"))
    include_dirs.append(os.path.dirname(os.path.realpath(__file__)))

    library_dirs = []
    library_dirs.append(os.path.join(get_system_root(), "lib"))

    sources = []
    sources.append(os.path.join(CSRC_DIR, "init.cpp"))
    sources.append(os.path.join(CSRC_DIR, "ops.cpp"))

    for f in os.listdir(GENERATED_DIR):
        path = os.path.join(GENERATED_DIR, f)
        if not f.endswith("Everything.cpp") and not os.path.isdir(path):
            sources.append(path)

    return CppExtension(
        name="torchdynamo_native._C",
        sources=sources,
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=[
            "LLVM"
        ],
        extra_compile_args=[
            "-std=c++14",
            "-DSTRIP_ERROR_MESSAGES=1",
            "-fvisibility=hidden"
        ]
    )

# First, generate 'aten_ops.cpp' file.
gen_aten_ops()
# Then, install the package.
setup(
    name="torchdynamo_native",
    version="0.0.1",
    packages=[
        "torchdynamo_native"
    ],
    install_requires=[
        "torchdynamo",
        "pybind11",
    ],
    ext_modules=[
        get_extension()
    ]
)
