from collections import defaultdict
import sys
import os

from typing import Sequence
from setuptools import setup
from setuptools.extension import Extension

from torch.utils.cpp_extension import CppExtension
from torchgen.gen import parse_native_yaml
from torchgen.model import NativeFunction
from torchgen.utils import FileManager

from codegen import gen

CSRC_DIR = os.path.join("torchdynamo_native", "csrc")
TEMPLATES_DIR = os.path.join(CSRC_DIR, "templates")
GENERATED_DIR = os.path.join(CSRC_DIR, "generated")
OPS_DIR = os.path.join(GENERATED_DIR, "ops")

ATEN_OPS_FILE = os.path.join(CSRC_DIR, "aten_ops.cpp")

def filter_nativefunctions(native_functions: Sequence[NativeFunction]) -> Sequence[NativeFunction]:
    # Ignore internal functions: those that start with '_'.
    return list(filter(lambda f: f.root_name[0] != "_", native_functions))

def gen_aten_ops() -> None:
    native_functions, _ = parse_native_yaml(gen.get_native_functions_yaml_path(), gen.get_tags_yaml_path())
    filtered_nativefunctions = filter_nativefunctions(native_functions)

    ops_fm = FileManager(OPS_DIR, TEMPLATES_DIR, False)
    fm = FileManager(GENERATED_DIR, TEMPLATES_DIR, False)

    # Group NativeFunction by its root name.
    grouped_nativefunctions = defaultdict(lambda: [])
    for f in filtered_nativefunctions:
        grouped_nativefunctions[f.root_name].append(f)

    # Generate operator structure declarations.
    for name, fs in grouped_nativefunctions.items():
        ops_fm.write_with_template(
            filename=f"{name}.h",
            template_fn="aten_ops.h",
            env_callable=lambda: {
                "generator_file": __file__,
                "aten_ops_decl": [gen.decl(f) for f in fs],
            }
        )

    # Generate sharded operator definitions.
    fm.write_sharded(
        filename="aten_ops.cpp",
        items=grouped_nativefunctions.items(),
        key_fn=lambda p: p[0],
        env_callable=lambda p: {
            "aten_ops_defn": [gen.defn(f) for f in p[1]],
            "aten_ops_include": [f"""#include \"{os.path.join(OPS_DIR, f"{p[0]}.h")}\""""],
        },
        base_env={
            "generator_file": __file__,
        },
        num_shards=5,
        sharded_keys={
            "aten_ops_defn",
            "aten_ops_include",
        }
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
