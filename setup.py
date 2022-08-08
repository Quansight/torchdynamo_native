import shutil
import sys
import sysconfig
import os
import json

from typing import List, Optional, Sequence
from setuptools import setup, Command
from setuptools.extension import Extension

from torch.utils.cpp_extension import CppExtension
from torchgen.gen import parse_native_yaml
from torchgen.model import NativeFunction, BaseType, BaseTy, ListType
from torchgen.utils import FileManager, mapMaybe

from helper.build.cmake import CMake
from helper.build.setupext import (
    build_ext,
    clean,
    CMakeExtension,
    wrap_with_cmake_instance
)
from helper.codegen import regfn, glregfn, utils

# ================================================================================
# Static Variables ===============================================================
# ================================================================================
PROJECT = "tdnat"

ROOT_DIR = os.path.realpath(os.path.dirname(sys.argv[0]))

CSRC_DIR = os.path.join(ROOT_DIR, "torchdynamo_native", "csrc")
BUILD_DIR = os.path.join(ROOT_DIR, "build")
TORCH_DIR = os.path.join(ROOT_DIR, "third-party", "pytorch", "torch")
LIB_DIR = os.path.join(ROOT_DIR, "lib")

GENERATED_DIR = os.path.join(LIB_DIR, "generated")
TEMPLATES_DIR = os.path.join(LIB_DIR, "templates")

TYPE_BLOCKLIST = [
    BaseType(BaseTy.Dimname),
    BaseType(BaseTy.Dimname),
    BaseType(BaseTy.DimVector),
    BaseType(BaseTy.QScheme),
    BaseType(BaseTy.SymInt),
    ListType(BaseType(BaseTy.SymInt), None),
]

SHARDS = 5

# ================================================================================
# Code Generation Entry-Point ====================================================
# ================================================================================

def filter_nativefunctions(native_functions: Sequence[NativeFunction]) -> Sequence[NativeFunction]:
    def predicate(f: NativeFunction) -> bool:
        return not (
            f.root_name[0] == "_"
            or (isinstance(f.func.returns, tuple) and len(f.func.returns) > 1)
            or any(r.type in TYPE_BLOCKLIST for r in f.func.returns)
            or any(a.type in TYPE_BLOCKLIST for a in f.func.arguments.flat_all)
        )
    # Ignore internal functions: those that start with '_'.
    return list(filter(predicate, native_functions))

def gen_aten_ops() -> None:
    native_functions, indices = parse_native_yaml(utils.get_native_functions_yaml_path(), utils.get_tags_yaml_path())
    filtered_nativefunctions = filter_nativefunctions(native_functions)

    fm = FileManager(GENERATED_DIR, TEMPLATES_DIR, False)

    fm.write_sharded(
        filename="register_function.cpp",
        items=filtered_nativefunctions,
        key_fn=lambda fn: fn.root_name,
        env_callable=lambda fn: {
            "ops_include": [regfn.include(fn, indices)],
            "ops_entry": [regfn.insert_entry(fn, indices)],
        },
        num_shards=SHARDS,
        base_env={
            "generator_file": __file__,
            "register_function_prefix": regfn.prefix(),
            "register_function_parameters": regfn.parameters()
        },
        sharded_keys={
            "ops_include",
            "ops_entry"
        }
    )

    fm.write(
        filename="global_register_function.cpp",
        env_callable=lambda: {
            "generator_file": __file__,
            "register_functions_decl": [glregfn.decl(i) for i in range(SHARDS)],
            "register_functions_call": [glregfn.call(i) for i in range(SHARDS)],
            "register_function_prefix": regfn.prefix(),
            "register_function_parameters": regfn.parameters(),
        }
    )

# Actually call the code generation.
gen_aten_ops()

# ================================================================================
# Build Process Setup ============================================================
# ================================================================================
def get_system_root() -> str:
    return os.path.dirname(os.path.dirname(sys.executable))

def get_extension() -> Extension:
    include_dirs = []
    include_dirs.append(os.path.join(get_system_root(), "include"))
    include_dirs.append(os.path.dirname(os.path.realpath(__file__)))
    include_dirs.append(sysconfig.get_path("include"))

    library_dirs = []
    library_dirs.append(os.path.join(get_system_root(), "lib"))

    sources = []
    sources.append(os.path.join(CSRC_DIR, "init.cpp"))

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

def generate_compile_commands(ext: Extension) -> None:
    compiler = sysconfig.get_config_var("CXX")
    include_dirs_args = [f"-I{directory}" for directory in ext.include_dirs]
    library_dirs_args = [f"-L{directory}" for directory in ext.library_dirs]
    library_args = [f"-l{lib}" for lib in ext.libraries]

    obj = []
    for s in ext.sources:
        obj.append({
            "directory": ROOT_DIR,
            "command": " ".join([
                compiler,
                *include_dirs_args,
                *library_dirs_args,
                *library_args,
                *ext.extra_compile_args
            ]),
            "file": s,
        })

    with open(os.path.join(ROOT_DIR, "compile_commands.json"), "w") as f:
        json.dump(obj, f, indent=4, sort_keys=True)

cpp_extension = get_extension()
generate_compile_commands(cpp_extension)

# ================================================================================
# Setting up CMake build directory ===============================================
# ================================================================================

if not os.path.isdir(BUILD_DIR):
    os.makedirs(BUILD_DIR, exist_ok=True)

cmake = CMake(BUILD_DIR)
cmake.run(ROOT_DIR, [
    f"--install-prefix={BUILD_DIR}/{PROJECT}",
    f"-DTORCH_DIR={TORCH_DIR}",
    f"-DENABLE_TESTS=ON",
    f"-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
])

# ================================================================================
# Setuptools Setup ===============================================================
# ================================================================================
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
        CMakeExtension(),
        cpp_extension,
    ],
    cmdclass={
        "clean": wrap_with_cmake_instance(clean, cmake),
        "build_ext": wrap_with_cmake_instance(build_ext, cmake)
    }
)
