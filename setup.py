import sys
import sysconfig
import os
import json
import warnings

from typing import Sequence
from setuptools import setup
from setuptools.extension import Extension

from torch.utils.cpp_extension import CppExtension
from torchgen.gen import parse_native_yaml
from torchgen.model import NativeFunction, BaseType, BaseTy, ListType, OptionalType, Type
from torchgen.utils import FileManager

from torchdynamo_native.buildhelper.build.cmake import CMake
from torchdynamo_native.buildhelper.build.setupext import (
    build_ext,
    clean,
    CMakeExtension,
    wrap_with_cmake_instance
)
from torchdynamo_native.buildhelper.codegen import regfn, glregfn, utils

# ================================================================================
# Static Variables ===============================================================
# ================================================================================
PROJECT = "tdnat"

ROOT_DIR = os.path.realpath(os.path.dirname(sys.argv[0]))

CSRC_DIR = os.path.join(ROOT_DIR, "torchdynamo_native", "csrc")
BUILD_DIR = os.path.join(ROOT_DIR, "build")
TORCH_DIR = os.path.join(ROOT_DIR, "third-party", "pytorch", "torch")
LIB_DIR = os.path.join(ROOT_DIR, "lib")
INCLUDE_DIR = os.path.join(ROOT_DIR, "include")

GENERATED_LIB_DIR = os.path.join(LIB_DIR, "generated")
GENERATED_INC_DIR = os.path.join(INCLUDE_DIR, "tdnat", "generated")
TEMPLATES_DIR = os.path.join(LIB_DIR, "templates")
LINK_DIR = os.path.join(BUILD_DIR, "tdnat", "lib")

TYPE_BLOCKLIST = [
    BaseType(BaseTy.Dimname),
    BaseType(BaseTy.QScheme),
    BaseType(BaseTy.Storage),
    BaseType(BaseTy.Stream),
]

SHARDS = 5

# ================================================================================
# Code Generation Entry-Point ====================================================
# ================================================================================


def contains_blocklist_type(t: Type) -> bool:
    if isinstance(t, BaseType):
        return t in TYPE_BLOCKLIST
    if isinstance(t, OptionalType) or isinstance(t, ListType):
        return contains_blocklist_type(t.elem)
    assert False, f"unknown type: {t}"


def filter_nativefunctions(native_functions: Sequence[NativeFunction]) -> Sequence[NativeFunction]:
    def predicate(f: NativeFunction) -> bool:
        return not (
            f.root_name[0] == "_"
            or (isinstance(f.func.returns, tuple) and len(f.func.returns) > 1)
            or any(contains_blocklist_type(r.type) for r in f.func.returns)
            or any(contains_blocklist_type(a.type) for a in f.func.arguments.flat_all)
        )
    # Ignore internal functions: those that start with '_'.
    return list(filter(predicate, native_functions))


def gen_aten_ops() -> None:
    native_functions, indices = parse_native_yaml(
        utils.get_native_functions_yaml_path(),
        utils.get_tags_yaml_path()
    )
    filtered_nativefunctions = filter_nativefunctions(native_functions)

    fm_lib = FileManager(GENERATED_LIB_DIR, TEMPLATES_DIR, False)
    fm_inc = FileManager(GENERATED_INC_DIR, TEMPLATES_DIR, False)

    def native_function_key(f: NativeFunction) -> str:
        return f.root_name

    fm_inc.write_sharded(
        filename="c_abi_wrappers.h",
        items=filtered_nativefunctions,
        key_fn=native_function_key,
        env_callable=lambda fn: {
            "ops_include": [regfn.include(fn, indices)],
            "c_abi_wrapper_functions": [regfn.c_abi(fn, indices)],
        },
        num_shards=SHARDS,
        base_env={
            "generator_file": __file__,
        },
        sharded_keys={
            "ops_include",
            "c_abi_wrapper_functions",
        }
    )

    fm_lib.write_sharded(
        filename="register_function.cpp",
        items=filtered_nativefunctions,
        key_fn=lambda fn: fn.root_name,
        env_callable=lambda fn: {
            "ops_entry": [regfn.insert_entry(fn, indices)],
        },
        num_shards=SHARDS,
        base_env={
            "generator_file": __file__,
            "register_function_prefix": regfn.prefix(),
            "register_function_parameters": regfn.parameters()
        },
        sharded_keys={
            "ops_entry"
        }
    )

    fm_lib.write(
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

    # torchdynamo_native include directory
    include_dirs.append(INCLUDE_DIR)
    # Python include directory
    include_dirs.append(sysconfig.get_path("include"))
    # System include directory (one level below Python's)
    include_dirs.append(os.path.dirname(sysconfig.get_path("include")))

    library_dirs = []

    # System library directory (sibling directory of /bin/python -- executable)
    library_dirs.append(os.path.join(get_system_root(), "lib"))
    # Compiled torchdynamo_native install path.
    library_dirs.append(LINK_DIR)

    sources = []
    sources.append(os.path.join(CSRC_DIR, "init.cpp"))

    return CppExtension(
        name="torchdynamo_native._C",
        sources=sources,
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=[
            "tdnat"
        ],
        extra_compile_args=[
            "-std=c++14",
            "-DSTRIP_ERROR_MESSAGES=1",
            "-fvisibility=hidden",
        ],
        extra_link_args=[
            f"-Wl,-rpath,{LINK_DIR}"
        ]
    )


def generate_compile_commands(ext: Extension) -> None:
    compiler = sysconfig.get_config_var("CXX")
    include_dirs_args = [f"-I{directory}" for directory in ext.include_dirs]
    library_dirs_args = [f"-L{directory}" for directory in ext.library_dirs]
    library_args = [f"-l{lib}" for lib in ext.libraries]

    if not isinstance(compiler, str):
        warnings.warn("set your CXX environment variable for generating compile_commands.json")
        return

    obj = []
    for s in ext.sources:
        obj.append({
            "directory": ROOT_DIR,
            "command": " ".join([
                compiler,
                *include_dirs_args,
                *library_dirs_args,
                *library_args,
                *ext.extra_compile_args,
                s,
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
    "-DENABLE_TESTS=ON",
    "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
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
        "pybind11",
        "expecttest",
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
