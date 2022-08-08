import shutil
import sys
import os

from typing import List, Optional, Sequence
from setuptools import setup, Command
from setuptools.extension import Extension
from setuptools.command.build_ext import build_ext

from torch.utils.cpp_extension import CppExtension
from torchgen.gen import parse_native_yaml
from torchgen.model import NativeFunction, BaseType, BaseTy, ListType
from torchgen.utils import FileManager, mapMaybe

from helper.build.cmake import CMake
from helper.build.setupext import build_ext, clean, CMakeExtension, wrap_with_cmake_instance
from helper.codegen import gen

# ================================================================================
# Static Variables ===============================================================
# ================================================================================
PROJECT = "tdnat"

BUILD_DIR = os.path.join(SCRIPT_DIR, "build")
CSRC_DIR = os.path.join(SCRIPT_DIR, "torchdynamo_native", "csrc")
GENERATED_DIR = os.path.join(CSRC_DIR, "generated")
OPS_DIR = os.path.join(GENERATED_DIR, "ops")
PYTORCH_LIBS_DIR = os.path.join(SCRIPT_DIR, "third-party", "pytorch", "torch", "lib")
SCRIPT_DIR = os.path.realpath(os.path.dirname(sys.argv[0]))
TEMPLATES_DIR = os.path.join(CSRC_DIR, "templates")

ATEN_OPS_FILE = os.path.join(CSRC_DIR, "aten_ops.cpp")

TYPE_BLOCKLIST = [
    BaseType(BaseTy.Dimname),
    BaseType(BaseTy.Dimname),
    BaseType(BaseTy.DimVector),
    BaseType(BaseTy.QScheme),
    BaseType(BaseTy.SymInt),
    ListType(BaseType(BaseTy.SymInt), None),
]

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

# ================================================================================
# Setting up CMake build directory ===============================================
# ================================================================================

if not os.path.isdir(BUILD_DIR):
    os.makedirs(BUILD_DIR, exist_ok=True)

cmake = CMake(BUILD_DIR)
cmake.run(SCRIPT_DIR, [
    f"--install-prefix={BUILD_DIR}/{PROJECT}",
    f"-DPYTORCH_LIBS_PATH={PYTORCH_LIBS_DIR}"
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
        get_extension()
    ],
    cmdclass={
        "clean": wrap_with_cmake_instance(clean, cmake),
        "build_ext": wrap_with_cmake_instance(build_ext, cmake)
    }
)
