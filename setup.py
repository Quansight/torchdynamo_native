from setuptools import setup
from setuptools.extension import Extension

from torch.utils.cpp_extension import CppExtension

import sys
import os

def get_system_root() -> str:
    return os.path.dirname(os.path.dirname(sys.executable))

def get_extension() -> Extension:
    include_dirs = []
    include_dirs.append(f"{get_system_root()}/include")

    library_dirs = []
    library_dirs.append(f"{get_system_root()}/lib")

    sources = []
    sources.append("torchdynamo_native/csrc/init.cpp")

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
        ]
    )

setup(
    name="torchdynamo_native",
    version="0.0.1",
    packages=[
        "torchdynamo_native"
    ],
    install_requires=[
        "torchdynamo",
        "pybind11"
    ],
    ext_modules=[
        get_extension()
    ]
)
