from distutils.command.clean import clean as setup_clean
from helper.build.cmake import CMake
from setuptools.command.build_ext import build_ext as setup_build_ext
from setuptools.extension import Extension
from typing import Type

class CMakeExtension(Extension):
    def __init__(self) -> None:
        super().__init__("CMake", [])

def is_cmake_extension(ext: Extension) -> bool:
    return isinstance(ext, CMakeExtension)

class clean(setup_clean):
    def run(self) -> None:
        super().run()
        self.cmake().clean()

class build_ext(setup_build_ext):
    def run(self) -> None:
        fallback_extensions = [ext for ext in self.extensions if not is_cmake_extension(ext)]

        if len(fallback_extensions) != len(self.extensions):  # type: ignore
            try:
                self.cmake().build()
                self.cmake().install()
            finally:
                pass

        if len(fallback_extensions) > 0:
            # Fallback to the original build process.
            self.extensions = fallback_extensions
            super().run()

def wrap_with_cmake_instance(cls: Type, cmake: CMake) -> Type:
    # Create new class with injected CMake instance.
    class CMakeInstanceInjectedWrapper(cls):
        def cmake(self) -> CMake:
            return cmake
    return CMakeInstanceInjectedWrapper
