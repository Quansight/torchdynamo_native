from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

import subprocess

class BuildSystem(ABC):
    @abstractmethod
    def cmake_args(self) -> List[str]: ...

    @abstractmethod
    def clean_command(self) -> List[str]: ...

class Ninja(BuildSystem):
    def cmake_args(self) -> List[str]:
        return ["-G", "Ninja"]

    def clean_command(self) -> List[str]:
        return ["ninja", "clean"]

@dataclass
class CMake:
    build_dir_path: str
    system: BuildSystem = Ninja()

    def run(self, root_dir_path: str, args: List[str]) -> None:
        subprocess.check_call(["cmake", *self.system.cmake_args(), *args, root_dir_path], cwd=self.build_dir_path)

    def build(self) -> None:
        subprocess.check_call(["cmake", "--build", "."], cwd=self.build_dir_path)

    def install(self) -> None:
        subprocess.check_call(["cmake", "--install", "."], cwd=self.build_dir_path)

    def clean(self) -> None:
        subprocess.check_call(self.system.clean_command(), cwd=self.build_dir_path)

