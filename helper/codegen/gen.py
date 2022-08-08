from dataclasses import dataclass
from typing import List

from torchgen.model import Argument, Arguments, Type, BaseTy, BaseType, ListType, NativeFunction, OptionalType
from torchgen.context import with_native_function
from torchgen.api import cpp

from helper.codegen.utils import get_native_functions_yaml_path, get_tags_yaml_path

def gen_type_list(ty: Type) -> List[str]:
    if isinstance(ty, ListType):
        return ["ArgType::List"] + gen_type_list(ty.elem)
    elif isinstance(ty, OptionalType):
        return ["ArgType::Optional"] + gen_type_list(ty.elem)
    else:
        return [f"ArgType::Base_{ty}"]

@dataclass(frozen=True)
class Arg:
    a: Argument
    position: int
    is_kwarg: bool

    @staticmethod
    def from_arguments(args: Arguments) -> List["Arg"]:
        arg_list = []
        for i, a in enumerate(args.flat_positional):
            arg_list.append(Arg(a, i, is_kwarg=False))
        for a in args.flat_kwarg_only:
            arg_list.append(Arg(a, -1, is_kwarg=True))
        for a in args.out:
            arg_list.append(Arg(a, -1, is_kwarg=True))
        return arg_list

    def gen(self) -> str:
        if self.a.default is not None:
            default_expr = cpp.default_expr(self.a.default, self.a.type)
            # 'contiguous_format' is translated to 'MemoryFormat::Contiguous'.
            # We need to insert 'at::' ourselves.
            if default_expr.startswith("MemoryFormat"):
                default_expr = f"at::{default_expr}"
            # Lists with more than 1 element already comes wrapped with '{}'.
            if not default_expr.startswith("{"):
                default_expr = f"{{{default_expr}}}"

            if self.a.type == BaseType(BaseTy.str):
                # Special case for strings.
                default_type = "std::string"
            else:
                # Since we are storing it, we need the owning types.
                default_type = cpp.argumenttype_type(
                    self.a.type, mutable=False, binds=self.a.name, remove_non_owning_ref_types=True)
                default_type = default_type.cpp_type(strip_ref=True)

            default = f"{default_type} {default_expr}"
        else:
            default = "c10::nullopt"

        position = f", {self.position}" if not self.is_kwarg else ""
        types = ", ".join(gen_type_list(self.a.type))

        return f"""{{ "{self.a.name}", {{{types}}}{position}, {default} }}"""


def struct_name(f: NativeFunction) -> str:
    return f"ATenOp_{f.func.name.unambiguous_name()}"

@with_native_function
def decl(f: NativeFunction) -> str:
    return f"""
struct {struct_name(f)} : public ATenOp {{
  static std::vector<ATenOpArg> arguments_;
  std::vector<ATenOpArg>& arguments() {{ return arguments_; }}
}};
"""

@with_native_function
def defn(f: NativeFunction) -> str:
    args = [a.gen() for a in Arg.from_arguments(f.func.arguments)]
    args_prefix_space = ",\n".join(map(lambda l: f"  {l}", args))

    return f"""
std::vector<ATenOpArg> {struct_name(f)}::arguments_ {{
{args_prefix_space}
}};
"""
