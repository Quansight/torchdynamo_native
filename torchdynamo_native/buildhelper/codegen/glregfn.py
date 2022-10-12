from torchdynamo_native.buildhelper.codegen import regfn


def decl(shard: int) -> str:
    return f"void {regfn.prefix()}_{shard}({regfn.parameters()});"


def call(shard: int) -> str:
    return f"{regfn.prefix()}_{shard}({regfn.parameter_name()});"
