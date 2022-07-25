from typing import List

import torch
import torchdynamo
import torchdynamo_native._C as N
import random

N.dump_operations()

def foo(a, b):
    return torch.mul(a, b)

def toy_example(a, b):
    y = torch.cat(a).mean(dim=-1)
    x = y / (torch.abs(y) + 1)
    x.add_(5)
    if b.sum() < 0:
        b = b * -1
    return x * b

def maybe_print_node(node):
    if isinstance(node, torch.fx.Node):
        print_node(node)

def print_args(args):
    for i, a in enumerate(args):
        print(f"> {i}: {a} ({type(a)} -- {id(a)})")
        maybe_print_node(a)

def print_kwargs(kwargs):
    for i, (k, v) in enumerate(kwargs.items()):
        print(f"> {i}: {k} -> {v} ({type(v)})")
        maybe_print_node(v)

def print_node(node: torch.fx.Node):
    print(f"id: {id(node)}; name: {node.name}; op: {node.op}; target: {node.target}")
    print_args(node.args)
    print_kwargs(node.kwargs)

def my_compiler(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    print("my_compiler() called with FX graph:")

    print(gm.graph)
    for node in gm.graph.nodes:
        print_node(node)

    return gm.forward  # return a python callable

with torchdynamo.optimize(my_compiler):
    for _ in range(5):
        toy_example([torch.randn(10) for _ in range(5)], torch.randn(10))

