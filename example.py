from typing import List

import torch
import torchdynamo
import torchdynamo_native._C as N
import random

N.dump_operations()

def toy_example(a, b, c):
    x = a / (torch.abs(a) + 1 / c)
    if b.sum() < 0:
        b = b * -1
    return x * b

def my_compiler(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    print("my_compiler() called with FX graph:")
    gm.graph.print_tabular()
    return gm.forward  # return a python callable

with torchdynamo.optimize(my_compiler):
    for _ in range(5):
        c = random.random()
        toy_example(torch.randn(10), torch.randn(10), c)

