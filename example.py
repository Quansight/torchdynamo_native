from typing import List

import torch
import torch.fx
import torch._dynamo as dynamo
import torchdynamo_native as nat
import random
import logging

logging.basicConfig(level=logging.DEBUG)


def my_compiler(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    print("my_compiler() called with FX graph:")
    gm.graph.print_tabular()
    return gm.forward  # return a python callable


@dynamo.optimize(nat.llvmjit)  # type: ignore
def toy_example(a, b, c):
    x = a / (torch.abs(a) + 1 / c)
    if b.sum() < 0:
        b = b * -1
    return x * b


for _ in range(5):
    c = random.random()
    toy_example(torch.randn(10), torch.randn(10), c)
