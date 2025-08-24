import torch
from torch import nn, Tensor
from typing import Callable, Tuple, List

class Alongside(nn.Module):
    def __init__(self, operation : Callable, *modules : nn.Module) -> None:
        super().__init__()
        self.module_list = nn.ModuleList(modules)
        if not self.module_list:
            raise Exception("At least one module must be passed")
        self.operation = operation

    def forward(self, x:Tensor) -> Tensor:
        return self.operation([module(x) for module in self.module_list])
    
def stack_modules(*modules : nn.Module) -> Alongside:
    return Alongside(torch.stack, *modules)

def cat_modules(*modules : nn.Module) -> Alongside:
    return Alongside(torch.cat, *modules)

def sum_modules(*modules : nn.Module) -> Alongside:
    def sum(x):
        return torch.stack(x).sum(-1)
    return Alongside(sum, *modules)

class OnRange(nn.Module):
    def __init__(self, layer: nn.Module, start : int = 0, end : int|None = None) -> None:
        super().__init__()
        self.layer = layer
        self.start = start
        self.end = end

    def forward(self, x: Tensor) -> Tensor:
        slice = x[:, self.start:self.end] if self.end is not None else x[:, self.start:]
        return self.layer(slice)

def catted(*modules_and_lengths : Tuple[nn.Module, int], final: nn.Module|None = None) -> Alongside:
    starts:List[int] = []
    sum : int = 0
    for (_, length) in modules_and_lengths:
        starts.append(sum)
        sum += length
        
    ends = starts[1:]
    starts.append(sum)
    on_ranges = [OnRange(layer, start, end) for (layer, _), start, end in zip(modules_and_lengths, starts, ends)]
    if final is not None:
        on_ranges.append(OnRange(final, sum))
    return cat_modules(*on_ranges)
     