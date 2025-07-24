import torch
from torch import nn

class Alongside(nn.Module):
    def __init__(self, operation, *modules):
        super().__init__()
        self.modules = nn.ModuleList(modules)
        if not self.modules:
            raise Exception("At least one module must be passed")
        self.operation = operation

    def forward(self, x):
        return self.operation([module(x) for module in self.modules])
    
def stack_modules(*modules):
    return Alongside(torch.stack, modules)

def cat_modules(*modules):
    return Alongside(torch.cat, modules)

def sum_modules(*modules):
    def sum(x):
        return torch.stack(x).sum(-1)
    return Alongside(sum, modules)