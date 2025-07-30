from alongside import Alongside, OnRange, catted
from unreliable_connection import UnreliableConnection, UnreliableConnectionDummy
import torch 
from torch import Tensor, nn
from torch.nn import Module, Sequential, Identity
from typing import Tuple
import math


class ExpandedAlongside(Module):
        def __init__(self, inner : Alongside,context_header_size = 0) -> None:
            super().__init__()
            unrolled_modules = [unroll_unreliable_connections(module, context_header_size=context_header_size)for  module in inner.module_list]
            self.module_list = nn.ModuleList([mod for mod, _ in unrolled_modules])
            self.timeline_splits = [n for _, n in unrolled_modules]
            self.total_splits = math.prod(self.timeline_splits)
        def forward(self, x : Tensor) -> Tensor:
            number_of_input_samples = x.size()[0]
            number_of_output_samples = number_of_input_samples * self.total_splits
            length_of_sample = x.size()[1]
            output = torch.zeros((number_of_output_samples, length_of_sample), dtype = x.dtype)
            branch_outputs:list[Tensor] = [module(x) for module in self.module_list]
            for timeline_number in range(self.total_splits):
                 


def unroll_unreliable_connections(dummy_model : Module, incoming_timelines = 1, context_header_size = 0) -> Tuple[Module, int]:
    
    if isinstance(dummy_model, UnreliableConnectionDummy):
        return UnreliableConnection(dummy_model, incoming_timelines), incoming_timelines + 1
    
    if isinstance(dummy_model, Sequential):
        modules = []
        timelines_out = incoming_timelines
        for module in dummy_model:
            unrolled, timelines_out = unroll_unreliable_connections(module, timelines_out, context_header_size)
            modules.append(unrolled)
        return Sequential(*modules), timelines_out

    if isinstance(dummy_model, Alongside):
        timelines_out = 1