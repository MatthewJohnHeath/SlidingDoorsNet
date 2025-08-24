import torch
from torch import nn

class UnreliableConnectionDummy(nn.Module):
    
    def __init__(self, probability_model: nn.Module):
        super().__init__()
        self.probability_model = probability_model

    
    def forward(self, _) -> torch.Tensor:
        raise Exception( "UnreliableConnectionDummy cannot be called on real data. It is only for code generation")
    
class UnreliableConnection(nn.Module):
    def __init__ (self, dummy:  UnreliableConnectionDummy, incoming_timelines: int):
        super().__init__()
        self.probability_model = dummy.probability_model
        self.incoming_timelines = incoming_timelines

    def forward(self, x: torch.Tensor)-> torch.Tensor:
        number_of_samples : int = x.shape[0]
        size_of_row : int= x.shape[1]
        samples_per_timeline : int = number_of_samples // self.incoming_timelines
        out : torch.Tensor = torch.zeros((number_of_samples + samples_per_timeline, size_of_row))
        out[:number_of_samples, :] = x

        without_probabilities: torch.Tensor = x[:, 1:]
        probabilities_at_connection  = self.probability_model(without_probabilities)
        out[:number_of_samples, 0] *= probabilities_at_connection
        out[number_of_samples: , 0] = 1.
        timelines = x.chunk(samples_per_timeline)
        
        for timeline in timelines:
            out[number_of_samples: , 0] -= timeline[:, 0]
            
        return out
    
