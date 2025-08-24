import torch
from torch import nn, Tensor

class SoftLength(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x:Tensor)->Tensor:
        number_of_samples = x.shape[0]
        if number_of_samples == 0:
            return x
        
        tail_max = torch.zeros_like(x[:, 0])
        soft_length = torch.zeros_like(x[:, 0])
        
        for column_number in reversed(range(number_of_samples)):
            column = x[: column_number]
            sigmoids = torch.sigmoid(column)
            tail_max = torch.max(sigmoids, tail_max)
            soft_length += tail_max
        
        return soft_length

class SimpleProbabilityModel(nn.Module):
    def __init__(self, one_value_mean, one_value_sd, available_time) -> None:
        super().__init__()
        self.one_value_mean = one_value_mean
        self.one_value_variance = one_value_sd * one_value_sd
        self.available_time = available_time
        self.soft_length = SoftLength()

    def forward(self, x : Tensor) -> Tensor:
        message_length = self.soft_length(x)
        mean = self.one_value_mean * message_length
        variance = self.one_value_variance * message_length
        standard_distribution = torch.sqrt(variance)

        distribution = torch.distributions.Normal(mean, standard_distribution)
        
        return distribution.cdf(self.available_time)
