'''
 # @ author: bcynuaa <bcynuaa@163.com>
 # @ date: 2024-09-26 21:56:18
 # @ license: MIT
 # @ description:
 '''

import torch
import torch.nn as nn

class BasicDNN(nn.Module):
    
    def __init__(self, input_dimension: int, output_dimension: int) -> None:
        super(BasicDNN, self).__init__()
        self.input_dimension: int = input_dimension
        self.output_dimension: int = output_dimension
        self.linear_layer: nn.Linear = nn.Linear(input_dimension, output_dimension)
        self.non_linear_layer: nn.ReLU = nn.ReLU()
        pass
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear_layer(x)
        x = self.non_linear_layer(x)
        return x
        pass
    
    pass