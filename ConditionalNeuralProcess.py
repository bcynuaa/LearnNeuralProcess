'''
 # @ author: bcynuaa <bcynuaa@163.com>
 # @ date: 2024-09-26 21:51:06
 # @ license: MIT
 # @ description:
 '''
 
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "."))

from BasicDNN import BasicDNN

import torch
import torch.nn as nn

class ConditionalNeuralProcess(nn.Module):
    
    def __init__(
        self,
        input_dimension: int,
        output_dimension: int,
        hidden_dimension: int,
    ) -> None:
        super(ConditionalNeuralProcess, self).__init__()
        self.input_dimension: int = input_dimension
        self.output_dimension: int = output_dimension
        self.hidden_dimension: int = hidden_dimension
        self.defineEncoder()
        self.defineDecoder()
        self.softplus: nn.Softplus = nn.Softplus()
        pass
    
    def getInputDimension(self) -> int:
        return self.input_dimension
        pass
    
    def getOutputDimension(self) -> int:
        return self.output_dimension
        pass
    
    def getHiddenDimension(self) -> int:
        return self.hidden_dimension
        pass
    
    def getEncoderInputDimension(self) -> int:
        return self.getInputDimension() + self.getOutputDimension()
        pass
    
    def defineEncoder(self) -> None:
        self.encoder: nn.Sequential = nn.Sequential(
            BasicDNN(self.getEncoderInputDimension(), 32),
            BasicDNN(32, 64),
            BasicDNN(64, 128),
            nn.Linear(128, self.getHiddenDimension())
        )
        pass
    
    def getDecoderInputDimension(self) -> int:
        return self.getHiddenDimension() + self.getInputDimension()
        pass
    
    def defineDecoder(self) -> None:
        self.decoder: nn.Sequential = nn.Sequential(
            BasicDNN(self.getDecoderInputDimension(), 128),
            BasicDNN(128, 32),
            nn.Linear(32, 2)
        )
        pass
    
    def forward(self, input_context: torch.Tensor, output_context: torch.Tensor, input_target: torch.Tensor) -> torch.Tensor:
        """
        input_context: torch.Tensor, shape: (batch_size, num_context, input_dimension)
        output_context: torch.Tensor, shape: (batch_size, num_context, output_dimension)
        input_target: torch.Tensor, shape: (batch_size, num_target, input_dimension)
        """
        # * encoder
        encoder_input: torch.Tensor = torch.cat([input_context, output_context], dim=-1) # dimension: (batch_size, num_context, input_dimension + output_dimension)
        encoder_batch_size, encoder_num_context, encoder_input_dimension = encoder_input.shape
        encoder_input = encoder_input.view(encoder_batch_size * encoder_num_context, encoder_input_dimension) # dimension: (batch_size * num_context, input_dimension + output_dimension)
        encoder_output: torch.Tensor = self.encoder(encoder_input) # dimension: (batch_size * num_context, hidden_dimension)
        encoder_output = encoder_output.view(encoder_batch_size, encoder_num_context, self.getHiddenDimension()) # dimension: (batch_size, num_context, hidden_dimension)
        hidden_representation: torch.Tensor = encoder_output.mean(dim=1) # dimension: (batch_size, hidden_dimension)
        # * decoder
        target_batch_size, target_num_target, target_input_dimension = input_target.shape
        hidden_representation = hidden_representation.unsqueeze(dim=1).repeat((1, target_num_target, 1)) # dimension: (batch_size, num_target, hidden_dimension)
        decoder_input: torch.Tensor = torch.cat([hidden_representation, input_target], dim=-1) # dimension: (batch_size, num_target, hidden_dimension + input_dimension)
        decoder_input = decoder_input.view(target_batch_size * target_num_target, self.getDecoderInputDimension()) # dimension: (batch_size * num_target, hidden_dimension + input_dimension)
        decoder_output: torch.Tensor = self.decoder(decoder_input) # dimension: (batch_size * num_target, 2)
        decoder_output = decoder_output.view(target_batch_size, target_num_target, 2) # dimension: (batch_size, num_target, 2)
        mu: torch.Tensor = decoder_output[:, :, 0] # dimension: (batch_size, num_target)
        log_sigma: torch.Tensor = decoder_output[:, :, 1] # dimension: (batch_size, num_target)
        sigma: torch.Tensor = 0.1 + 0.9 * self.softplus(log_sigma) # dimension: (batch_size, num_target)
        return mu, sigma
        pass
    
    pass