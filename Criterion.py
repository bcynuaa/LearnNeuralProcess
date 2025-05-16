'''
 # @ author: bcynuaa <bcynuaa@163.com>
 # @ date: 2024-10-07 15:02:51
 # @ license: MIT
 # @ description:
 '''

import torch
import torch.nn as nn

class Criterion(nn.Module):
    
    def __init__(self) -> None:
        super(Criterion, self).__init__()
        pass
    
    def forward(self, mu: torch.Tensor, sigma: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        mu: torch.Tensor, shape: (batch_size, output_dimension)
        sigma: torch.Tensor, shape: (batch_size, output_dimension)
        target: torch.Tensor, shape: (batch_size, output_dimension)
        """
        loss: float = 0.0
        batch_size: int = mu.shape[0]
        for batch in range(batch_size):
            distribution = torch.distributions.multivariate_normal.MultivariateNormal(
                loc=mu[batch],
                covariance_matrix=torch.diag(sigma[batch])
            )
            log_probability = distribution.log_prob(target[batch])
            loss -= 1.0 * torch.mean(log_probability) # make the distribution gains from target as least as possible (make them the same)
            pass
        loss /= batch_size
        return loss
        pass
    
    pass