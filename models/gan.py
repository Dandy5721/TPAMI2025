import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List


@dataclass
class GeneratorConfig:
    input_dim: int
    hidden_dims: List[int]
    output_dim: int
    dropout_rate: float = 0.2

class Generator(nn.Module):
    def __init__(self, config: GeneratorConfig):
        """
        Generator network for GAN.
        
        Args:
            input_dim (int): Dimension of the input data
            hidden_dims (list of int): List of hidden layer dimensions
            output_dim (int): Dimension of the output (target data)
            dropout_rate (float): Dropout probability for regularization
        """
        super(Generator, self).__init__()
        
        # Create list to hold all layers
        layers = []
        
        # Input layer
        layers.append(nn.Linear(config.input_dim, config.hidden_dims[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(config.dropout_rate))
        
        # Hidden layers
        for i in range(len(config.hidden_dims) - 1):
            layers.append(nn.Linear(config.hidden_dims[i], config.hidden_dims[i+1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(config.dropout_rate))
        
        # Output layer
        layers.append(nn.Linear(config.hidden_dims[-1], config.output_dim))
        # layers.append(nn.Tanh())  # Tanh activation to bound outputs between -1 and 1
        
        # Combine all layers into a sequential model
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        """
        Forward pass through the generator.
        
        Args:
            x (torch.Tensor): Input data of shape (batch_size, input_dim)
            
        Returns:
            torch.Tensor: Generated target data of shape (batch_size, output_dim)
        """
        return self.network(x)


@dataclass
class DiscriminatorConfig:
    input_dim: int
    hidden_dims: List[int]
    dropout_rate: float = 0.2

class Discriminator(nn.Module):
    def __init__(self, config: DiscriminatorConfig):
        """
        Discriminator network for GAN.
        
        Args:
            input_dim (int): Dimension of input data (target data)
            hidden_dims (list of int): List of hidden layer dimensions
            dropout_rate (float): Dropout probability for regularization
        """
        super(Discriminator, self).__init__()
        
        # Create list to hold all layers
        layers = []
        
        # Input layer
        layers.append(nn.Linear(config.input_dim, config.hidden_dims[0]))
        layers.append(nn.LeakyReLU(0.2))
        layers.append(nn.Dropout(config.dropout_rate))
        
        # Hidden layers
        for i in range(len(config.hidden_dims) - 1):
            layers.append(nn.Linear(config.hidden_dims[i], config.hidden_dims[i+1]))
            layers.append(nn.LeakyReLU(0.2))
            layers.append(nn.Dropout(config.dropout_rate))
        
        # Output layer - single neuron with sigmoid for binary classification
        layers.append(nn.Linear(config.hidden_dims[-1], 1))
        layers.append(nn.Sigmoid())
        
        # Combine all layers into a sequential model
        self.network = nn.Sequential(*layers)
        
    def forward(self, x): 
        """
        Forward pass through the discriminator.
        
        Args:
            x (torch.Tensor): Input data of shape (batch_size, input_dim)
            
        Returns:
            torch.Tensor: Probability that input is real (batch_size, 1)
        """
        return self.network(x)