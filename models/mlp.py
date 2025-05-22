from dataclasses import dataclass
from typing import List, Optional, Union, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from interfaces import ModelProtocol

@dataclass
class MLPConfig:
    """Configuration for MLP model."""
    input_dim: int
    hidden_dims: List[int]
    output_dim: int
    activation: str = "relu"
    dropout: float = 0.2
    batch_norm: bool = True

class MLP(nn.Module):
    """Multi-layer Perceptron implementation."""
    
    def __init__(self, config: MLPConfig):
        super().__init__()
        self.config = config
        
        # Build layers
        layers = []
        prev_dim = config.input_dim
        
        # Add hidden layers
        for hidden_dim in config.hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            if config.batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
                
            layers.append(self._get_activation(config.activation))
            
            if config.dropout > 0:
                layers.append(nn.Dropout(config.dropout))
                
            prev_dim = hidden_dim
        
        # Add output layer
        layers.append(nn.Linear(prev_dim, config.output_dim))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.model(x)
    
    def _get_activation(self, activation_name: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            "relu": nn.ReLU(),
            "leaky_relu": nn.LeakyReLU(0.2),
            "sigmoid": nn.Sigmoid(),
            "tanh": nn.Tanh(),
            "softmax": nn.Softmax(dim=1),
            "elu": nn.ELU(),
            "selu": nn.SELU(),
        }
        
        if activation_name.lower() not in activations:
            raise ValueError(f"Unknown activation: {activation_name}")
            
        return activations[activation_name.lower()] 