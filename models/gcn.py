import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch.nn import Linear
from interfaces import ModelProtocol, ConfigDict
import torch.nn as nn

class GCNConfig:
    """Configuration for GCN model."""
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 16,
        output_dim: int = 1
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

class GCN(nn.Module):
    """Graph Convolutional Network model.
    
    Args:
        config: Configuration for GCN model.
        input_dim: Input dimension of the model.
        hidden_dim: Hidden dimension of the model.
        output_dim: Output dimension of the model.
    """
    
    def __init__(self, config: GCNConfig):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(config.input_dim, config.hidden_dim)
        self.conv2 = GCNConv(config.hidden_dim, config.output_dim)

    def forward(self, x, edge_index, edge_weight):
        """
        Forward pass of the GCN model.
        
        Args:
            x: Node feature matrix of shape [num_nodes, input_dim]
            edge_index: Adjacency matrix in sparse format of shape [2, num_edges]
            edge_weight: Edge weight matrix of shape [num_edges]
        Returns:
            Output features of shape [num_nodes, output_dim]
        """
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_weight)
        return x 