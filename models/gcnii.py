import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class GCNIIConfig:
    input_dim: int
    hidden_dim: int
    output_dim: int
    num_layers: int
    alpha: float = 0.1
    lambda_val: float = 0.5
    dropout: float = 0.5
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class GCNIIConv(nn.Module):
    def __init__(self, hidden_dim, alpha, layer_idx, lambda_val):
        super(GCNIIConv, self).__init__()
        self.alpha = alpha
        self.beta = lambda_val / (layer_idx + 1)
        self.linear = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, x0, edge_index, edge_weight=None):
        num_nodes = x.size(0)
        
        # Process edge weights
        if edge_weight is None:
            edge_weight = torch.ones(edge_index.size(1), device=edge_index.device)
            
        # Add self-loops
        self_loop_idx = torch.stack((
            torch.arange(num_nodes, device=edge_index.device),
            torch.arange(num_nodes, device=edge_index.device)
        ))
        self_loop_val = torch.ones(num_nodes, device=edge_index.device)
        
        # Merge original edges and self-loops
        indices = torch.cat((self_loop_idx, edge_index), dim=1)
        values = torch.cat((self_loop_val, edge_weight))
        
        # Create sparse adjacency matrix
        adj = torch.sparse_coo_tensor(
            indices, values, 
            (num_nodes, num_nodes)
        ).coalesce()
        
        # Normalize adjacency matrix
        rowsum = torch.sparse.sum(adj, dim=1).to_dense()
        d_inv = torch.pow(rowsum, -0.5)
        d_inv[torch.isinf(d_inv)] = 0.
        d_mat_inv = torch.sparse_coo_tensor(
            torch.stack((
                torch.arange(num_nodes, device=edge_index.device),
                torch.arange(num_nodes, device=edge_index.device)
            )),
            d_inv,
            (num_nodes, num_nodes)
        )
        
        # Calculate normalized adjacency matrix
        adj = torch.sparse.mm(torch.sparse.mm(d_mat_inv, adj), d_mat_inv)

        out = (1 - self.alpha) * (torch.sparse.mm(adj, x) * self.beta + x * (1 - self.beta)) + self.alpha * x0
        out = self.linear(out)
        return out


class GCNII(nn.Module):
    def __init__(self, config: GCNIIConfig):
        super(GCNII, self).__init__()
        self.config = config

        self.input_fc = nn.Linear(config.input_dim, config.hidden_dim)
        self.layers = nn.ModuleList()
        for i in range(config.num_layers):
            self.layers.append(GCNIIConv(config.hidden_dim, config.alpha, i, config.lambda_val))
        self.output_fc = nn.Linear(config.hidden_dim, config.output_dim)
        self.dropout = config.dropout

    def forward(self, x, edge_index, edge_weight=None):
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.input_fc(x))
        x0 = x  # save initial features

        for layer in self.layers:
            x = F.dropout(x, self.dropout, training=self.training)
            x = F.relu(layer(x, x0, edge_index, edge_weight))

        x = F.dropout(x, self.dropout, training=self.training)
        out = self.output_fc(x)
        return out
