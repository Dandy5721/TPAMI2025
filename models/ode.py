import torch
import torch.nn as nn
from torchdiffeq import odeint
from dataclasses import dataclass

@dataclass
class ODEConfig:
    """Configuration for ODE model."""
    input_dim: int
    hidden_dim: int
    output_dim: int

# -----------------------------
# GCN Layer (supports [N, F] input)
# -----------------------------
class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x, adj):
        support = self.linear(x)          # [N, F_out]
        # Use sparse matrix multiplication
        if adj.is_sparse:
            out = torch.sparse.mm(adj, support)  # [N, F_out]
        else:
            out = adj @ support               # [N, F_out]
        return out

# -----------------------------
# ODE Function (GCN as propagation function)
# -----------------------------
class ODEFunc(nn.Module):
    def __init__(self, in_dim=1, hidden_dim=32, out_dim=1):
        super().__init__()
        self.gcn1 = GCNLayer(in_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.gcn2 = GCNLayer(hidden_dim, out_dim)  # Ensure output dimension is 1
        self.adj = None  # Adjacency matrix will be set before forward

    def set_adj(self, adj):
        # Set adjacency matrix as current instance attribute
        self.adj = adj

    def forward(self, t, x):
        # Ensure adj is set
        if self.adj is None:
            raise ValueError("Adjacency matrix (adj) has not been set, please call set_adj method first")
        
        x = self.gcn1(x, self.adj)
        x = self.relu(x)
        x = self.gcn2(x, self.adj)  # This will ensure output dimension is 1
        return x[:, :1]  # Force output to be [N, 1]

# -----------------------------
# Graph Neural ODE Model
# -----------------------------
class GraphNeuralODE(nn.Module):
    def __init__(self, config: ODEConfig):
        super().__init__()
        # Initialize layers
        self.gcn1 = GCNLayer(config.input_dim, 1)
        # Explicitly specify out_dim=1
        self.func = ODEFunc(in_dim=1, hidden_dim=config.hidden_dim, out_dim=config.output_dim)

    def forward(self, x0, t, adj):
        # Set ODEFunc's adjacency matrix
        self.func.set_adj(adj)
        # Ensure time tensor is on the same device
        if x0.device != t.device:
            t = t.to(x0.device)

        # Process input with GCN layer
        x0 = self.gcn1(x0, adj)
        
        # Solve ODE
        out = odeint(self.func, x0, t, method='rk4')  # [2, N, 2]
        return out[-1][:, :1]  # Only take the first feature of output at t=1



if __name__ == "__main__":
    model = GraphNeuralODE(in_dim=2, hidden_dim=32, out_dim=1)
    x0 = torch.rand(163842, 2)
    t = torch.tensor([0., 1.])
    adj = torch.rand(163842, 163842)
    out = model(x0, t, adj)
    print(out.shape)