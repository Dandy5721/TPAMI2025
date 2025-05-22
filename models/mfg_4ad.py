import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj
# import torch_scatter
from utils import functions as functions
from utils.functions import count_double
import numpy as np
import torch.nn.functional as F
from utils.symbolic_network import SymbolicNetL00
from torch_geometric.nn import GCNConv
import torch.nn.utils.spectral_norm as SN
from torch_geometric.nn import GCNConv, global_mean_pool
import time
from scipy.linalg import expm
from torch.linalg import matrix_exp
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import expm_multiply
from dataclasses import dataclass

def predict_next_tau(u_t, edge_index, edge_weight, beta=0.05, delta_t=1.0):
    """
    Predict tau(t+1) using network diffusion model.
    u_t: (N,), A: (N, N)
    """
    device = u_t.device
    # tensor to numpy
    u_t = u_t.cpu().numpy()
    
    # Convert to sparse adjacency matrix
    N = u_t.shape[0]
    row, col = edge_index.cpu().numpy()
    A_sparse = csr_matrix((edge_weight.cpu().numpy(), (row, col)), shape=(N, N))
    
    # Use sparse implementation
    res = predict_next_tau_sparse(u_t, A_sparse, beta, delta_t)
    
    return torch.from_numpy(res).unsqueeze(0).to(device)




def predict_next_tau_torch_bk(u_t, A, beta=0.05, delta_t=1.0):
    """
    u_t: (N,) torch tensor
    A: (N, N) torch tensor (adjacency matrix)
    """
    D = torch.diag(A.sum(dim=1))
    H = D - A
    diffusion_op = matrix_exp(-beta * H * delta_t)

    res = diffusion_op @ u_t
    return res.unsqueeze(0)


def predict_next_tau_torch(u_t, edge_index, edge_weight, beta=0.05, delta_t=1.0):
    """
    u_t: (N,) torch tensor
    edge_index: (2, E) torch tensor
    edge_weight: (E,) torch tensor
    """
    device = u_t.device
    N = u_t.shape[0]
    
    # Convert to numpy for scipy sparse operations
    u_t_np = u_t.cpu().numpy()
    row, col = edge_index.cpu().numpy()
    edge_weight_np = edge_weight.cpu().numpy()
    
    # Create sparse adjacency matrix
    A_sparse = csr_matrix((edge_weight_np, (row, col)), shape=(N, N))
    
    # Compute degree matrix (sparse)
    D_sparse = diags(A_sparse.sum(axis=1).A1)
    
    # Compute Laplacian matrix (sparse)
    H_sparse = D_sparse - A_sparse
    
    # Compute diffusion operator using sparse matrix exponential
    diffusion_op = expm_multiply(-beta * H_sparse * delta_t, u_t_np)
    
    # Convert back to torch tensor
    res = torch.from_numpy(diffusion_op).to(device)
    return res.unsqueeze(0)


def predict_next_tau_sparse(u_t_np, A_np, beta=0.05, delta_t=1.0):
    """
    Predict tau(t+1) using network diffusion model with sparse matrices.
    u_t_np: (N,) numpy array
    A_np: (N, N) scipy sparse matrix
    """
    # Convert to sparse matrix if not already
    if not isinstance(A_np, csr_matrix):
        A_np = csr_matrix(A_np)
    
    # Compute degree matrix
    D = diags(A_np.sum(axis=1).A1)
    
    # Compute Laplacian matrix
    H = D - A_np
    
    # Compute diffusion operator using sparse matrix exponential
    diffusion_op = expm_multiply(-beta * H * delta_t, u_t_np)
    
    return diffusion_op

# generator - refactored to explicitly define each layer
class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim=1):
        super().__init__()
        
        # Define layers
        self.layers = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        # Build all layers
        dims = [input_dim] + hidden_dims + [output_dim]
        for i in range(len(dims)-1):
            self.layers.append(GCNConv(dims[i], dims[i+1]))
            # Add BatchNorm to all layers except the last one
            if i < len(dims)-2:
                self.bns.append(nn.BatchNorm1d(dims[i+1]))
        
        # Dropout layer
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, edge_index, edge_weight):
        pre = x
        for i, (layer, bn) in enumerate(zip(self.layers[:-1], self.bns)):
            pre = layer(pre, edge_index, edge_weight)
            pre = bn(pre)
            pre = F.relu(pre)
            pre = self.dropout(pre)
        
        # Last layer has no activation function
        pre = self.layers[-1](pre, edge_index, edge_weight)
        
        return pre.unsqueeze(0).squeeze(-1)


activation_funcs = [
    *[functions.Constant()] * 2,
    *[functions.Identity()] * 4,
    *[functions.Square()]   * 4,
    *[functions.Sin()]      * 2,
    *[functions.Exp()]      * 2,
    *[functions.Sigmoid()]  * 2,
    *[functions.Product(1.0)] * 2,
]

# Refactor MLP class, directly define each layer
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=[320, 240, 200, 160, 120, 80], output_dim=160):
        super().__init__()

        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

        # Build hidden layers
        dims = [input_dim] + hidden_dims
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            self.layers.append(nn.Linear(in_dim, out_dim))
            self.norms.append(nn.LayerNorm(out_dim))

        # Output layer (no Norm and activation)
        self.output_layer = nn.Linear(dims[-1], output_dim)

    def forward(self, x):
        if x.dim() > 2:
            x = x.squeeze(-1)

        for fc, norm in zip(self.layers, self.norms):
            x = fc(x)
            x = norm(x)
            x = self.relu(x)
            x = self.dropout(x)

        x = self.output_layer(x)
        return x
    

@dataclass
class MFG4ADGeneratorConfig:
    num_nodes: int
    input_dim: int
    dt: float
    hidden_gcn: list
    hidden_flow: list
    mlp_hidden: list
    ablation: str
    device: str

class MFG4ADGenerator(nn.Module):
    """
    Generator with advection–diffusion–reaction physics on a graph,
    inferring a 3D velocity field per node.
    """
    def __init__(
        self,
        config: MFG4ADGeneratorConfig
    ):
        super().__init__()
        self.N = config.num_nodes
        self.dt = config.dt
        self.ablation = config.ablation

        # GCN for diffusion term P
        self.gcn = GCN(input_dim=1, hidden_dims=config.hidden_gcn, output_dim=1)
        # Symbolic net for reaction term Q
        width = len(activation_funcs)
        n_double = count_double(activation_funcs)
        self.symbolic_net = SymbolicNetL00(
            symbolic_depth=1,
            initial_weights=[
                torch.fmod(torch.normal(0, 0.1, size=(config.input_dim, width + n_double)), 2),
                torch.fmod(torch.normal(0, 0.5, size=(width, width + n_double)), 2),
                torch.fmod(torch.normal(0, 0.5, size=(width, width + n_double)), 2),
                torch.fmod(torch.normal(0, 1.0, size=(width, 1)), 2)
            ],
            in_dim=2,
            funcs=activation_funcs
        )
        # Flow MLP to predict 3D velocity vector per node
        self.flow_mlp = MLP(
            input_dim=3,        # F = [u, P, Q]
            hidden_dims=config.hidden_flow,
            output_dim=3        # 3D velocity
        )
        # Final MLP: input size = 3 * N, output size = N
        self.mlp = MLP(
            input_dim=3 * config.num_nodes,
            hidden_dims=config.mlp_hidden,
            output_dim=config.num_nodes
        )

        self.gamma = nn.Parameter(torch.tensor(1.0))
        
        self.lambda_ = nn.Parameter(torch.tensor(1.0))
        
        self.mu     = nn.Parameter(torch.tensor(1.0))

    def print_params(self):
        print(f"gamma: {self.gamma.item()}")
        print(f"lambda: {self.lambda_.item()}")
        print(f"mu: {self.mu.item()}")


    def forward(self, tau: torch.Tensor,
                amyloid: torch.Tensor, edge_index: torch.Tensor,
                edge_weight: torch.Tensor) -> torch.Tensor:
        """
        :param tau:       shape (B, N)
        :param amyloid:   shape (B, N)
        :param edge_index, edge_weight: graph connectivity
        :return: updated tau shape (B, N)
        """
        B, N = tau.shape
        assert N == self.N, f"Expected {self.N} nodes, got {N}"
        device = tau.device

        # Node states
        u = tau                                    # (B, N)
        origin_x = u.unsqueeze(-1)                 # (B, N, 1)
        am = amyloid.unsqueeze(-1)                 # (B, N, 1)

        # Diffusion term P
        if self.ablation == "all" or self.ablation == "only_diffusion":
            P = predict_next_tau_torch(u.squeeze(0), edge_index, edge_weight)
            # P = self.gcn(origin_x.view(B * N, 1),
            #             edge_index, edge_weight).view(B, N) #anothor way
        elif self.ablation == "only_symbolic":
            P = torch.zeros_like(origin_x.view(B, N))
        end = time.time()
        # print(f"GCN time: {end - start} seconds")

        # start = time.time()
        # Reaction term Q
        if self.ablation == "all" or self.ablation == "only_symbolic":
            sy_input = torch.cat([origin_x, am], dim=-1).view(B * N, 2)
            Q = self.symbolic_net(sy_input).view(B, N)
        elif self.ablation == "only_diffusion":
            Q = torch.zeros_like(origin_x.view(B, N))
        end = time.time()
        # print(f"SymbolicNet time: {end - start} seconds")

        # 1) Build physics-informed embedding F = [u, P, Q]
        F = torch.stack([ self.gamma * u,  self.lambda_ * P, self.mu * Q], dim=-1).view(B * N, 3)

        # print(f"F shape: {F.shape}")

        # 2) Infer per-node 3D velocity vectors nu
        # start = time.time()
        nu = self.flow_mlp(F).view(B, N, 3)        # (B, N, 3)

        row, col = edge_index    
        # gather u_nu at i and j: shape (B, E, 3)
        u_nu = u.unsqueeze(-1) * nu             # (B, N, 3)
        u_i = u_nu[:, row, :]                   # (B, E, 3)
        u_j = u_nu[:, col, :]                   # (B, E, 3)
        diff_e = u_i - u_j                       # (B, E, 3)
        # weight by edge weight
        flux = diff_e * edge_weight.view(1, -1, 1)  # (B, E, 3)
        # scatter add to nodes
        idx = row.view(1, -1, 1).expand(B, -1, 3)   # (B, E, 3)
        div = torch.zeros(B, N, 3, device=device)
        dx = row.view(1, -1, 1).expand(B, -1, 3)  # (B, E, 3)
        div.scatter_add_(1, idx, flux)             # (B, N, 3)
        # div = torch_scatter.scatter_add(flux, idx, dim=1, dim_size=N)  # (B, N, 3)
        # collapse to scalar divergence if needed
        div_scalar = div.norm(dim=-1)            # (B, N)

        #   d) divergence per dimension: div[b,i,k] = sum_j adj[i,j] * diff[b,i,j,k]
        # div = torch.einsum('ij,bijk->bik', adj, diff)  # (B, N, 3)
        # #   e) collapse spatial dims to scalar divergence if needed
        # div_scalar = div.norm(dim=-1)             # (B, N)

        # 4) Advection–diffusion–reaction update: hat_u
        hat_u = u + self.dt * (-div_scalar + P + Q)  # (B, N)


        # 5) Combine features and predict residual via MLP
        concat = torch.cat([hat_u, P, Q], dim=1)  # (B, 3*N)

        # print(f"concat shape: {concat.shape}")
        res = self.mlp(concat)    
        # end = time.time()
        # print(f"MLP time: {end - start} seconds")
                   # (B, N)

        # print(f"residual shape: {residual.shape}")
        # slice_attention = SliceAttention(num_slices=3)
        # res = slice_attention(concat)
        return res  + tau


@dataclass
class MFG4ADCriticConfig:
    input_dim: int
    hidden_dims: list
    mlp_dims: list
    device: str

class MFG4ADCritic(nn.Module):
    """
    Wasserstein-1 Lipschitz critic for tau distributions,
    with deeper MLP after graph convolution and pooling.
    Uses spectral normalization to enforce 1-Lipschitz.

    Args:
        in_channels (int): number of node features (e.g., 1 for tau concentration).
        hidden_dims (list of int): hidden dimensions for GCN layers.
        mlp_dims (list of int): hidden dims for post-pooling MLP layers.
    """
    def __init__(
        self,
        config: MFG4ADCriticConfig
    ):
        super().__init__()
        # GCN layers with Spectral Norm
        self.convs = nn.ModuleList()
        prev = config.input_dim
        for h in config.hidden_dims:
            conv = GCNConv(prev, h)
            conv.lin = SN(conv.lin)
            self.convs.append(conv)
            prev = h
        # Post-pooling MLP with additional layers
        dims = [prev] + config.mlp_dims + [1]
        self.mlps = nn.ModuleList()
        for in_d, out_d in zip(dims[:-1], dims[1:]):
            linear = SN(nn.Linear(in_d, out_d))
            self.mlps.append(linear)

    def forward(self, x, edge_index, edge_weight=None, batch=None):
        # GCN feature extraction
        for conv in self.convs:
            x = conv(x, edge_index, edge_weight)
            x = F.leaky_relu(x, 0.2)

        # print(f"x shape: {x.shape}")

        # Global pooling
        x = global_mean_pool(x, batch)
        # print(f"global pooling shape: {x.shape}")
        # MLP head
        for lin in self.mlps[:-1]:
            x = F.leaky_relu(lin(x), 0.2)
        score = self.mlps[-1](x)

        # print(f"score shape: {score.shape}")
        return score

# if __name__ == "__main__":
#     # Set random seed for reproducibility
#     torch.manual_seed(42)
    
#     # Test parameters
#     batch_size = 1
#     num_nodes = 163842
#     input_dim = 1
    
#     # Construct random input data
#     tau = torch.randn(batch_size, num_nodes)  
#     target_tau = torch.randn(batch_size, num_nodes)  
#     amyloid = torch.randn(batch_size, num_nodes)  
    
#     # Construct random graph structure
#     edge_index = torch.randint(0, num_nodes, (2, num_nodes * 2))  
#     edge_weight = torch.rand(num_nodes * 2) 
#     adj = to_dense_adj(edge_index, batch=None, edge_attr=edge_weight).squeeze(0)  # (N, N)
    
#     gcn = GCN(input_dim=1, hidden_dims=[1, 8], output_dim=1)
#     gcn_output = gcn(tau.squeeze(0).unsqueeze(-1), edge_index, edge_weight)


#     predict_next_tau_output = predict_next_tau(tau.squeeze(0), edge_index, edge_weight)

#     predict_next_tau_output_torch = predict_next_tau_torch(tau.squeeze(0), edge_index, edge_weight)

    
#     mlp = MLP(input_dim=480, hidden_dims=[400, 320, 240, 200], output_dim=160)
#     mlp_input = torch.randn(batch_size, 480)
#     mlp_output = mlp(mlp_input)

#     generator = Generator(num_nodes=num_nodes)
#     gen_output = generator(tau, target_tau, amyloid, edge_index, edge_weight, adj)
    
#     critic = Critic(in_channels=1)
#     critic_input = torch.randn(num_nodes, 1)  
#     critic_output = critic(critic_input, edge_index, edge_weight)

