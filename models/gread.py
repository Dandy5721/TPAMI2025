import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_sparse
from torch.nn.init import xavier_uniform_
from torchdiffeq import odeint


class GREADConfig:
    """Configuration for GREAD model."""
    def __init__(self, input_dim, hidden_dim, output_dim, device):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.device = device

# python -m models.gread

class ODEFunc(nn.Module):
    def __init__(self, device):
        super(ODEFunc, self).__init__()
        self.device = device
        self.edge_index = None
        self.edge_weight = None
        self.attention_weights = None
        
        # Trainable parameters
        self.alpha_train = nn.Parameter(torch.tensor(0.0))
        self.source_train = nn.Parameter(torch.tensor(0.0))
        self.beta_train = nn.Parameter(torch.tensor(0.0))
        
        self.x0 = None
        self.nfe = 0
        self.alpha_sc = nn.Parameter(torch.ones(1))
        self.source_sc = nn.Parameter(torch.ones(1))

class ODEFuncGread(ODEFunc):
    def __init__(self, in_features, out_features, device):
        super(ODEFuncGread, self).__init__(device)
        
        self.in_features = in_features
        self.out_features = out_features
        
        self.reaction_tanh = False
        self.b_W = nn.Parameter(torch.Tensor(in_features, 1))
        self.reset_parameters()
        self.epoch = 0
        # self.max_nfe = 2000
    
    def reset_parameters(self):
        xavier_uniform_(self.b_W)
    
    def set_Beta(self):
        Beta = torch.diag(self.b_W.squeeze())
        return Beta

    def sparse_multiply(self, x, edge_index, edge_weight):
        # Modified to accept adjacency matrix for each sample
        ax = torch_sparse.spmm(edge_index, edge_weight, x.shape[0], x.shape[0], x)
        return ax

    def forward(self, t, x, edge_index, edge_weight):
        # if self.nfe > self.max_nfe:
        #     raise Exception("Maximum number of function evaluations reached")
        # self.nfe += 1
        
        # Limit parameter ranges for stability
        alpha = torch.sigmoid(self.alpha_train) * 0.1  # Reduce alpha range
        beta = torch.sigmoid(self.beta_train) * 0.1    # Reduce beta range

        # Diffusion term
        ax = self.sparse_multiply(x, edge_index, edge_weight)
        diffusion = (ax - x)

        # Reaction term (Fisher)
        reaction = -(x-1)*x
        
        # Final reaction-diffusion form
        f = alpha*diffusion + beta*reaction
        
        # Add source term (add check to ensure x0 is not None)
        if self.x0 is not None:
            source_term = self.source_train * self.x0
            # Limit source term size
            source_term = torch.clamp(source_term, -1.0, 1.0)
            f = f + source_term * 0.1  # Reduce source term influence
        else:
            self.x0 = x.clone().detach()
            f = f + self.source_train * self.x0 * 0.1
        
        return f

class ODEblock(nn.Module):
    def __init__(self, odefunc, device, t=torch.tensor([0, 1])):
        super(ODEblock, self).__init__()
        self.t = t
        self.odefunc = odefunc
        self.device = device
        
        # ODE solver settings
        self.method = 'explicit_adams'
        self.step_size = 0.1
        self.max_iters = 1000
        self.atol = 1e-6
        self.rtol = 1e-6

    def set_x0(self, x0):
        self.odefunc.x0 = x0.clone().detach()

    def forward(self, x, edge_index, edge_weight):
        t = self.t.type_as(x)
        state = x
        
        # Modify ODE solver to support batch processing
        def ode_func(t, state):
            return self.odefunc(t, state, edge_index, edge_weight)
        
        solution = odeint(
            ode_func, state, t,
            method=self.method,
            options=dict(step_size=self.step_size, max_iters=self.max_iters),
            atol=self.atol,
            rtol=self.rtol
        )
        
        return solution[-1]

class GREAD(nn.Module):
    def __init__(self, config: GREADConfig):
        super(GREAD, self).__init__()
        self.hidden_dim = config.hidden_dim
        self.input_dim = config.input_dim
        self.output_dim = config.output_dim
        self.device = config.device
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        # ODE function and solver
        self.odefunc = ODEFuncGread(self.hidden_dim, self.hidden_dim, self.device)
        self.odeblock = ODEblock(self.odefunc, self.device)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(self.hidden_dim, self.output_dim)
        )
        
        # Batch normalization
        self.bn_in = nn.BatchNorm1d(self.hidden_dim)
        self.bn_out = nn.BatchNorm1d(self.hidden_dim)

    def forward(self, curr_x, edge_index, edge_weight):

        # Encoding
        curr_x = self.encoder(curr_x)  
        curr_x = self.bn_in(curr_x)
            
        # Set initial state
        self.odeblock.set_x0(curr_x)
        self.odefunc.Beta = self.odefunc.set_Beta()
            
        # Solve ODE
        curr_x = self.odeblock(curr_x, edge_index, edge_weight)
            
        # Decoding
        curr_x = self.bn_out(curr_x)
        output = self.decoder(curr_x)  
        
        return output

if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create test data
    nodes = 163842
    input_dim = 1
    
    # Create random input data - without batch dimension
    x = torch.rand(nodes, input_dim).to(device)
    
    # Create random adjacency matrix (keep only some edges to simulate real graph)
    adj = torch.rand(nodes, nodes).to(device)
    adj = (adj < 0.2).float()  # Make sparse
    # Ensure symmetry
    adj = torch.logical_or(adj, adj.t()).float()
    # Add self-loops
    adj = adj + torch.eye(nodes).to(device)
    
    print(f"Input shape: {x.shape}")
    print(f"Adjacency matrix shape: {adj.shape}")
    
    # Test ODEFuncGread
    print("\n--- Testing ODEFuncGread ---")
    odefunc = ODEFuncGread(64, 64, device).to(device)
    # Generate random hidden state and edge information
    h = torch.rand(nodes, 64).to(device)
    edge_index = torch.nonzero(adj).t()
    edge_weight = adj[edge_index[0], edge_index[1]]
    # Test forward pass
    t = torch.tensor(0.0).to(device)
    out = odefunc(t, h, edge_index, edge_weight)
    print(f"ODEFuncGread output shape: {out.shape}")
    
    # Test ODEblock
    print("\n--- Testing ODEblock ---")
    odeblock = ODEblock(odefunc, device).to(device)
    odeblock.set_x0(h)
    out = odeblock(h, edge_index, edge_weight)
    print(f"ODEblock output shape: {out.shape}")
    
    # Test entire GNN model
    print("\n--- Testing GNN ---")
    model = GREAD(input_dim=input_dim, hidden_dim=64, output_dim=1, device=device).to(device)

    print(f"x shape: {x.shape}")
    # Switch to evaluation mode to disable dropout
    model.eval()
    with torch.no_grad():
        output = model(x, adj)
    print(f"GNN output shape: {output.shape}")
    
    # Test model components
    print("\n--- Testing GNN components individually ---")
    print(f"x shape: {x.shape}")
    print(f"adj shape: {adj.shape}")
    print(f"edge_index shape: {edge_index.shape}")
    print(f"edge_weight shape: {edge_weight.shape}")
    
    # Encoder
    print("Encoder test:")
    encoded = model.encoder(x)
    print(f"  Input: {x.shape} -> Encoded: {encoded.shape}")
    
    # Normalization
    normalized = model.bn_in(encoded)
    print(f"  BatchNorm: {encoded.shape} -> {normalized.shape}")
    
    # ODE module
    model.odeblock.set_x0(normalized)
    model.odefunc.Beta = model.odefunc.set_Beta()
    ode_out = model.odeblock(normalized, edge_index, edge_weight)
    print(f"  ODE: {normalized.shape} -> {ode_out.shape}")
    
    # Decoder
    bn_out = model.bn_out(ode_out)
    decoded = model.decoder(bn_out)
    print(f"  Decoder: {bn_out.shape} -> {decoded.shape}")
    
    print("\nAll tests completed successfully!")