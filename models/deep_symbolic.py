import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import sympy as sp
from dataclasses import dataclass

# Base function classes and implementations
class BaseFunction:
    """Abstract class for primitive functions"""
    def __init__(self, norm=1):
        self.norm = norm

    def sp(self, x):
        """Sympy implementation"""
        return None

    def torch(self, x):
        """No need for base function"""
        return None

    def name(self, x):
        return str(self.sp)

class BaseFunction2:
    """Abstract class for primitive functions with 2 inputs"""
    def __init__(self, norm=1.):
        self.norm = norm

    def sp(self, x, y):
        """Sympy implementation"""
        return None

    def torch(self, x, y):
        return None

    def name(self, x, y):
        return str(self.sp)

class Constant(BaseFunction):
    def torch(self, x):
        return torch.ones_like(x)

    def sp(self, x):
        return 1

class Identity(BaseFunction):
    def torch(self, x):
        return x / self.norm

    def sp(self, x):
        return x / self.norm

class Square(BaseFunction):
    def torch(self, x):
        return torch.square(x) / self.norm

    def sp(self, x):
        return x ** 2 / self.norm

class Sin(BaseFunction):
    def torch(self, x):
        return torch.sin(x * 2 * 2 * np.pi) / self.norm

    def sp(self, x):
        return sp.sin(x * 2*2*np.pi) / self.norm

class Exp(BaseFunction):
    def __init__(self, norm=np.e):
        super().__init__(norm)

    def torch(self, x):
        return (torch.exp(x) - 1) / self.norm

    def sp(self, x):
        return (sp.exp(x) - 1) / self.norm

class Sigmoid(BaseFunction):
    def torch(self, x):
        return torch.sigmoid(x) / self.norm

    def sp(self, x):
        return 1 / (1 + sp.exp(-20*x)) / self.norm

    def name(self, x):
        return "sigmoid(x)"

class Product(BaseFunction2):
    def __init__(self, norm=0.1):
        super().__init__(norm=norm)

    def torch(self, x, y):
        return x * y / self.norm

    def sp(self, x, y):
        return x*y / self.norm




def count_double(funcs):
    """Count number of functions that take 2 inputs"""
    return sum(1 for f in funcs if isinstance(f, BaseFunction2))

# Pretty print functions
def apply_activation(W, funcs, n_double=0):
    """Given an (n, m) matrix W and (m) vector of funcs, apply funcs to W."""
    W = sp.Matrix(W)
    if n_double == 0:
        for i in range(W.shape[0]):
            for j in range(W.shape[1]):
                W[i, j] = funcs[j](W[i, j])
    else:
        W_new = W.copy()
        out_size = len(funcs)
        for i in range(W.shape[0]):
            in_j = 0
            out_j = 0
            while out_j < out_size - n_double:
                W_new[i, out_j] = funcs[j](W[i, in_j])
                in_j += 1
                out_j += 1
            while out_j < out_size:
                W_new[i, out_j] = funcs[j](W[i, in_j], W[i, in_j+1])
                in_j += 2
                out_j += 1
        for i in range(n_double):
            W_new.col_del(-1)
        W = W_new
    return W

def filter_mat(mat, threshold=0.01):
    """Remove elements of a matrix below a threshold."""
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if abs(mat[i, j]) < threshold:
                mat[i, j] = 0
    return mat

def sym_pp(W_list, funcs, var_names, threshold=0.01, n_double=0):
    """Pretty print the hidden layers of the symbolic regression network"""
    vars = []
    for var in var_names:
        if isinstance(var, str):
            vars.append(sp.Symbol(var))
        else:
            vars.append(var)
    expr = sp.Matrix(vars).T
    for W in W_list:
        W = filter_mat(sp.Matrix(W), threshold=threshold)
        expr = expr * W
        expr = apply_activation(expr, funcs, n_double=n_double)
    return expr

def last_pp(eq, W):
    """Pretty print the last layer."""
    return eq * filter_mat(sp.Matrix(W))

def network(weights, funcs, var_names, threshold=0.01):
    """Pretty print the entire symbolic regression network."""
    n_double = count_double(funcs)
    funcs = [func.sp for func in funcs]
    expr = sym_pp(weights[:-1], funcs, var_names, threshold=threshold, n_double=n_double)
    expr = last_pp(expr, weights[-1])
    expr = expr[0, 0]
    return expr

# Symbolic network classes
class SymbolicLayerL0(nn.Module):
    """Neural network layer for symbolic regression with L0 regularization"""
    def __init__(self, in_dim=None, funcs=None, initial_weight=None, init_stddev=0.1,
                 bias=False, droprate_init=0.5, lamba=1.,
                 beta=2/3, gamma=-0.1, zeta=1.1, epsilon=1e-6):
        super().__init__()
        
        if funcs is None:
            funcs = [
                *[Constant()] * 2,
                *[Identity()] * 4,
                *[Square()] * 4,
                *[Sin()] * 2,
                *[Exp()] * 2,
                *[Sigmoid()] * 2,
                *[Product(1.0)] * 2
            ]
            
        self.initial_weight = initial_weight
        self.W = None
        self.built = False
        
        self.output = None
        self.n_funcs = len(funcs)
        self.funcs = [func.torch for func in funcs]
        self.n_double = count_double(funcs)
        self.n_single = self.n_funcs - self.n_double
        
        self.out_dim = self.n_funcs + self.n_double
        
        if self.initial_weight is not None:
            self.W = nn.Parameter(self.initial_weight.clone().detach())
            self.built = True
        else:
            self.W = torch.normal(mean=0.0, std=init_stddev, size=(in_dim, self.out_dim))
            
        self.droprate_init = droprate_init if droprate_init != 0 else 0.5
        self.use_bias = bias
        self.lamba = lamba
        self.bias = None
        self.in_dim = in_dim
        self.eps = None
        
        self.beta = beta
        self.gamma = gamma
        self.zeta = zeta
        self.epsilon = epsilon
        
        if self.use_bias:
            self.bias = nn.Parameter(0.1 * torch.ones((1, self.out_dim)))
        self.qz_log_alpha = nn.Parameter(torch.normal(
            mean=np.log(1 - self.droprate_init) - np.log(self.droprate_init),
            std=1e-2, size=(in_dim, self.out_dim)
        ))

    def quantile_concrete(self, u):
        y = torch.sigmoid((torch.log(u) - torch.log(1.0 - u) + self.qz_log_alpha) / self.beta)
        return y * (self.zeta - self.gamma) + self.gamma

    def sample_u(self, shape, reuse_u=False):
        if self.eps is None or not reuse_u:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.eps = torch.rand(size=shape).to(device) * (1 - 2 * self.epsilon) + self.epsilon
        return self.eps

    def sample_z(self, batch_size, sample=True):
        if sample:
            eps = self.sample_u((batch_size, self.in_dim, self.out_dim))
            z = self.quantile_concrete(eps)
            return torch.clamp(z, min=0, max=1)
        else:
            pi = torch.sigmoid(self.qz_log_alpha)
            return torch.clamp(pi * (self.zeta - self.gamma) + self.gamma, min=0.0, max=1.0)

    def get_z_mean(self):
        pi = torch.sigmoid(self.qz_log_alpha)
        return torch.clamp(pi * (self.zeta - self.gamma) + self.gamma, min=0.0, max=1.0)

    def sample_weights(self, reuse_u=False):
        z = self.quantile_concrete(self.sample_u((self.in_dim, self.out_dim), reuse_u=reuse_u))
        mask = torch.clamp(z, min=0.0, max=1.0)
        return mask * self.W

    def get_weight(self):
        return self.W * self.get_z_mean()

    def loss(self):
        return torch.sum(torch.sigmoid(self.qz_log_alpha - self.beta * np.log(-self.gamma / self.zeta)))

    def forward(self, x, sample=True, reuse_u=False):
        if sample:
            h = torch.matmul(x, self.sample_weights(reuse_u=reuse_u))
        else:
            w = self.get_weight()
            h = torch.matmul(x, w)

        if self.use_bias:
            h = h + self.bias

        output = []
        in_i = 0
        out_i = 0
        while out_i < self.n_single:
            output.append(self.funcs[out_i](h[:, in_i]))
            in_i += 1
            out_i += 1
        while out_i < self.n_funcs:
            output.append(self.funcs[out_i](h[:, in_i], h[:, in_i + 1]))
            in_i += 2
            out_i += 1
        output = torch.stack(output, dim=1)
        return output


activation_funcs = [
  *[Constant()] * 2,
  *[Identity()] * 4,
  *[Square()] * 4,
  *[Sin()] * 2,
  *[Exp()] * 2,
  *[Sigmoid()] * 2,
  *[Product(1.0)] * 2
]


@dataclass
class SymbolicNetConfig:
    symbolic_depth: int
    in_dim: int


class SymbolicNetL0(nn.Module):
    """Symbolic regression network with multiple layers and L0 regularization"""
    def __init__(self, symbolic_depth, in_dim=1, funcs=activation_funcs, initial_weights=None, init_stddev=0.1):
        super(SymbolicNetL0, self).__init__()


        self.depth = symbolic_depth
        self.funcs = funcs

        width = len(funcs)
        n_double = count_double(funcs)

        if initial_weights is None:
            initial_weights = [
                torch.fmod(torch.normal(0, 0.1, size=(in_dim, width + n_double)), 2),
                torch.fmod(torch.normal(0, 0.5, size=(width, width + n_double)), 2),
                torch.fmod(torch.normal(0, 0.5, size=(width, width + n_double)), 2),
                torch.fmod(torch.normal(0, 1.0, size=(width, 1)), 2)
            ]
        
        layer_in_dim = [in_dim] + self.depth * [len(funcs)]
        if initial_weights is not None:
            layers = [SymbolicLayerL0(funcs=funcs, initial_weight=initial_weights[i],
                                    in_dim=layer_in_dim[i])
                     for i in range(self.depth)]
            self.output_weight = nn.Parameter(initial_weights[-1].clone().detach())
        else:
            if not isinstance(init_stddev, list):
                init_stddev = [init_stddev] * self.depth
            layers = [SymbolicLayerL0(funcs=funcs, init_stddev=init_stddev[i], in_dim=layer_in_dim[i])
                     for i in range(self.depth)]
            self.output_weight = nn.Parameter(torch.rand(size=(layers[-1].n_funcs, 1)) * 2)
        self.hidden_layers = nn.Sequential(*layers)

    def forward(self, input, sample=True, reuse_u=False):
        h = input
        for i in range(self.depth):
            h = self.hidden_layers[i](h, sample=sample, reuse_u=reuse_u)
        h = torch.matmul(h, self.output_weight)
        return h

    def get_loss(self):
        return torch.sum(torch.stack([self.hidden_layers[i].loss() for i in range(self.depth)]))

    def get_weights(self):
        return [self.hidden_layers[i].get_weight().cpu().detach().numpy() for i in range(self.depth)] + \
               [self.output_weight.cpu().detach().numpy()]

class CustomTrainer:
    def __init__(self, results_dir='results/custom'):
        # Set activation function
        self.activation_funcs = [
            *[Constant()] * 2,
            *[Identity()] * 4,
            *[Square()] * 4,
            *[Sin()] * 2,
            *[Exp()] * 2,
            *[Sigmoid()] * 2,
            *[Product(1.0)] * 2
        ]
        
        # Network parameters
        self.n_layers = 2
        self.reg_weight = 5e-3
        self.learning_rate = 1e-2
        self.n_epochs1 = 10001
        self.n_epochs2 = 10001
        self.results_dir = results_dir
        
        # Create results directory
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

    def train(self, X_train, y_train, X_test, y_test, trials=1):
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        print(f"Using device: {device}")
        
        # Ensure data is float32 type
        data = torch.tensor(X_train, dtype=torch.float32).to(device)
        target = torch.tensor(y_train, dtype=torch.float32).to(device)
        test_data = torch.tensor(X_test, dtype=torch.float32).to(device)
        test_target = torch.tensor(y_test, dtype=torch.float32).to(device)
        
        print(f"Training data shape: {data.shape}")
        print(f"Target data shape: {target.shape}")
        
        x_dim = 1  # Input dimension is 1
        width = len(self.activation_funcs)
        n_double = count_double(self.activation_funcs)
        
        best_expr = None
        best_test_loss = float('inf')
        
        for trial in range(trials):
            print(f"\nTrial {trial + 1}/{trials}")
            
            # Initialize network
            net = SymbolicNetL0(
                self.n_layers,
                funcs=self.activation_funcs,
                initial_weights=[
                    torch.fmod(torch.normal(0, 0.1, size=(x_dim, width + n_double)), 2),
                    torch.fmod(torch.normal(0, 0.5, size=(width, width + n_double)), 2),
                    torch.fmod(torch.normal(0, 0.5, size=(width, width + n_double)), 2),
                    torch.fmod(torch.normal(0, 1.0, size=(width, 1)), 2)
                ]
            ).to(device)
            
            # Set optimizer
            criterion = nn.MSELoss()
            optimizer = optim.RMSprop(
                net.parameters(),
                lr=self.learning_rate * 10,
                alpha=0.9,
                eps=1e-10,
                momentum=0.0,
                centered=False
            )
            
            # Training loop
            for epoch in range(self.n_epochs1 + self.n_epochs2 + 2000):
                optimizer.zero_grad()
                outputs = net(data)
                mse_loss = criterion(outputs, target)
                reg_loss = net.get_loss()
                loss = mse_loss + self.reg_weight * reg_loss
                loss.backward()
                optimizer.step()
                
                if epoch % 1000 == 0:
                    with torch.no_grad():
                        test_outputs = net(test_data)
                        test_loss = criterion(test_outputs, test_target)
                        print(f"Epoch {epoch}, Loss: {loss.item():.6f}, Test Loss: {test_loss.item():.6f}")
            
            # Get final results
            with torch.no_grad():
                weights = net.get_weights()
                expr = network(weights, self.activation_funcs, ["x"])
                test_outputs = net(test_data)
                final_test_loss = criterion(test_outputs, test_target).item()
                
                print(f"\nTrial {trial + 1} Results:")
                print(f"Discovered expression: {expr}")
                print(f"Final test loss: {final_test_loss:.6f}")
                
                # Save best results
                if final_test_loss < best_test_loss:
                    best_test_loss = final_test_loss
                    best_expr = expr
                    
                    # Save model weights
                    torch.save(net.state_dict(), os.path.join(self.results_dir, 'best_model.pth'))
        
        print("\nBest Results:")
        print(f"Best expression: {best_expr}")
        print(f"Best test loss: {best_test_loss:.6f}")
        
        return best_expr, best_test_loss

def load_data(X, y, test_size=0.2):
    """Split data into training and test sets"""
    n_samples = len(X)
    indices = np.random.permutation(n_samples)
    test_size = int(n_samples * test_size)
    
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]
    
    X_train = X[train_indices]
    y_train = y[train_indices]
    X_test = X[test_indices]
    y_test = y[test_indices]
    
    return X_train, y_train, X_test, y_test

if __name__ == "__main__":
    # Example: Generate some test data
    # Replace with your actual data loading code
    X = np.random.randn(163842, 1)  # Input data
    y = np.sin(X) + 0.1 * np.random.randn(163842, 1)  # Output data
    
    # Split data into training and test sets
    X_train, y_train, X_test, y_test = load_data(X, y)
    
    # Create trainer and train
    trainer = CustomTrainer()
    best_expr, best_loss = trainer.train(X_train, y_train, X_test, y_test, trials=3)