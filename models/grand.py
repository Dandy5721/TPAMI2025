"""Torch module for GCN."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import scipy.sparse as sp


# in_features=2, out_features=1, hidden_features=8, n_layers=30, dropout=0.5
class GRANDConfig:
    """Configuration for GRAND model."""
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, dropout, activation, layer_norm, residual, feat_norm):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.dropout = dropout
        # self.activation = activation
        self.layer_norm = layer_norm
        self.residual = residual
        self.feat_norm = feat_norm


def GCNAdjNorm(adj, order=-0.5):
    r"""

    Description
    -----------
    Normalization of adjacency matrix proposed in `GCN <https://arxiv.org/abs/1609.02907>`__.

    Parameters
    ----------
    adj : scipy.sparse.csr.csr_matrix or torch.FloatTensor
        Adjacency matrix in form of ``N * N`` sparse matrix (or in form of ``N * N`` dense tensor).
    order : float, optional
        Order of degree matrix. Default: ``-0.5``.


    Returns
    -------
    adj : scipy.sparse.csr.csr_matrix
        Normalized adjacency matrix in form of ``N * N`` sparse matrix.

    """
    if sp.issparse(adj):
        adj = sp.eye(adj.shape[0]) + adj
        adj.data[np.where((adj.data > 0) * (adj.data == 1))[0]] = 1
        adj = sp.coo_matrix(adj)
        rowsum = np.array(adj.sum(1))
        d_inv = np.power(rowsum, order).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        adj = d_mat_inv @ adj @ d_mat_inv
    else:
        rowsum = torch.sparse.mm(adj, torch.ones((adj.shape[0], 1), device=adj.device)) + 1
        d_inv = torch.pow(rowsum, order).flatten()
        d_inv[torch.isinf(d_inv)] = 0.

        self_loop_idx = torch.stack((
            torch.arange(adj.shape[0], device=adj.device),
            torch.arange(adj.shape[0], device=adj.device)
        ))
        self_loop_val = torch.ones_like(self_loop_idx[0], dtype=adj.dtype)
        indices = torch.cat((self_loop_idx, adj.indices()), dim=1)
        values = torch.cat((self_loop_val, adj.values()))
        values = d_inv[indices[0]] * values * d_inv[indices[1]]
        adj = torch.sparse.FloatTensor(indices, values, adj.shape).coalesce()

    return adj


class GRAND(nn.Module):
    r"""

    Description
    -----------
    Graph Convolutional Networks (`GCN <https://arxiv.org/abs/1609.02907>`__)

    Parameters
    ----------
    in_features : int
        Dimension of input features.
    out_features : int
        Dimension of output features.
    hidden_features : int or list of int
        Dimension of hidden features. List if multi-layer.
    n_layers : int
        Number of layers.
    layer_norm : bool, optional
        Whether to use layer normalization. Default: ``False``.
    activation : func of torch.nn.functional, optional
        Activation function. Default: ``torch.nn.functional.relu``.
    residual : bool, optional
        Whether to use residual connection. Default: ``False``.
    feat_norm : str, optional
        Type of features normalization, choose from ["arctan", "tanh", None]. Default: ``None``.
    adj_norm_func : func of utils.normalize, optional
        Function that normalizes adjacency matrix. Default: ``GCNAdjNorm``.
    dropout : float, optional
        Dropout rate during training. Default: ``0.0``.

    """


    def __init__(self, config: GRANDConfig):
        super(GRAND, self).__init__()
        self.input_dim = config.input_dim
        self.output_dim = config.output_dim
        self.feat_norm = config.feat_norm

        n_layers = config.n_layers
        input_dim = config.input_dim
        output_dim = config.output_dim
        hidden_dim = config.hidden_dim
        activation = F.relu
        layer_norm = config.layer_norm
        residual = config.residual
        dropout = config.dropout
        self.adj_norm_func = GCNAdjNorm
        if type(hidden_dim) is int:
            hidden_dim = [hidden_dim] * (n_layers - 1)
        elif type(hidden_dim) is list or type(hidden_dim) is tuple:
            assert len(hidden_dim) == (n_layers - 1), "Incompatible sizes between hidden_features and n_layers."
        n_features = [input_dim] + hidden_dim + [output_dim]

        self.layers = nn.ModuleList()
        for i in range(n_layers):
            if layer_norm:
                self.layers.append(nn.LayerNorm(n_features[i]))
            self.layers.append(GCNConv(in_features=n_features[i],
                                       out_features=n_features[i + 1],
                                       activation=activation if i != n_layers - 1 else None,
                                       residual=residual if i != n_layers - 1 else False,
                                       dropout=dropout if i != n_layers - 1 else 0.0))
        self.reset_parameters()

    @property
    def model_type(self):
        """Indicate type of implementation."""
        return "torch"

    @property
    def model_name(self):
        return "gcn"

    def reset_parameters(self):
        """Reset parameters."""
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, x, edge_index, edge_weight=None):
        r"""

        Parameters
        ----------
        x : torch.Tensor
            Tensor of input features.
        edge_index : torch.LongTensor
            Graph connectivity in COO format with shape [2, num_edges].
        edge_weight : torch.Tensor, optional
            Edge weights. If None, all edges have weight 1. Default: ``None``.

        Returns
        -------
        x : torch.Tensor
            Output of model (logits without activation).

        """
        for layer in self.layers:
            if isinstance(layer, nn.LayerNorm):
                x = layer(x)
            else:
                x = layer(x, edge_index, edge_weight)

        return x


class GCNGC(nn.Module):
    r"""

    Description
    -----------
    Graph Convolutional Networks (`GCN <https://arxiv.org/abs/1609.02907>`__)

    Parameters
    ----------
    in_features : int
        Dimension of input features.
    out_features : int
        Dimension of output features.
    hidden_features : int or list of int
        Dimension of hidden features. List if multi-layer.
    n_layers : int
        Number of layers.
    layer_norm : bool, optional
        Whether to use layer normalization. Default: ``False``.
    activation : func of torch.nn.functional, optional
        Activation function. Default: ``torch.nn.functional.relu``.
    residual : bool, optional
        Whether to use residual connection. Default: ``False``.
    feat_norm : str, optional
        Type of features normalization, choose from ["arctan", "tanh", None]. Default: ``None``.
    adj_norm_func : func of utils.normalize, optional
        Function that normalizes adjacency matrix. Default: ``GCNAdjNorm``.
    dropout : float, optional
        Dropout rate during training. Default: ``0.0``.

    """

    def __init__(self,
                 in_features,
                 out_features,
                 hidden_features,
                 n_layers,
                 activation=F.relu,
                 layer_norm=False,
                 residual=False,
                 feat_norm=None,
                 adj_norm_func=GCNAdjNorm,
                 dropout=0.0):
        super(GCNGC, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.feat_norm = feat_norm
        self.adj_norm_func = adj_norm_func
        if type(hidden_features) is int:
            hidden_features = [hidden_features] * (n_layers - 1)
        elif type(hidden_features) is list or type(hidden_features) is tuple:
            assert len(hidden_features) == (n_layers - 1), "Incompatible sizes between hidden_features and n_layers."
        n_features = [in_features] + hidden_features

        self.layers = nn.ModuleList()
        for i in range(n_layers - 1):
            if layer_norm:
                self.layers.append(nn.LayerNorm(n_features[i]))
            self.layers.append(GCNConv(in_features=n_features[i],
                                       out_features=n_features[i + 1],
                                       activation=activation,
                                       residual=residual,
                                       dropout=dropout))
        self.linear = nn.Linear(hidden_features[-1], out_features)
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    @property
    def model_type(self):
        """Indicate type of implementation."""
        return "torch"

    @property
    def model_name(self):
        return "gcn"

    def reset_parameters(self):
        """Reset parameters."""
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, x, edge_index, edge_weight=None, batch_index=None):
        r"""

        Parameters
        ----------
        x : torch.Tensor
            Tensor of input features.
        edge_index : torch.LongTensor
            Graph connectivity in COO format with shape [2, num_edges].
        edge_weight : torch.Tensor, optional
            Edge weights. If None, all edges have weight 1. Default: ``None``.
        batch_index : torch.LongTensor, optional
            Batch index for graph-level tasks. Default: ``None``.

        Returns
        -------
        x : torch.Tensor
            Output of model (logits without activation).

        """
        for layer in self.layers:
            if isinstance(layer, nn.LayerNorm):
                x = layer(x)
            else:
                x = layer(x, edge_index, edge_weight)

        if batch_index is not None:
            batch_size = int(torch.max(batch_index)) + 1
            out = torch.zeros(batch_size, x.shape[1]).to(x.device)
            out = out.scatter_add_(dim=0, index=batch_index.view(-1, 1).repeat(1, x.shape[1]), src=x)
        else:
            out = torch.sum(x, dim=0)
        out = self.dropout(self.linear(out))

        return out


class GCNConv(nn.Module):
    r"""

    Description
    -----------
    GCN convolutional layer.

    Parameters
    ----------
    in_features : int
        Dimension of input features.
    out_features : int
        Dimension of output features.
    activation : func of torch.nn.functional, optional
        Activation function. Default: ``None``.
    residual : bool, optional
        Whether to use residual connection. Default: ``False``.
    dropout : float, optional
        Dropout rate during training. Default: ``0.0``.

    """

    def __init__(self,
                 in_features,
                 out_features,
                 activation=None,
                 residual=False,
                 dropout=0.0):
        super(GCNConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features)

        if residual:
            self.residual = nn.Linear(in_features, out_features)
        else:
            self.residual = None
        self.activation = activation

        if dropout > 0.0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

        self.reset_parameters()

    def reset_parameters(self):
        """Reset parameters."""
        if self.activation == F.leaky_relu:
            gain = nn.init.calculate_gain('leaky_relu')
        else:
            gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.linear.weight, gain=gain)

    def forward(self, x, edge_index, edge_weight=None):
        r"""

        Parameters
        ----------
        x : torch.Tensor
            Tensor of input features.
        edge_index : torch.LongTensor
            Graph connectivity in COO format with shape [2, num_edges].
        edge_weight : torch.Tensor, optional
            Edge weights. If None, all edges have weight 1. Default: ``None``.

        Returns
        -------
        x : torch.Tensor
            Output of layer.

        """
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

        # Apply GCN layer
        x = self.linear(x)
        x = torch.sparse.mm(adj, x)
        
        if self.activation is not None:
            x = self.activation(x)
        if self.residual is not None:
            x = x + self.residual(x)
        if self.dropout is not None:
            x = self.dropout(x)

        return x
