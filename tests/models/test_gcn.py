import os
import yaml
import torch
from models.gcn import GCN, GCNConfig
from data.datasets.tau_dataset import TauDataset
from torch.utils.data import DataLoader

def test_gcn_forward():
    """Test GCN forward pass with tau dataset."""
    # Load config
    with open("configs/gcn_oasis_no_concat.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # Create dataset
    dataset = TauDataset(
        data_dir=config["dataset"]["params"]["data_dir"],
        preload=config["dataset"]["params"]["preload"]
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=config["train"]["batch_size"],
        shuffle=True,
        num_workers=config["train"]["num_workers"]
    )
    
    # Get a sample to determine input dimensions
    sample = next(iter(dataloader))
    
    # Create GCN model
    gcn_config = GCNConfig(
        input_dim=config["model"]["input_dim"],
        hidden_dim=config["model"]["hidden_dim"],
        output_dim=config["model"]["output_dim"],
    )

    print(gcn_config)



    model = GCN(gcn_config)
    
    # Test forward pass
    print("\nTesting GCN forward pass:")
    print(f"Input shape: {sample['input_tau'].shape}")

    input_tau = sample["input_tau"].squeeze(0).unsqueeze(-1)
    adj  = sample["structural"].squeeze(0)
    edge_index = adj.nonzero().t()
    edge_weight = adj[edge_index[0], edge_index[1]]

    print(input_tau.shape)
    print(edge_index.shape)
    print(edge_weight.shape)

    output = model(input_tau, edge_index, edge_weight)
    print(f"Output shape: {output.shape}")
    print(f"Output values: {output[0][:5]}")  # Print first 5 values of first sample

if __name__ == "__main__":
    test_gcn_forward() 