import os
import yaml
import torch
from models.mlp import MLP, MLPConfig
from data.datasets.tau_dataset import TauDataset
from torch.utils.data import DataLoader

def test_mlp_forward():
    """Test MLP forward pass with tau dataset."""
    # Load config
    with open("configs/mlp_tau.yaml", "r") as f:
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
    input_dim = sample["input_tau"].shape[1]
    output_dim = sample["output_tau"].shape[1]
    
    # Create MLP model
    mlp_config = MLPConfig(
        input_dim=config["model"]["input_dim"],
        hidden_dims=config["model"]["hidden_dims"],
        output_dim=config["model"]["output_dim"],
        activation=config["model"]["activation"],
        dropout=config["model"]["dropout"],
        batch_norm=config["model"]["batch_norm"]
    )

    print(mlp_config)

    model = MLP(mlp_config)
    
    # Test forward pass
    print("\nTesting MLP forward pass:")
    print(f"Input shape: {sample['input_tau'].shape}")

    output = model(sample["input_tau"])
    print(f"Output shape: {output.shape}")
    print(f"Output values: {output[0][:5]}")  # Print first 5 values of first sample

if __name__ == "__main__":
    test_mlp_forward() 