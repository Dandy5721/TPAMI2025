import os
import yaml
import torch
from models.ode import GraphNeuralODE, ODEConfig
from data.datasets.tau_dataset import TauDataset
from torch.utils.data import DataLoader

def test_ode_forward():
    """Test ODE forward pass with tau dataset."""
    # Load config
    with open("configs/ode_tau_no_concat.yaml", "r") as f:
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
    
    # Create MLP model
    ode_config = ODEConfig(
        input_dim=config["model"]["input_dim"],
        hidden_dim=config["model"]["hidden_dim"],
        output_dim=config["model"]["output_dim"],
    )

    print(ode_config)



    model = GraphNeuralODE(ode_config)
    
    # Test forward pass
    print("\nTesting ODE forward pass:")
    print(f"Input shape: {sample['input_tau'].shape}")

    input_tau = sample["input_tau"].squeeze(0).unsqueeze(-1)
    adj  = sample["structural"].squeeze(0)
    t = torch.tensor([0, 1], dtype=torch.float32)


    output = model(input_tau, t, adj)
    print(f"Output shape: {output.shape}")
    print(f"Output values: {output[0][:5]}")  # Print first 5 values of first sample

if __name__ == "__main__":
    test_ode_forward() 