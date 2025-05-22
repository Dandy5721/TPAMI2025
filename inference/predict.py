import os
import argparse
import yaml
import json
from typing import Dict, Any, List, Optional, Union, Tuple

import torch
import numpy as np

from models.model_factory import ModelFactory
from models.gan import Generator
from models.wgan import WGANGenerator

def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file.
    
    Args:
        config_path: Path to YAML configuration file.
        
    Returns:
        Configuration dictionary.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config

def load_model(
    model_config: Dict[str, Any], 
    checkpoint_path: str, 
    device: torch.device
) -> Union[torch.nn.Module, Dict[str, torch.nn.Module]]:
    """Load model from checkpoint.
    
    Args:
        model_config: Model configuration dictionary.
        checkpoint_path: Path to model checkpoint.
        device: Device to load model on.
        
    Returns:
        Loaded model(s).
    """
    # Create model factory
    model_factory = ModelFactory()
    
    # Get trainer type
    trainer_type = model_config.get("trainer", {}).get("type")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if trainer_type == "mlp":
        # Create model
        model = model_factory.create(model_config.get("model", {}))
        
        # Load model state
        model.load_state_dict(checkpoint["model_state_dict"])
        
        # Move model to device
        model = model.to(device)
        
        # Set model to evaluation mode
        model.eval()
        
        return model
    
    elif trainer_type == "wgan":
        # Create generator and critic
        generator = model_factory.create(model_config.get("generator", {}))
        critic = model_factory.create(model_config.get("critic", {}))
        
        # Load model states
        generator.load_state_dict(checkpoint["generator_state_dict"])
        critic.load_state_dict(checkpoint["critic_state_dict"])
        
        # Move models to device
        generator = generator.to(device)
        critic = critic.to(device)
        
        # Set models to evaluation mode
        generator.eval()
        critic.eval()
        
        return {"generator": generator, "critic": critic}
    
    else:
        raise ValueError(f"Unknown trainer type: {trainer_type}")

def predict_mlp(model: torch.nn.Module, inputs: torch.Tensor, task_type: str = "regression") -> torch.Tensor:
    """Make predictions with MLP model.
    
    Args:
        model: MLP model.
        inputs: Input tensor.
        task_type: Task type (regression or classification).
        
    Returns:
        Predictions.
    """
    with torch.no_grad():
        outputs = model(inputs)
        
        if task_type == "classification":
            if outputs.size(1) == 1:  # Binary classification
                outputs = (outputs >= 0.5).float()
            else:  # Multi-class classification
                outputs = torch.argmax(outputs, dim=1, keepdim=True)
    
    return outputs

def generate_samples(generator: Union[Generator, WGANGenerator], num_samples: int, latent_dim: int, device: torch.device) -> torch.Tensor:
    """Generate samples with GAN generator.
    
    Args:
        generator: GAN generator.
        num_samples: Number of samples to generate.
        latent_dim: Latent dimension.
        device: Device to generate samples on.
        
    Returns:
        Generated samples.
    """
    with torch.no_grad():
        # Sample random noise
        z = torch.randn(num_samples, latent_dim, device=device)
        
        # Generate samples
        fake_samples = generator(z)
    
    return fake_samples

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Model Inference")
    parser.add_argument("--config", type=str, required=True, help="Path to configuration file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", 
                        help="Device to run on (cuda or cpu)")
    parser.add_argument("--output", type=str, default="predictions.json", help="Path to output file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # GAN specific arguments
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to generate (for GANs)")
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Load configuration
    config = load_config(args.config)
    
    # Set device
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Load model
    model = load_model(config, args.checkpoint, device)
    
    # Get trainer type
    trainer_type = config.get("trainer", {}).get("type")
    
    # Make predictions or generate samples
    if trainer_type == "mlp":
        # Create dummy inputs for MLP
        input_dim = config.get("model", {}).get("input_dim", 10)
        inputs = torch.randn(args.num_samples, input_dim, device=device)
        
        # Get task type
        task_type = config.get("task", {}).get("type", "regression")
        
        # Make predictions
        predictions = predict_mlp(model, inputs, task_type)
        
        # Convert to numpy and save
        inputs_np = inputs.cpu().numpy().tolist()
        predictions_np = predictions.cpu().numpy().tolist()
        
        # Save predictions
        with open(args.output, "w") as f:
            json.dump({
                "inputs": inputs_np,
                "predictions": predictions_np,
            }, f, indent=2)
        
        print(f"Saved predictions to {args.output}")
    
    elif trainer_type == "wgan":
        generator = model["generator"]
        critic = model["critic"]
        
        # Get latent dimension
        latent_dim = config.get("wgan", {}).get("latent_dim", 100)
        
        # Generate samples
        samples = generate_samples(generator, args.num_samples, latent_dim, device)
        
        # Compute critic scores
        critic_scores = critic(samples)
        
        # Convert to numpy and save
        samples_np = samples.cpu().numpy().tolist()
        critic_scores_np = critic_scores.cpu().numpy().tolist()
        
        # Save samples and scores
        with open(args.output, "w") as f:
            json.dump({
                "samples": samples_np,
                "critic_scores": critic_scores_np,
            }, f, indent=2)
        
        print(f"Saved {args.num_samples} generated samples to {args.output}")

if __name__ == "__main__":
    main() 