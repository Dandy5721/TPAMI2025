import os
import argparse
import json
import yaml
import numpy as np
from typing import Dict, Any, List, Optional
from collections import defaultdict

import torch
from torch.utils.data import DataLoader

from models.model_factory import ModelFactory
from losses.loss_factory import LossFactory
from optim.optim_factory import OptimFactory
from data.dataloader_factory import DataLoaderFactory
from trainers.trainer_factory import TrainerFactory
from hooks.early_stopping import EarlyStopping
from hooks.best_tracker import BestModelTracker
from hooks.logger import JsonLogger

# Register datasets for testing
from data.datasets.synthetic_dataset import SyntheticDataset
from data.datasets.oasis_dataset import OASISDataset
from data.datasets.tau_dataset import TauDataset
from data.datasets.surface_dataset import SurfaceDataset
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

def setup_factories(config: Dict[str, Any]):
    """Setup model, loss, optimizer, and trainer factories.
    
    Args:
        config: Configuration dictionary.
        
    Returns:
        Tuple of (model_factory, loss_factory, optim_factory, data_factory, trainer_factory).
    """
    # Create factories
    model_factory = ModelFactory()
    loss_factory = LossFactory()
    optim_factory = OptimFactory()
    data_factory = DataLoaderFactory()
    trainer_factory = TrainerFactory()
    
    # Register datasets
    data_factory.register_dataset("synthetic", SyntheticDataset)
    data_factory.register_dataset("oasis", OASISDataset)
    data_factory.register_dataset("tau", TauDataset)
    data_factory.register_dataset("surface", SurfaceDataset)
    
    return model_factory, loss_factory, optim_factory, data_factory, trainer_factory

def setup_hooks(config: Dict[str, Any], fold_idx: int) -> List[Any]:
    """Setup training hooks.
    
    Args:
        config: Configuration dictionary.
        fold_idx: Fold index for K-Fold cross validation.
        
    Returns:
        List of hooks.
    """
    hooks_config = config.get("hooks", {})
    exp_name = config.get("exp_name", "exp")
    hooks = []
    
    # Early stopping hook
    if hooks_config.get("early_stopping", {}).get("enabled", True):
        early_stopping_config = hooks_config.get("early_stopping", {})
        hooks.append(EarlyStopping(
            monitor=early_stopping_config.get("monitor", "val_loss"),
            min_delta=early_stopping_config.get("min_delta", 0.0),
            patience=early_stopping_config.get("patience", 10),
            mode=early_stopping_config.get("mode", "min"),
            verbose=early_stopping_config.get("verbose", True),
        ))
    
    # Model checkpoint hook
    if hooks_config.get("model_checkpoint", {}).get("enabled", True):
        checkpoint_config = hooks_config.get("model_checkpoint", {})
        save_dir = checkpoint_config.get("save_dir", "checkpoints")
        # Create fold-specific directory
        fold_dir = os.path.join(save_dir, exp_name, f"fold_{fold_idx}")
        os.makedirs(fold_dir, exist_ok=True)
        
        hooks.append(BestModelTracker(
            monitor=checkpoint_config.get("monitor", "val_loss"),
            mode=checkpoint_config.get("mode", "min"),
            save_dir=fold_dir,
            filename=checkpoint_config.get("filename", "best.pt"),
            verbose=checkpoint_config.get("verbose", True),
        ))
    
    # Logger hook
    # if hooks_config.get("logger", {}).get("enabled", True):
    #     logger_config = hooks_config.get("logger", {})
    #     log_dir = logger_config.get("log_dir", "logs")
    #     # Create fold-specific directory
    #     fold_log_dir = os.path.join(log_dir, exp_name, f"fold_{fold_idx}")
    #     os.makedirs(fold_log_dir, exist_ok=True)
        
    #     hooks.append(JsonLogger(
    #         log_dir=fold_log_dir,
    #         filename=logger_config.get("filename", "metrics"),
    #         log_batch=logger_config.get("log_batch", False),
    #         log_epoch=logger_config.get("log_epoch", True),
    #     ))
    
    return hooks

def train_fold(
    config: Dict[str, Any],
    fold_idx: int,
    model_factory: ModelFactory,
    loss_factory: LossFactory,
    optim_factory: OptimFactory,
    data_factory: DataLoaderFactory,
    trainer_factory: TrainerFactory,
    device: torch.device,
) -> Dict[str, Any]:
    """Train a model for a specific fold.
    
    Args:
        config: Configuration dictionary.
        fold_idx: Fold index for K-Fold cross validation.
        model_factory: Model factory.
        loss_factory: Loss factory.
        optim_factory: Optimizer factory.
        data_factory: Data loader factory.
        trainer_factory: Trainer factory.
        device: Device to train on.
        
    Returns:
        Training results.
    """
    print(f"\n--- Training Fold {fold_idx} ---")
    
    # Create data provider
    data_provider = data_factory.create(config)
    
    # Get train and validation datasets for this fold
    num_folds = config.get("train", {}).get("num_folds", 5)
    train_dataset, val_dataset = data_provider.get_fold(fold_idx, num_folds)
    
    # Create dataloaders
    batch_size = config.get("train", {}).get("batch_size", 32)
    num_workers = config.get("train", {}).get("num_workers", 4)
    train_loader, val_loader = data_provider.get_train_val_dataloader(
        train_dataset, val_dataset, batch_size, num_workers
    )
    
    # Create models, losses, optimizers, and schedulers
    models = {}
    losses = {}
    optimizers = {}
    schedulers = {}
    
    # Get trainer type to determine what components to create
    trainer_type = config.get("trainer", {}).get("type")
    
    if trainer_type == "mlp":
        # Create single model
        model_config = config.get("model", {})
        model = model_factory.create(model_config)
        models["model"] = model
        
        # Create loss function
        loss_config = config.get("loss", {})
        loss = loss_factory.create(loss_config)
        losses["loss"] = loss
        
        # Create optimizer and scheduler
        optim_config = config.get("optim", {})
        optimizer, scheduler = optim_factory.create(model.parameters(), optim_config)
        optimizers["optimizer"] = optimizer
        schedulers["scheduler"] = scheduler
        
    elif trainer_type == "mfg4ad":
        # Create generator
        gen_config = config.get("generator", {})
        generator = model_factory.create(gen_config)
        models["generator"] = generator
        
        # Create critic
        critic_config = config.get("critic", {})
        critic = model_factory.create(critic_config)
        models["critic"] = critic
        
        # Create loss functions
        gen_loss_config = config.get("generator_loss", {})
        critic_loss_config = config.get("critic_loss", {})
        gen_loss = loss_factory.create(gen_loss_config)
        critic_loss = loss_factory.create(critic_loss_config)
        losses["generator_loss"] = gen_loss
        losses["critic_loss"] = critic_loss
        
        # Create optimizers and schedulers
        gen_optim_config = config.get("generator_optim", {})
        critic_optim_config = config.get("critic_optim", {})
        gen_optimizer, gen_scheduler = optim_factory.create(generator.parameters(), gen_optim_config)
        critic_optimizer, critic_scheduler = optim_factory.create(critic.parameters(), critic_optim_config)
        optimizers["generator"] = gen_optimizer
        optimizers["critic"] = critic_optimizer
        schedulers["generator"] = gen_scheduler
        schedulers["critic"] = critic_scheduler

    elif trainer_type == "ode":
        # Create single model
        model_config = config.get("model", {})
        model = model_factory.create(model_config)
        models["model"] = model
        
        # Create loss function
        loss_config = config.get("loss", {})
        loss = loss_factory.create(loss_config)
        losses["loss"] = loss

        # Create optimizer
        optim_config = config.get("optim", {})
        optimizer, scheduler = optim_factory.create(model.parameters(), optim_config)
        optimizers["optimizer"] = optimizer
        schedulers["scheduler"] = scheduler            
    
    elif trainer_type == "gcn" or trainer_type == "deep_symbolic":
        # Create single model
        model_config = config.get("model", {})
        model = model_factory.create(model_config)
        models["model"] = model

        # Create loss function
        loss_config = config.get("loss", {})
        loss = loss_factory.create(loss_config)
        losses["loss"] = loss

        # Create optimizer and scheduler
        optim_config = config.get("optim", {})
        optimizer, scheduler = optim_factory.create(model.parameters(), optim_config)
        optimizers["optimizer"] = optimizer
        schedulers["scheduler"] = scheduler

    elif trainer_type == "ridge":
        # Create single model
        model_config = config.get("model", {})
        model = model_factory.create(model_config)
        models["model"] = model

    elif trainer_type == "raw_gan":
        # Create generator
        gen_config = config.get("generator", {})
        generator = model_factory.create(gen_config)
        models["generator"] = generator

        # Create critic
        critic_config = config.get("critic", {})
        critic = model_factory.create(critic_config)
        models["critic"] = critic

        # Create loss functions
        gen_loss_config = config.get("generator_loss", {})
        critic_loss_config = config.get("critic_loss", {})
        gen_loss = loss_factory.create(gen_loss_config)
        critic_loss = loss_factory.create(critic_loss_config)
        losses["generator_loss"] = gen_loss
        losses["critic_loss"] = critic_loss

        # Create optimizers and schedulers
        gen_optim_config = config.get("generator_optim", {})
        critic_optim_config = config.get("critic_optim", {})
        gen_optimizer, gen_scheduler = optim_factory.create(generator.parameters(), gen_optim_config)
        critic_optimizer, critic_scheduler = optim_factory.create(critic.parameters(), critic_optim_config)
        optimizers["generator"] = gen_optimizer
        optimizers["critic"] = critic_optimizer

        
    # Move models to device
    for model in models.values():
        if isinstance(model, torch.nn.Module):
            model.to(device)
    
    # Setup hooks
    hooks = setup_hooks(config, fold_idx)
    
    # Create trainer
    trainer = trainer_factory.create(
        config=config,
        models=models,
        losses=losses,
        optimizers=optimizers,
        schedulers=schedulers,
        train_loader=train_loader,
        val_loader=val_loader,
        hooks=hooks,
        device=device,
    )
    
    # Train model
    results = trainer.train(fold_idx)
    
    return results

def run_k_fold_training(config: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """Run K-Fold cross validation training.
    
    Args:
        config: Configuration dictionary.
        device: Device to train on.
        
    Returns:
        Training results for all folds.
    """
    # Setup factories
    model_factory, loss_factory, optim_factory, data_factory, trainer_factory = setup_factories(config)
    
    # Get number of folds
    num_folds = config.get("train", {}).get("num_folds", 5)
    
    # Train each fold
    fold_results = []
    for fold_idx in range(num_folds):
        fold_result = train_fold(
            config=config,
            fold_idx=fold_idx,
            model_factory=model_factory,
            loss_factory=loss_factory,
            optim_factory=optim_factory,
            data_factory=data_factory,
            trainer_factory=trainer_factory,
            device=device,
        )
        fold_results.append(fold_result)
    
    # Aggregate results
    return aggregate_fold_results(fold_results, config)

def aggregate_fold_results(fold_results: List[Dict[str, Any]], config: Dict[str, Any]) -> Dict[str, Any]:
    """Aggregate results from K-Fold cross validation.
    
    Args:
        fold_results: List of results for each fold.
        config: Configuration dictionary.
        
    Returns:
        Aggregated results.
    """
    print("\n--- Aggregating K-Fold Results ---")
    
    # Get experiment name
    exp_name = config.get("exp_name", "exp")
    
    # Collect metrics from all folds
    all_metrics = defaultdict(list)
    
    for fold_result in fold_results:
        metrics = fold_result["metrics"]
        for key, value in metrics.items():
            all_metrics[key].append(value)
    
    # Calculate mean and std for each metric
    aggregated_metrics = {}
    for key, values in all_metrics.items():
        aggregated_metrics[f"{key}_mean"] = float(np.mean(values))
        aggregated_metrics[f"{key}_std"] = float(np.std(values))
    
    # Create aggregated results
    aggregated_results = {
        "num_folds": len(fold_results),
        "fold_metrics": [result["metrics"] for result in fold_results],
        "aggregated_metrics": aggregated_metrics,
    }
    
    # Save aggregated results
    save_dir = config.get("hooks", {}).get("model_checkpoint", {}).get("save_dir", "checkpoints")
    save_path = os.path.join(save_dir, exp_name, "metrics_avg.json")
    print(f"Saving aggregated results to {save_path}")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, "w") as f:
        json.dump(aggregated_results, f, indent=2)
    
    print("\n--- K-Fold Results ---")
    for key, value in aggregated_metrics.items():
        print(f"{key}: {value:.4f}")
    
    return aggregated_results

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Deep Learning Training System")
    parser.add_argument("--config", type=str, required=True, help="Path to configuration file")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", 
                        help="Device to run on (cuda or cpu)")
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set device
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Run K-Fold training
    run_k_fold_training(config, device)

if __name__ == "__main__":
    main() 