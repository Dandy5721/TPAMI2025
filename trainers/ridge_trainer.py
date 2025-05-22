from typing import Dict, Any, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from interfaces import ModelProtocol, HookProtocol, MetricsDict, ConfigDict, LossProtocol
from trainers.base_trainer import BaseTrainer


class RidgeTrainer(BaseTrainer):
    """Trainer for Ridge regression models.
    
    This trainer handles the specific requirements of Ridge regression training, including:
    - L2 regularization
    - Direct solution computation
    - No need for gradient descent
    """
    
    def __init__(
        self,
        config: ConfigDict,
        model: ModelProtocol,
        train_loader: DataLoader,
        val_loader: DataLoader,
        hooks: List[HookProtocol],
        device: torch.device,
        loss_fn: LossProtocol,
        optimizer: Optimizer,
        scheduler: _LRScheduler,
    ):
        """Initialize Ridge trainer.
        
        Args:
            config: Configuration dict.
            model: Model to train.
            loss_fn: Loss function.
            optimizer: Optimizer (not used in Ridge regression).
            scheduler: Learning rate scheduler (not used in Ridge regression).
            train_loader: Training data loader.
            val_loader: Validation data loader.
            hooks: List of hooks.
            device: Device to train on.
        """
        super().__init__(
            config=config,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            hooks=hooks,
            device=device,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        
        # Get Ridge specific parameters from config
        ridge_config = config.get("ridge", {})
        self.alpha = ridge_config.get("alpha", 1.0)  # Regularization strength
        
        # Override the metrics dictionary
        self._metrics = {
            "train_loss": 0.0,
            "train_mae": 0.0,
            "train_rmse": 0.0,
            "val_loss": 0.0,
            "val_mae": 0.0,
            "val_rmse": 0.0,
        }
    
    def train(self, fold_idx: int) -> Dict[str, Any]:
        """Train Ridge regression model using batch processing.
        
        Args:
            fold_idx: Fold index for K-Fold cross validation.
            
        Returns:
            Dictionary containing fold index and training metrics.
        """
        # Call hooks for training begin
        for hook in self.hooks:
            hook.on_train_begin(self)
        
        all_predictions = []
        all_targets = []
        
        # Process each batch
        for batch in self.train_loader:
            inputs = batch['input_tau'].to(self.device)
            targets = batch['output_tau'].to(self.device)
            
            # Convert to numpy for scikit-learn
            inputs_np = inputs.cpu().numpy()
            targets_np = targets.cpu().numpy()
            
            # Get predictions for this batch
            predictions_np = self.model.forward(inputs_np, targets_np)
            predictions = torch.from_numpy(predictions_np).to(self.device)
            
            # Store predictions and targets
            all_predictions.append(predictions)
            all_targets.append(targets)
        
        # Concatenate all predictions and targets
        predictions = torch.cat(all_predictions, dim=0)
        targets = torch.cat(all_targets, dim=0)
        
        # Compute training metrics
        mae = torch.abs(predictions - targets).mean().item()
        rmse = torch.sqrt(((predictions - targets) ** 2).mean()).item()
        
        # Update metrics
        self._metrics = {
            "train_mae": mae,
            "train_rmse": rmse,
        }

        # test eval
        val_metrics = self.eval()
        self._metrics.update(val_metrics)
        
        # Call hooks for training end
        for hook in self.hooks:
            hook.on_train_end(self)
        
        # Return results
        return {
            "fold": fold_idx,
            "metrics": self._metrics,
        }
    
    def eval(self) -> MetricsDict:
        """Validate model.
        
        Returns:
            Validation metrics.
        """
        all_predictions = []
        all_targets = []
        
        # No gradient computation during validation
        with torch.no_grad():
            for batch in self.val_loader:
                # Move data to device
                inputs = batch['input_tau'].to(self.device)
                targets = batch['output_tau'].to(self.device)
                
                # Convert to numpy for scikit-learn
                inputs_np = inputs.cpu().numpy()
                
                # Get predictions and convert back to tensor
                predictions_np = self.model.eval(inputs_np)
                predictions = torch.from_numpy(predictions_np).to(self.device)
                
                # Update metrics
                all_predictions.append(predictions)
                all_targets.append(targets)
        
        # Calculate MAE and RMSE
        predictions = torch.cat(all_predictions, dim=0)
        targets = torch.cat(all_targets, dim=0)
        mae = torch.abs(predictions - targets).mean().item()
        rmse = torch.sqrt(((predictions - targets) ** 2).mean()).item()
        
        # Return metrics
        return {
            "val_mae": mae,
            "val_rmse": rmse,
        }