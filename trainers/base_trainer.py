import os
import time
from typing import Dict, Any, List, Tuple, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from interfaces import TrainerProtocol, ModelProtocol, HookProtocol, MetricsDict, TensorOrFloat, ConfigDict

class BaseTrainer(TrainerProtocol):
    """Base trainer implementation.
    
    This class provides the basic functionality for training models, such as:
    - Training loop (epochs, batches)
    - Validation
    - Checkpoint saving and loading
    - Hooks for customization
    """
    
    def __init__(
        self,
        config: ConfigDict,
        model: ModelProtocol,
        loss_fn: nn.Module,
        optimizer: Optimizer,
        scheduler: Optional[_LRScheduler],
        train_loader: DataLoader,
        val_loader: DataLoader,
        hooks: List[HookProtocol],
        device: torch.device,
    ):
        """Initialize base trainer.
        
        Args:
            config: Configuration dict.
            model: Model to train.
            loss_fn: Loss function.
            optimizer: Optimizer.
            scheduler: Learning rate scheduler.
            train_loader: Training data loader.
            val_loader: Validation data loader.
            hooks: List of hooks.
            device: Device to train on.
        """
        self.config = config
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.hooks = hooks
        self.device = device
        
        # Get training parameters from config
        self.num_epochs = config.get("trainer", {}).get("epochs", 100)
        self.log_interval = config.get("trainer", {}).get("log_interval", 10)
        
        # Initialize metrics
        self._metrics = {}
    
    def train(self, fold_idx: int) -> Dict[str, Any]:
        """Train model.
        
        Args:
            fold_idx: Fold index for K-Fold cross validation.
            
        Returns:
            Training results.
        """
        # Call hooks for training begin
        for hook in self.hooks:
            hook.on_train_begin(self)
        
        # Initialize best validation loss for early stopping
        best_val_loss = float('inf')
        
        # Training loop
        for epoch in range(1, self.num_epochs + 1):
            # Call hooks for epoch begin
            for hook in self.hooks:
                hook.on_epoch_begin(self, epoch)
            
            # Train one epoch
            train_metrics = self._train_epoch(epoch)
            
            # Validate
            val_metrics = self.validate()
            
            # Update learning rate
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics.get("val_loss", 0))
                else:
                    self.scheduler.step()
            
            # Combine metrics
            epoch_metrics = {**train_metrics, **val_metrics}
            
            # Update metrics
            self._metrics = epoch_metrics
            
            # Call hooks for epoch end
            for hook in self.hooks:
                hook.on_epoch_end(self, epoch, epoch_metrics)
            
            # Print progress
            if epoch % self.log_interval == 0:
                metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in epoch_metrics.items()])
                print(f"Fold {fold_idx}, Epoch {epoch}/{self.num_epochs} - {metrics_str}")
            
            # Check for early stopping
            early_stop = any(getattr(hook, "should_stop", False) for hook in self.hooks)
            if early_stop:
                print(f"Early stopping at epoch {epoch}")
                break
        
        # Save final model
        self.save_checkpoint(os.path.join("checkpoints", f"fold_{fold_idx}_last.pt"))
        
        # Call hooks for training end
        for hook in self.hooks:
            hook.on_train_end(self)
        
        # Return results
        return {
            "fold": fold_idx,
            "epochs": epoch,
            "metrics": self._metrics,
        }
    
    def _train_epoch(self, epoch: int) -> MetricsDict:
        """Train one epoch.
        
        Args:
            epoch: Current epoch.
            
        Returns:
            Training metrics.
        """
        # Set model to training mode
        self.model.train()
        
        # Initialize metrics
        running_loss = 0.0
        all_predictions = []
        all_targets = []
        
        # Loop over batches
        for batch_idx, batch in enumerate(self.train_loader):
            # Call hooks for batch begin
            for hook in self.hooks:
                hook.on_batch_begin(self, batch_idx)
            
            # Move data to device
            inputs = batch['input_tau'].to(self.device)
            targets = batch['output_tau'].to(self.device)
            
            # Forward pass
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, targets)
            
            # Backward pass and optimization
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            running_loss += loss.item()
            all_predictions.append(outputs.detach())
            all_targets.append(targets)
            
            batch_metrics = {"train_loss": loss.item()}
            
            # Call hooks for batch end
            for hook in self.hooks:
                hook.on_batch_end(self, batch_idx, loss, batch_metrics)
        
        # Compute average loss
        avg_loss = running_loss / len(self.train_loader)
        
        # Calculate MAE and RMSE
        predictions = torch.cat(all_predictions, dim=0)
        targets = torch.cat(all_targets, dim=0)
        mae = torch.abs(predictions - targets).mean().item()
        rmse = torch.sqrt(((predictions - targets) ** 2).mean()).item()
        
        # Return metrics
        return {
            "train_loss": avg_loss,
            "train_mae": mae,
            "train_rmse": rmse,
        }
    
    def validate(self) -> MetricsDict:
        """Validate model.
        
        Returns:
            Validation metrics.
        """
        # Set model to evaluation mode
        self.model.eval()
        
        # Initialize metrics
        val_loss = 0.0
        all_predictions = []
        all_targets = []
        
        # No gradient computation during validation
        with torch.no_grad():
            for batch in self.val_loader:
                # Move data to device
                inputs = batch['input_tau'].to(self.device)
                targets = batch['output_tau'].to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)
                
                # Update metrics
                val_loss += loss.item()
                all_predictions.append(outputs)
                all_targets.append(targets)
        
        # Compute average loss
        avg_loss = val_loss / len(self.val_loader)
        
        # Calculate MAE and RMSE
        predictions = torch.cat(all_predictions, dim=0)
        targets = torch.cat(all_targets, dim=0)
        mae = torch.abs(predictions - targets).mean().item()
        rmse = torch.sqrt(((predictions - targets) ** 2).mean()).item()
        
        # Return metrics
        return {
            "val_loss": avg_loss,
            "val_mae": mae,
            "val_rmse": rmse,
        }
    
    def save_checkpoint(self, path: str) -> None:
        """Save checkpoint.
        
        Args:
            path: Path to save checkpoint.
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Create checkpoint
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": self._metrics,
        }
        
        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
        
        # Save checkpoint
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str) -> None:
        """Load checkpoint.
        
        Args:
            path: Path to load checkpoint from.
        """
        # Load checkpoint
        checkpoint = torch.load(path, map_location=self.device)
        
        # Load model state
        self.model.load_state_dict(checkpoint["model_state_dict"])
        
        # Load optimizer state
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        # Load scheduler state
        if "scheduler_state_dict" in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        # Load metrics
        self._metrics = checkpoint.get("metrics", {})
    
    @property
    def current_metrics(self) -> MetricsDict:
        """Get current metrics."""
        return self._metrics 