from typing import Dict, Any, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from interfaces import ModelProtocol, HookProtocol, MetricsDict, ConfigDict
from trainers.base_trainer import BaseTrainer

class ODETrainer(BaseTrainer):
    """Trainer for ODE models.
    
    This trainer handles the specific requirements of ODE training, including:
    - Integration of ODEs over time steps
    - Handling of initial conditions
    - Multiple evaluation points
    - ODE-specific loss functions
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
        """Initialize ODE trainer.
        
        Args:
            config: Configuration dict.
            model: ODE model to train.
            loss_fn: Loss function.
            optimizer: Optimizer.
            scheduler: Learning rate scheduler.
            train_loader: Training data loader.
            val_loader: Validation data loader.
            hooks: List of hooks.
            device: Device to train on.
        """
        super().__init__(
            config=config,
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            train_loader=train_loader,
            val_loader=val_loader,
            hooks=hooks,
            device=device,
        )
        
        # Get ODE specific parameters from config
        # ode_config = config.get("ode", {})
        # self.time_steps = ode_config.get("time_steps", 100)
        # self.solver = ode_config.get("solver", "dopri5")  # Default to dopri5 solver
        # self.rtol = ode_config.get("rtol", 1e-7)
        # self.atol = ode_config.get("atol", 1e-9)
        
        # Override the metrics dictionary
        self._metrics = {
            "train_loss": 0.0,
            "train_mae": 0.0,
            "train_rmse": 0.0,
            "val_loss": 0.0,
            "val_mae": 0.0,
            "val_rmse": 0.0,
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
            input_tau = batch['input_tau'].squeeze(0).unsqueeze(-1).to(self.device)
            adj = batch['structural'].squeeze(0).to(self.device)
            targets = batch['output_tau'].squeeze(0).unsqueeze(-1).to(self.device)

            t = torch.tensor([0, 1], dtype=torch.float32)
            
            # Forward pass through ODE solver
            predictions = self.model(
                input_tau,
                t,
                adj
            )
            
            # Compute loss
            loss = self.loss_fn(predictions, targets)
            
            # Backward pass and optimization
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            running_loss += loss.item()
            all_predictions.append(predictions.detach())
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
                input_tau = batch['input_tau'].squeeze(0).unsqueeze(-1).to(self.device)
                adj = batch['structural'].squeeze(0).to(self.device)
                targets = batch['output_tau'].squeeze(0).unsqueeze(-1).to(self.device)

                t = torch.tensor([0, 1], dtype=torch.float32)
                
                # Forward pass through ODE solver
                predictions = self.model(
                    input_tau,
                    t,
                    adj
                )
                
                # Compute loss
                loss = self.loss_fn(predictions, targets)
                
                # Update metrics
                val_loss += loss.item()
                all_predictions.append(predictions)
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