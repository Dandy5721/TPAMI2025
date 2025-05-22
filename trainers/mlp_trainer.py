from typing import Dict, Any, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from interfaces import ModelProtocol, HookProtocol, MetricsDict, ConfigDict
from trainers.base_trainer import BaseTrainer

class MLPTrainer(BaseTrainer):
    """Trainer for MLP models."""
    
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
        """Initialize MLP trainer.
        
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
        