import os
import torch
import numpy as np
from typing import Optional, Dict, Any

from interfaces import TrainerProtocol, MetricsDict
from hooks.base_hook import BaseHook

class BestModelTracker(BaseHook):
    """Hook to track and save the best model during training.
    
    Monitors a specified metric and saves the model when it improves.
    """
    
    def __init__(
        self,
        monitor: str = "val_loss",
        mode: str = "min",
        save_dir: str = "checkpoints",
        filename: str = "best.pt",
        verbose: bool = True,
    ):
        """Initialize best model tracker hook.
        
        Args:
            monitor: Metric to monitor.
            mode: One of {'min', 'max'}. In 'min' mode, the best model is saved when the
                metric is minimized; in 'max' mode, the best model is saved when the
                metric is maximized.
            save_dir: Directory to save the best model.
            filename: Filename to save the best model.
            verbose: Whether to print a message when saving the best model.
        """
        self.monitor = monitor
        self.mode = mode
        self.save_dir = save_dir
        self.filename = filename
        self.verbose = verbose
        
        # Initialize state
        self.best_value = np.inf if mode == "min" else -np.inf
        self.best_epoch = 0
    
    def on_train_begin(self, trainer: TrainerProtocol) -> None:
        """Reset state at the beginning of training."""
        self.best_value = np.inf if self.mode == "min" else -np.inf
        self.best_epoch = 0
        
        # Create save directory if it doesn't exist
        os.makedirs(self.save_dir, exist_ok=True)
    
    def on_epoch_end(self, trainer: TrainerProtocol, epoch: int, metrics: MetricsDict) -> None:
        """Check for improvement at the end of each epoch."""
        # Get current value
        current = metrics.get(self.monitor)
        
        if current is None:
            return
        
        # Check if improved
        if self.mode == "min":
            improved = current < self.best_value
        else:
            improved = current > self.best_value
        
        if improved:
            # Save the new best model
            self.best_value = current
            self.best_epoch = epoch
            
            # Save checkpoint
            checkpoint_path = os.path.join(self.save_dir, self.filename)
            trainer.save_checkpoint(checkpoint_path)
            
            if self.verbose:
                print(f"Epoch {epoch}: {self.monitor} improved to {current:.6f}, saving model to {checkpoint_path}.")
    
    def on_train_end(self, trainer: TrainerProtocol) -> None:
        """Print best results at the end of training."""
        if self.verbose:
            print(f"Best {self.monitor}: {self.best_value:.6f} at epoch {self.best_epoch}") 