import numpy as np
from typing import Optional

from interfaces import TrainerProtocol, MetricsDict
from hooks.base_hook import BaseHook

class EarlyStopping(BaseHook):
    """Early stopping hook to prevent overfitting.
    
    Monitors a specified metric and stops training if it doesn't improve for a
    specified number of epochs (patience).
    """
    
    def __init__(
        self,
        monitor: str = "val_loss",
        min_delta: float = 0.0,
        patience: int = 10,
        mode: str = "min",
        verbose: bool = True,
    ):
        """Initialize early stopping hook.
        
        Args:
            monitor: Metric to monitor.
            min_delta: Minimum change to qualify as improvement.
            patience: Number of epochs with no improvement after which to stop.
            mode: One of {'min', 'max'}. In 'min' mode, training stops when the
                metric stops decreasing; in 'max' mode, training stops when the
                metric stops increasing.
            verbose: Whether to print a message when stopping early.
        """
        self.monitor = monitor
        self.min_delta = min_delta
        self.patience = patience
        self.mode = mode
        self.verbose = verbose
        
        # Initialize state
        self.wait = 0
        self.stopped_epoch = 0
        self.best_value = np.inf if mode == "min" else -np.inf
        self.should_stop = False
    
    def on_train_begin(self, trainer: TrainerProtocol) -> None:
        """Reset state at the beginning of training."""
        self.wait = 0
        self.stopped_epoch = 0
        self.best_value = np.inf if self.mode == "min" else -np.inf
        self.should_stop = False
    
    def on_epoch_end(self, trainer: TrainerProtocol, epoch: int, metrics: MetricsDict) -> None:
        """Check for early stopping at the end of each epoch."""
        # Get current value
        current = metrics.get(self.monitor)
        
        if current is None:
            return
        
        # Check if improved
        if self.mode == "min":
            improved = current < (self.best_value - self.min_delta)
        else:
            improved = current > (self.best_value + self.min_delta)
        
        if improved:
            # Reset counter if improved
            self.best_value = current
            self.wait = 0
        else:
            # Increment counter if not improved
            self.wait += 1
            if self.wait >= self.patience:
                self.should_stop = True
                self.stopped_epoch = epoch
                
                if self.verbose:
                    direction = "decrease" if self.mode == "min" else "increase"
                    print(f"Early stopping: No {direction} in {self.monitor} for {self.wait} epochs.")
                    print(f"Best {self.monitor}: {self.best_value:.6f}")
    
    def on_train_end(self, trainer: TrainerProtocol) -> None:
        """Print message at the end of training if stopped early."""
        if self.stopped_epoch > 0 and self.verbose:
            print(f"Epoch {self.stopped_epoch}: early stopping activated.") 