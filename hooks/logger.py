import os
import json
import time
from typing import Dict, Any, Optional, List

from interfaces import TrainerProtocol, MetricsDict, TensorOrFloat
from hooks.base_hook import BaseHook

class JsonLogger(BaseHook):
    """Hook to log metrics to a JSON file."""
    
    def __init__(
        self,
        log_dir: str = "logs",
        filename: Optional[str] = None,
        log_batch: bool = False,
        log_epoch: bool = True,
    ):
        """Initialize JSON logger hook.
        
        Args:
            log_dir: Directory to save logs.
            filename: Filename to save logs (without extension).
                If None, a timestamp-based filename will be used.
            log_batch: Whether to log batch-level metrics.
            log_epoch: Whether to log epoch-level metrics.
        """
        self.log_dir = log_dir
        self.filename = filename or f"run_{int(time.time())}"
        self.log_batch = log_batch
        self.log_epoch = log_epoch
        
        # Initialize state
        self.logs = {
            "hyperparams": {},
            "epochs": [],
            "batches": [],
        }
        self.start_time = None
    
    def on_train_begin(self, trainer: TrainerProtocol) -> None:
        """Initialize logger at the beginning of training."""
        # Create log directory if it doesn't exist
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Reset logs
        self.logs = {
            "hyperparams": {},
            "epochs": [],
            "batches": [],
        }
        
        # Record start time
        self.start_time = time.time()
    
    def on_train_end(self, trainer: TrainerProtocol) -> None:
        """Save logs at the end of training."""
        # Add total training time
        self.logs["total_time"] = time.time() - self.start_time
        
        # Save logs to JSON file
        self.save_to_json()
    
    def on_epoch_begin(self, trainer: TrainerProtocol, epoch: int) -> None:
        """Initialize epoch logs at the beginning of each epoch."""
        if self.log_epoch:
            # Record epoch start time
            self._epoch_start_time = time.time()
    
    def on_epoch_end(self, trainer: TrainerProtocol, epoch: int, metrics: MetricsDict) -> None:
        """Log metrics at the end of each epoch."""
        if self.log_epoch:
            # Add epoch time
            epoch_time = time.time() - self._epoch_start_time
            
            # Create epoch log
            epoch_log = {
                "epoch": epoch,
                "time": epoch_time,
                **{k: self._convert_to_primitive(v) for k, v in metrics.items()}
            }
            
            # Add to logs
            self.logs["epochs"].append(epoch_log)
            
            # Save logs to JSON file (periodic update)
            self.save_to_json()
    
    def on_batch_end(self, trainer: TrainerProtocol, batch_idx: int, loss: TensorOrFloat, metrics: MetricsDict) -> None:
        """Log metrics at the end of each batch."""
        if self.log_batch:
            # Create batch log
            batch_log = {
                "batch": batch_idx,
                "loss": self._convert_to_primitive(loss),
                **{k: self._convert_to_primitive(v) for k, v in metrics.items()}
            }
            
            # Add to logs
            self.logs["batches"].append(batch_log)
    
    def log_hyperparams(self, params: Dict[str, Any]) -> None:
        """Log hyperparameters."""
        # Convert tensors to primitives
        processed_params = {k: self._convert_to_primitive(v) for k, v in params.items()}
        
        # Update hyperparams
        self.logs["hyperparams"].update(processed_params)
    
    def log_metrics(self, metrics: MetricsDict, step: int) -> None:
        """Log custom metrics."""
        # Create metric log
        metric_log = {
            "step": step,
            **{k: self._convert_to_primitive(v) for k, v in metrics.items()}
        }
        
        # Add to custom metrics
        if "custom_metrics" not in self.logs:
            self.logs["custom_metrics"] = []
        
        self.logs["custom_metrics"].append(metric_log)
    
    def save_to_json(self, path: Optional[str] = None) -> None:
        """Save logs to JSON file."""
        if path is None:
            path = os.path.join(self.log_dir, f"{self.filename}.json")
        
        with open(path, "w") as f:
            json.dump(self.logs, f, indent=2)
    
    def _convert_to_primitive(self, value: Any) -> Any:
        """Convert value to JSON-serializable primitive."""
        if hasattr(value, "item"):
            return value.item()
        return value