from typing import Dict, Any

from interfaces import HookProtocol, TrainerProtocol, MetricsDict, TensorOrFloat

class BaseHook(HookProtocol):
    """Base hook implementation."""
    
    def on_train_begin(self, trainer: TrainerProtocol) -> None:
        """Called at the beginning of training."""
        pass
    
    def on_train_end(self, trainer: TrainerProtocol) -> None:
        """Called at the end of training."""
        pass
    
    def on_epoch_begin(self, trainer: TrainerProtocol, epoch: int) -> None:
        """Called at the beginning of each epoch."""
        pass
    
    def on_epoch_end(self, trainer: TrainerProtocol, epoch: int, metrics: MetricsDict) -> None:
        """Called at the end of each epoch."""
        pass
    
    def on_batch_begin(self, trainer: TrainerProtocol, batch_idx: int) -> None:
        """Called at the beginning of each batch."""
        pass
    
    def on_batch_end(self, trainer: TrainerProtocol, batch_idx: int, loss: TensorOrFloat, metrics: MetricsDict) -> None:
        """Called at the end of each batch."""
        pass 