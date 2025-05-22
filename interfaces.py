from typing import Protocol, Dict, Any, List, Tuple, Union, Optional, TypeVar, Callable
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

# Type aliases for clarity
ConfigDict = Dict[str, Any]
Tensor = torch.Tensor
TensorOrFloat = Union[Tensor, float]
BatchType = Tuple[Tensor, Tensor]  # (inputs, targets)
MetricsDict = Dict[str, float]

# Generic type for models
T = TypeVar('T', bound='ModelProtocol')

class ModelProtocol(Protocol):
    """Protocol defining a model interface."""
    
    def forward(self, *args, **kwargs) -> Any:
        """Forward pass of the model."""
        ...
    
    def parameters(self) -> Any:
        """Return model parameters."""
        ...
        
    def train(self, mode: bool = True) -> T:
        """Set model to training mode."""
        ...
        
    def eval(self) -> T:
        """Set model to evaluation mode."""
        ...
        
    def to(self, device: torch.device) -> T:
        """Move model to device."""
        ...

class LossProtocol(Protocol):
    """Protocol defining a loss function."""
    
    def __call__(self, *args, **kwargs) -> Tensor:
        """Calculate loss."""
        ...

class OptimizerProtocol(Protocol):
    """Protocol defining an optimizer."""
    
    def zero_grad(self) -> None:
        """Zero gradients."""
        ...
        
    def step(self) -> None:
        """Update weights."""
        ...
        
    def state_dict(self) -> Dict[str, Any]:
        """Return state dict."""
        ...
        
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load state dict."""
        ...

class SchedulerProtocol(Protocol):
    """Protocol defining a learning rate scheduler."""
    
    def step(self, metrics: Optional[float] = None) -> None:
        """Update learning rate."""
        ...
        
    def state_dict(self) -> Dict[str, Any]:
        """Return state dict."""
        ...
        
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load state dict."""
        ...

class DataLoaderProtocol(Protocol):
    """Protocol defining a data loader."""
    
    def __iter__(self) -> Any:
        """Return iterator."""
        ...
        
    def __len__(self) -> int:
        """Return number of batches."""
        ...

class DatasetProviderProtocol(Protocol):
    """Protocol for dataset provider that supports K-Fold."""
    
    def get_fold(self, fold_idx: int, num_folds: int) -> Tuple[Dataset, Dataset]:
        """Get train and validation datasets for a specific fold."""
        ...
        
    def get_train_val_dataloader(self, train_dataset: Dataset, val_dataset: Dataset, 
                               batch_size: int, num_workers: int) -> Tuple[DataLoader, DataLoader]:
        """Get train and validation dataloaders."""
        ...

class HookProtocol(Protocol):
    """Protocol defining a training hook."""
    
    def on_train_begin(self, trainer: 'TrainerProtocol') -> None:
        """Called at the beginning of training."""
        ...
        
    def on_train_end(self, trainer: 'TrainerProtocol') -> None:
        """Called at the end of training."""
        ...
        
    def on_epoch_begin(self, trainer: 'TrainerProtocol', epoch: int) -> None:
        """Called at the beginning of each epoch."""
        ...
        
    def on_epoch_end(self, trainer: 'TrainerProtocol', epoch: int, metrics: MetricsDict) -> None:
        """Called at the end of each epoch."""
        ...
        
    def on_batch_begin(self, trainer: 'TrainerProtocol', batch_idx: int) -> None:
        """Called at the beginning of each batch."""
        ...
        
    def on_batch_end(self, trainer: 'TrainerProtocol', batch_idx: int, loss: TensorOrFloat, metrics: MetricsDict) -> None:
        """Called at the end of each batch."""
        ...

class LoggerProtocol(Protocol):
    """Protocol defining a logger."""
    
    def log_metrics(self, metrics: MetricsDict, step: int) -> None:
        """Log metrics."""
        ...
        
    def log_hyperparams(self, params: Dict[str, Any]) -> None:
        """Log hyperparameters."""
        ...
        
    def save_to_json(self, path: str) -> None:
        """Save logs to JSON file."""
        ...

class TrainerProtocol(Protocol):
    """Protocol defining a trainer."""
    
    def train(self, fold_idx: int) -> Dict[str, Any]:
        """Train model."""
        ...
        
    def validate(self) -> MetricsDict:
        """Validate model."""
        ...
        
    def save_checkpoint(self, path: str) -> None:
        """Save checkpoint."""
        ...
        
    def load_checkpoint(self, path: str) -> None:
        """Load checkpoint."""
        ...
        
    @property
    def current_metrics(self) -> MetricsDict:
        """Get current metrics."""
        ...

class ModelFactoryProtocol(Protocol):
    """Protocol defining a model factory."""
    
    def create(self, config: ConfigDict) -> ModelProtocol:
        """Create model from config."""
        ...

class LossFactoryProtocol(Protocol):
    """Protocol defining a loss factory."""
    
    def create(self, config: ConfigDict) -> LossProtocol:
        """Create loss from config."""
        ...

class OptimizerFactoryProtocol(Protocol):
    """Protocol defining an optimizer factory."""
    
    def create(self, model_parameters: Any, config: ConfigDict) -> Tuple[Optimizer, Optional[_LRScheduler]]:
        """Create optimizer and scheduler from config."""
        ...

class DataLoaderFactoryProtocol(Protocol):
    """Protocol defining a data loader factory."""
    
    def create(self, config: ConfigDict) -> DatasetProviderProtocol:
        """Create dataset provider from config."""
        ...

class TrainerFactoryProtocol(Protocol):
    """Protocol defining a trainer factory."""
    
    def create(self, config: ConfigDict, models: Dict[str, ModelProtocol], 
              losses: Dict[str, LossProtocol], 
              optimizers: Dict[str, Optimizer],
              schedulers: Dict[str, Optional[_LRScheduler]],
              train_loader: DataLoader, val_loader: DataLoader,
              hooks: List[HookProtocol], device: torch.device) -> TrainerProtocol:
        """Create trainer from config."""
        ... 