from typing import Dict, Any, Tuple, Optional, Iterator

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import StepLR, MultiStepLR, ReduceLROnPlateau, CosineAnnealingLR

from interfaces import ConfigDict

class OptimFactory:
    """Factory for creating optimizers and schedulers from config."""
    
    def __init__(self):
        self._optimizer_registry = {
            "adam": self._create_adam,
            "sgd": self._create_sgd,
            "rmsprop": self._create_rmsprop,
            "adamw": self._create_adamw,
        }
        
        self._scheduler_registry = {
            "step_lr": self._create_step_lr,
            "multi_step_lr": self._create_multi_step_lr,
            "reduce_on_plateau": self._create_reduce_on_plateau,
            "cosine_annealing_lr": self._create_cosine_annealing_lr,
            "none": self._create_none_scheduler,
        }
    
    def create(self, model_parameters: Iterator[nn.Parameter], config: ConfigDict) -> Tuple[Optimizer, Optional[_LRScheduler]]:
        """Create optimizer and scheduler from config.
        
        Args:
            model_parameters: Model parameters.
            config: Configuration dict.
                
        Returns:
            Tuple of (optimizer, scheduler).
        """
        # Create optimizer
        optim_config = config.get("optimizer", {})
        optim_type = optim_config.get("type", "adam")
        
        if optim_type not in self._optimizer_registry:
            raise ValueError(f"Unknown optimizer type: {optim_type}")
            
        optimizer = self._optimizer_registry[optim_type](model_parameters, optim_config)
        
        # Create scheduler
        scheduler_config = config.get("scheduler", {"type": "none"})
        scheduler_type = scheduler_config.get("type", "none")
        
        if scheduler_type not in self._scheduler_registry:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")
            
        scheduler = self._scheduler_registry[scheduler_type](optimizer, scheduler_config)
        
        return optimizer, scheduler
    
    def _create_adam(self, parameters: Iterator[nn.Parameter], config: Dict[str, Any]) -> optim.Adam:
        """Create Adam optimizer."""
        return optim.Adam(
            parameters,
            lr=config.get("lr", 1e-3),
            betas=(config.get("beta1", 0.9), config.get("beta2", 0.999)),
            eps=config.get("eps", 1e-8),
            weight_decay=config.get("weight_decay", 0),
            amsgrad=config.get("amsgrad", False),
        )
    
    def _create_sgd(self, parameters: Iterator[nn.Parameter], config: Dict[str, Any]) -> optim.SGD:
        """Create SGD optimizer."""
        return optim.SGD(
            parameters,
            lr=config.get("lr", 1e-3),
            momentum=config.get("momentum", 0.0),
            dampening=config.get("dampening", 0.0),
            weight_decay=config.get("weight_decay", 0.0),
            nesterov=config.get("nesterov", False),
        )
    
    def _create_rmsprop(self, parameters: Iterator[nn.Parameter], config: Dict[str, Any]) -> optim.RMSprop:
        """Create RMSprop optimizer."""
        return optim.RMSprop(
            parameters,
            lr=config.get("lr", 1e-3),
            alpha=config.get("alpha", 0.99),
            eps=config.get("eps", 1e-8),
            weight_decay=config.get("weight_decay", 0.0),
            momentum=config.get("momentum", 0.0),
            centered=config.get("centered", False),
        )
    
    def _create_adamw(self, parameters: Iterator[nn.Parameter], config: Dict[str, Any]) -> optim.AdamW:
        """Create AdamW optimizer."""
        return optim.AdamW(
            parameters,
            lr=config.get("lr", 1e-3),
            betas=(config.get("beta1", 0.9), config.get("beta2", 0.999)),
            eps=config.get("eps", 1e-8),
            weight_decay=config.get("weight_decay", 1e-2),
            amsgrad=config.get("amsgrad", False),
        )
        
    def _create_step_lr(self, optimizer: Optimizer, config: Dict[str, Any]) -> StepLR:
        """Create StepLR scheduler."""
        return StepLR(
            optimizer,
            step_size=config.get("step_size", 10),
            gamma=config.get("gamma", 0.1),
        )
    
    def _create_multi_step_lr(self, optimizer: Optimizer, config: Dict[str, Any]) -> MultiStepLR:
        """Create MultiStepLR scheduler."""
        return MultiStepLR(
            optimizer,
            milestones=config.get("milestones", [30, 60, 90]),
            gamma=config.get("gamma", 0.1),
        )
    
    def _create_reduce_on_plateau(self, optimizer: Optimizer, config: Dict[str, Any]) -> ReduceLROnPlateau:
        """Create ReduceLROnPlateau scheduler."""
        return ReduceLROnPlateau(
            optimizer,
            mode=config.get("mode", "min"),
            factor=config.get("factor", 0.1),
            patience=config.get("patience", 10),
            threshold=config.get("threshold", 1e-4),
            threshold_mode=config.get("threshold_mode", "rel"),
            cooldown=config.get("cooldown", 0),
            min_lr=config.get("min_lr", 0),
            eps=config.get("eps", 1e-8),
        )
    
    def _create_cosine_annealing_lr(self, optimizer: Optimizer, config: Dict[str, Any]) -> CosineAnnealingLR:
        """Create CosineAnnealingLR scheduler."""
        return CosineAnnealingLR(
            optimizer,
            T_max=config.get("T_max", 10),
            eta_min=config.get("eta_min", 0),
        )
    
    def _create_none_scheduler(self, optimizer: Optimizer, config: Dict[str, Any]) -> None:
        """Return None for no scheduler."""
        return None 