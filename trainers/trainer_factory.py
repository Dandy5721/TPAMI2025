from typing import Dict, Any, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from interfaces import ModelProtocol, LossProtocol, HookProtocol, TrainerProtocol, ConfigDict
from trainers.mlp_trainer import MLPTrainer
from trainers.ode_trainer import ODETrainer
from trainers.gcn_trainer import GCNTrainer
from trainers.ridge_trainer import RidgeTrainer
from trainers.mfg_4ad import MFG4ADTrainer
from trainers.deep_symbolic_trainer import DeepSymbolicTrainer
from trainers.raw_gan import RawGANTrainer
class TrainerFactory:
    """Factory for creating trainers from config."""
    
    def __init__(self):
        """Initialize trainer factory."""
        self._trainer_registry = {
            "mlp": self._create_mlp_trainer,
            "ode": self._create_ode_trainer,
            "gcn": self._create_gcn_trainer,
            "ridge": self._create_ridge_trainer,
            "mfg4ad": self._create_mfg4ad_trainer,
            "deep_symbolic": self._create_deep_symbolic_trainer,
            "raw_gan": self._create_raw_gan_trainer,
        }
    
    def create(
        self,
        config: ConfigDict,
        models: Dict[str, ModelProtocol],
        losses: Dict[str, LossProtocol],
        optimizers: Dict[str, Optimizer],
        schedulers: Dict[str, Optional[_LRScheduler]],
        train_loader: DataLoader,
        val_loader: DataLoader,
        hooks: List[HookProtocol],
        device: torch.device,
    ) -> TrainerProtocol:
        """Create trainer from config.
        
        Args:
            config: Configuration dict.
            models: Dictionary of models.
            losses: Dictionary of loss functions.
            optimizers: Dictionary of optimizers.
            schedulers: Dictionary of schedulers.
            train_loader: Training data loader.
            val_loader: Validation data loader.
            hooks: List of hooks.
            device: Device to train on.
            
        Returns:
            Trainer instance.
        """
        trainer_type = config.get("trainer", {}).get("type")
        if trainer_type not in self._trainer_registry:
            raise ValueError(f"Unknown trainer type: {trainer_type}")
        
        return self._trainer_registry[trainer_type](
            config=config,
            models=models,
            losses=losses,
            optimizers=optimizers,
            schedulers=schedulers,
            train_loader=train_loader,
            val_loader=val_loader,
            hooks=hooks,
            device=device,
        )
    
    def _create_ode_trainer(
        self,
        config: ConfigDict,
        models: Dict[str, ModelProtocol],
        losses: Dict[str, LossProtocol],
        optimizers: Dict[str, Optimizer],
        schedulers: Dict[str, Optional[_LRScheduler]],
        train_loader: DataLoader,
        val_loader: DataLoader,
        hooks: List[HookProtocol],
        device: torch.device,
    ) -> ODETrainer:
        """Create ODE trainer.
        
        Args:
            config: Configuration dict.
            models: Dictionary of models.
            losses: Dictionary of loss functions.
            optimizers: Dictionary of optimizers.
            schedulers: Dictionary of schedulers.
            train_loader: Training data loader.
            val_loader: Validation data loader.
            hooks: List of hooks.
            device: Device to train on.
            
        Returns:
            ODE trainer instance.
        """
        return ODETrainer(
            config=config,
            model=models["model"],
            loss_fn=losses["loss"],
            optimizer=optimizers["optimizer"],
            scheduler=schedulers.get("scheduler"),
            train_loader=train_loader,
            val_loader=val_loader,
            hooks=hooks,
            device=device,
        )
    
    
    def _create_mlp_trainer(
        self,
        config: ConfigDict,
        models: Dict[str, ModelProtocol],
        losses: Dict[str, LossProtocol],
        optimizers: Dict[str, Optimizer],
        schedulers: Dict[str, Optional[_LRScheduler]],
        train_loader: DataLoader,
        val_loader: DataLoader,
        hooks: List[HookProtocol],
        device: torch.device,
    ) -> MLPTrainer:
        """Create MLP trainer.
        
        Args:
            config: Configuration dict.
            models: Dictionary of models.
            losses: Dictionary of loss functions.
            optimizers: Dictionary of optimizers.
            schedulers: Dictionary of schedulers.
            train_loader: Training data loader.
            val_loader: Validation data loader.
            hooks: List of hooks.
            device: Device to train on.
            
        Returns:
            MLP trainer instance.
        """
        return MLPTrainer(
            config=config,
            model=models["model"],
            loss_fn=losses["loss"],
            optimizer=optimizers["optimizer"],
            scheduler=schedulers.get("scheduler"),
            train_loader=train_loader,
            val_loader=val_loader,
            hooks=hooks,
            device=device,
        )
    
    def _create_gcn_trainer(
        self,
        config: ConfigDict,
        models: Dict[str, ModelProtocol],
        losses: Dict[str, LossProtocol],
        optimizers: Dict[str, Optimizer],
        schedulers: Dict[str, Optional[_LRScheduler]],
        train_loader: DataLoader,
        val_loader: DataLoader,
        hooks: List[HookProtocol],
        device: torch.device,
    ) -> GCNTrainer:
        """Create GCN trainer."""
        return GCNTrainer(
            config=config,
            model=models["model"],
            loss_fn=losses["loss"],
            optimizer=optimizers["optimizer"],
            scheduler=schedulers.get("scheduler"),
            train_loader=train_loader,
            val_loader=val_loader,
            hooks=hooks,
            device=device,
        ) 
    
    def _create_ridge_trainer(
        self,
        config: ConfigDict,
        models: Dict[str, ModelProtocol],
        losses: Dict[str, LossProtocol],
        optimizers: Dict[str, Optimizer],
        schedulers: Dict[str, Optional[_LRScheduler]],
        train_loader: DataLoader,
        val_loader: DataLoader,
        hooks: List[HookProtocol],
        device: torch.device,
    ) -> RidgeTrainer:
        """Create Ridge trainer."""
        return RidgeTrainer(
            config=config,
            model=models["model"],
            loss_fn=losses.get("loss"),
            optimizer=optimizers.get("optimizer"),
            scheduler=schedulers.get("scheduler"),
            train_loader=train_loader,
            val_loader=val_loader,
            hooks=hooks,
            device=device,
        )
    
    def _create_mfg4ad_trainer(
        self,
        config: ConfigDict,
        models: Dict[str, ModelProtocol],
        losses: Dict[str, LossProtocol],
        optimizers: Dict[str, Optimizer],
        schedulers: Dict[str, Optional[_LRScheduler]],
        train_loader: DataLoader,
        val_loader: DataLoader,
        hooks: List[HookProtocol],
        device: torch.device,
    ) -> MFG4ADTrainer:
        """Create MFG4AD trainer."""
        return MFG4ADTrainer(
            config=config,
            # include models["generator"], models["critic"]
            models=models,

            # include losses["generator_loss"], losses["critic_loss"]   
            loss_fn=losses,

            # include optimizers["optimizer"], optimizers["critic_optimizer"]
            optimizer=optimizers,

            # include schedulers["scheduler"], schedulers["critic_scheduler"]
            scheduler=schedulers,
            train_loader=train_loader,
            val_loader=val_loader,
            hooks=hooks,
            device=device,
        )
    
    def _create_deep_symbolic_trainer(
        self,
        config: ConfigDict,
        models: Dict[str, ModelProtocol],
        losses: Dict[str, LossProtocol],
        optimizers: Dict[str, Optimizer],
        schedulers: Dict[str, Optional[_LRScheduler]],
        train_loader: DataLoader,
        val_loader: DataLoader,
        hooks: List[HookProtocol],
        device: torch.device,
    ) -> DeepSymbolicTrainer:
        """Create DeepSymbolic trainer."""
        return DeepSymbolicTrainer(
            config=config,
            model=models["model"],
            loss_fn=losses["loss"],
            optimizer=optimizers["optimizer"],
            scheduler=schedulers["scheduler"],
            train_loader=train_loader,
            val_loader=val_loader,
            hooks=hooks,
            device=device,
        )
    
    def _create_raw_gan_trainer(
        self,
        config: ConfigDict,
        models: Dict[str, ModelProtocol],
        losses: Dict[str, LossProtocol],
        optimizers: Dict[str, Optimizer],
        schedulers: Dict[str, Optional[_LRScheduler]],
        train_loader: DataLoader,
        val_loader: DataLoader,
        hooks: List[HookProtocol],
        device: torch.device,
    ) -> RawGANTrainer:
        """Create RawGAN trainer."""
        return RawGANTrainer(
            config=config,
            models=models,
            loss_fn=losses,
            optimizer=optimizers,
            scheduler=schedulers,
            train_loader=train_loader,
            val_loader=val_loader,
            hooks=hooks,
            device=device,
        )
    

