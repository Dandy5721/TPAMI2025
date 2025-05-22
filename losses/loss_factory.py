from typing import Dict, Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from interfaces import LossProtocol, ConfigDict
from losses.wgan_loss import WGANGeneratorLoss, WGANDiscriminatorLoss

class LossFactory:
    """Factory for creating loss functions from config."""
    
    def __init__(self):
        self._loss_registry = {
            "mse": self._create_mse_loss,
            "bce": self._create_bce_loss,
            "bce_with_logits": self._create_bce_with_logits_loss,
            "l1": self._create_l1_loss,
            "cross_entropy": self._create_cross_entropy_loss,
            "wgan_generator": self._create_wgan_generator_loss,
            "wgan_discriminator": self._create_wgan_discriminator_loss,
        }
    
    def create(self, config: ConfigDict) -> LossProtocol:
        """Create loss function from config."""
        loss_type = config.get("type")
        if loss_type not in self._loss_registry:
            raise ValueError(f"Unknown loss type: {loss_type}")
        
        return self._loss_registry[loss_type](config)
    
    def _create_mse_loss(self, config: ConfigDict) -> nn.MSELoss:
        """Create MSE loss."""
        return nn.MSELoss(reduction=config.get("reduction", "mean"))
    
    def _create_bce_loss(self, config: ConfigDict) -> nn.BCELoss:
        """Create BCE loss."""
        return nn.BCELoss(
            weight=config.get("weight", None),
            reduction=config.get("reduction", "mean")
        )
    
    def _create_bce_with_logits_loss(self, config: ConfigDict) -> nn.BCEWithLogitsLoss:
        """Create BCE with logits loss."""
        return nn.BCEWithLogitsLoss(
            weight=config.get("weight", None),
            reduction=config.get("reduction", "mean"),
            pos_weight=config.get("pos_weight", None)
        )
    
    def _create_l1_loss(self, config: ConfigDict) -> nn.L1Loss:
        """Create L1 loss."""
        return nn.L1Loss(reduction=config.get("reduction", "mean"))
    
    def _create_cross_entropy_loss(self, config: ConfigDict) -> nn.CrossEntropyLoss:
        """Create cross entropy loss."""
        return nn.CrossEntropyLoss(
            weight=config.get("weight", None),
            reduction=config.get("reduction", "mean"),
            ignore_index=config.get("ignore_index", -100)
        )
    
    def _create_wgan_generator_loss(self, config: ConfigDict) -> WGANGeneratorLoss:
        """Create WGAN generator loss."""
        return WGANGeneratorLoss()
    
    def _create_wgan_discriminator_loss(self, config: ConfigDict) -> WGANDiscriminatorLoss:
        """Create WGAN discriminator loss."""
        return WGANDiscriminatorLoss(
            gradient_penalty_weight=config.get("gradient_penalty_weight", 10.0)
        ) 