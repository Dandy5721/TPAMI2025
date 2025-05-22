from typing import Dict, Any, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.nn import functional as F

from interfaces import ModelProtocol, HookProtocol, MetricsDict, ConfigDict
from trainers.base_trainer import BaseTrainer


class RawGANTrainer(BaseTrainer):
    """Trainer for RawGAN models.
    
    This trainer handles the specific requirements of RawGAN training, including:
    - Integration of ODEs over time steps
    - Handling of initial conditions
    - Multiple evaluation points
    - ODE-specific loss functions
    """
    
    def __init__(
        self,
        config: ConfigDict,
        models: Dict[str, ModelProtocol],
        loss_fn: Dict[str, nn.Module],
        optimizer: Dict[str, Optimizer],
        scheduler: Dict[str, Optional[_LRScheduler]],
        train_loader: DataLoader,
        val_loader: DataLoader,
        hooks: List[HookProtocol],
        device: torch.device,
    ):
        """Initialize RawGAN trainer.
        
        Args:
            config: Configuration dict.
            models: Dictionary containing generator and discriminator models.
            loss_fn: Dictionary containing generator and discriminator loss functions.
            optimizer: Dictionary containing generator and discriminator optimizers.
            scheduler: Dictionary containing generator and discriminator schedulers.
            train_loader: Training data loader.
            val_loader: Validation data loader.
            hooks: List of hooks.
            device: Device to train on.
        """
        # Store the dictionaries of components
        self.models = models
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        
        # Initialize base trainer with the generator model and its components
        super().__init__(
            config=config,
            model=models["generator"],  # Use generator as the main model
            loss_fn=loss_fn["generator_loss"],  # Use generator loss as the main loss
            optimizer=optimizer["generator"],  # Use generator optimizer as the main optimizer
            scheduler=scheduler.get("generator"),  # Use generator scheduler as the main scheduler
            train_loader=train_loader,
            val_loader=val_loader,
            hooks=hooks,
            device=device,
        )

        self.generator = models["generator"]
        self.generator_loss = loss_fn["generator_loss"]
        self.generator_optimizer = optimizer["generator"]
        self.generator_scheduler = scheduler.get("generator")
        
        # Store critic-specific components
        self.critic = models["critic"]
        self.critic_loss = loss_fn["critic_loss"]
        self.critic_optimizer = optimizer["critic"]
        self.critic_scheduler = scheduler.get("critic")
        
        # Override the metrics dictionary
        self._metrics = {
            "train_g_loss": 0.0,
            "train_d_loss": 0.0,
            "train_mae": 0.0,
            "val_g_loss": 0.0,
            "val_d_loss": 0.0,
            "val_mae": 0.0,
        }
    
    def _train_epoch(self, epoch: int) -> MetricsDict:
        """Train one epoch.
        
        Args:
            epoch: Current epoch.
            
        Returns:
            Training metrics.
        """
        generator = self.models["generator"]
        critic = self.models["critic"]

        # Set models to training mode
        generator.train()
        critic.train()
        
        # Initialize metrics
        total_d_loss = 0.0
        total_g_loss = 0.0
        total_mae = 0.0
        batches = 0
        
      
        # Loop over batches
        for batch_idx, batch in enumerate(self.train_loader):
            batches += 1

            batch_size = batch['input_tau'].shape[0]
            
            # Call hooks for batch begin
            for hook in self.hooks:
                hook.on_batch_begin(self, batch_idx)
            
            # Move data to device
            tau = batch['input_tau'].to(self.device)
            target = batch['output_tau'].to(self.device)
            
            # Create labels for real and fake data
            real_labels = torch.ones(batch_size, 1, device=self.device)
            fake_labels = torch.zeros(batch_size, 1, device=self.device)
            
            # -----------------
            # Train Discriminator
            # -----------------
            self.critic_optimizer.zero_grad()
            
            # Train with real data
            real_outputs = self.critic(target)
            d_loss_real = F.binary_cross_entropy_with_logits(real_outputs, real_labels)
            
            # Train with fake data
            fake_data = self.generator(tau)
            fake_outputs = self.critic(fake_data.detach())  # Detach to avoid training generator
            d_loss_fake = F.binary_cross_entropy_with_logits(fake_outputs, fake_labels)
            
            # Combined discriminator loss
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            self.critic_optimizer.step()
        
            # -----------------
            # Train Generator
            # -----------------
            self.generator_optimizer.zero_grad()
        
            # Generate fake data and try to fool the discriminator
            fake_data = self.generator(tau)
            fake_outputs = self.critic(fake_data)
        
            # Generator loss - we want discriminator to classify fake data as real
            g_loss = F.binary_cross_entropy_with_logits(fake_outputs, real_labels)
            g_loss.backward()
            self.generator_optimizer.step()
           
            
            # Calculate MAE
            mae = F.l1_loss(fake_data, target)
            total_g_loss += g_loss.item()
            total_mae += mae.item()
            
            # Call hooks for batch end
            batch_metrics = {
                "train_d_loss": d_loss.item(),
                "train_g_loss": g_loss.item(),
                "train_mae": mae.item()
            }
            for hook in self.hooks:
                hook.on_batch_end(self, batch_idx, g_loss, batch_metrics)
        
        # Compute average metrics
        avg_d_loss = total_d_loss / batches 
        avg_g_loss = total_g_loss / batches
        avg_mae = total_mae / batches
        
        # Return metrics
        return {
            "train_d_loss": avg_d_loss,
            "train_g_loss": avg_g_loss,
            "train_mae": avg_mae
        }
    
    def validate(self) -> MetricsDict:
        """Validate model.
        
        Returns:
            Validation metrics.
        """
        generator = self.models["generator"]
        critic = self.models["critic"]
        
        # Set models to evaluation mode
        generator.eval()
        critic.eval()
        
        # Initialize metrics
        total_d_loss = 0.0
        total_g_loss = 0.0
        total_mae = 0.0
        batches = 0
        
        # No gradient computation during validation
        with torch.no_grad():
            for batch in self.val_loader:
                batches += 1
                
                # Move data to device
                tau = batch['input_tau'].to(self.device)
                target = batch['output_tau'].to(self.device)
                
                batch_size = tau.shape[0]
                real_labels = torch.ones(batch_size, 1, device=self.device)
                fake_labels = torch.zeros(batch_size, 1, device=self.device)
                
                # Generate predictions
                fake = generator(tau)
                
                # Calculate validation losses
                real_outputs = critic(target)
                fake_outputs = critic(fake)
                
                # Calculate losses using binary cross entropy with logits
                val_d_loss = (F.binary_cross_entropy_with_logits(real_outputs, real_labels) + 
                             F.binary_cross_entropy_with_logits(fake_outputs, fake_labels))
                val_g_loss = F.binary_cross_entropy_with_logits(fake_outputs, real_labels)
                
                # Calculate MAE
                mae = F.l1_loss(fake, target)
                
                total_d_loss += val_d_loss.item()
                total_g_loss += val_g_loss.item()
                total_mae += mae.item()
        
        # Compute average metrics
        avg_d_loss = total_d_loss / batches
        avg_g_loss = total_g_loss / batches
        avg_mae = total_mae / batches
        
        # Return metrics
        return {
            "val_d_loss": avg_d_loss,
            "val_g_loss": avg_g_loss,
            "val_mae": avg_mae
        }