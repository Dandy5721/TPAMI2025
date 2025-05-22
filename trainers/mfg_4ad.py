from typing import Dict, Any, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.nn import functional as F

from interfaces import ModelProtocol, HookProtocol, MetricsDict, ConfigDict
from trainers.base_trainer import BaseTrainer


def compute_gradient_penalty(D, real_samples, fake_samples, edge_index, edge_weight):
    device = real_samples.device
    alpha = torch.rand(real_samples.size(0), 1, device=device)
    alpha = alpha.expand_as(real_samples)
    
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
    interpolates = torch.clamp(interpolates, -1, 1) 
    
    d_interpolates = D(interpolates, edge_index, edge_weight)
    fake = torch.ones_like(d_interpolates, device=device)
    
    try:
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        
        if not torch.isfinite(gradients).all():
            return torch.tensor(0.0, device=device)
            
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        
        if gradient_penalty > 100:
            return torch.tensor(100.0, device=device)
            
        return gradient_penalty
        
    except RuntimeError:
        return torch.tensor(0.0, device=device)

class MFG4ADTrainer(BaseTrainer):
    """Trainer for MFG4AD models.
    
    This trainer handles the specific requirements of MFG4AD training, including:
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
        """Initialize MFG4AD trainer.
        
        Args:
            config: Configuration dict.
            models: Dictionary containing generator and critic models.
            loss_fn: Dictionary containing generator and critic loss functions.
            optimizer: Dictionary containing generator and critic optimizers.
            scheduler: Dictionary containing generator and critic schedulers.
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
        print(f"train_epoch: {epoch}")
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
        
        # Get training parameters from config
        critic_iter = self.config.get("critic_iter", 5)
        lambda_gp = self.config.get("lambda_gp", 10)
        lambda_rec = self.config.get("lambda_rec", 10.0)
        
        # Loop over batches
        for batch_idx, batch in enumerate(self.train_loader):
            batches += 1
            # Call hooks for batch begin
            for hook in self.hooks:
                hook.on_batch_begin(self, batch_idx)
            
            # Move data to device
            tau = batch['input_tau'].to(self.device)
            target = batch['output_tau'].to(self.device)
            amy = batch['amyloid'].to(self.device)
            
            # Prepare edge information
            if isinstance(batch['edge_index'], list):
                edge_index = batch['edge_index'][0].squeeze(0).to(self.device)
            else:
                edge_index = batch['edge_index'].squeeze(0).to(self.device)
            
            if isinstance(batch['edge_weight'], list):
                edge_weight = batch['edge_weight'][0].squeeze(0).to(self.device)
            else:
                edge_weight = batch['edge_weight'].squeeze(0).to(self.device)
            
            # ---------------------
            # 1) Train Critic
            # ---------------------
            for _ in range(critic_iter):
                self.critic_optimizer.zero_grad()
                
                with torch.no_grad():
                    fake = generator(tau, amy, edge_index, edge_weight)
                
                # Reshape for critic
                real_x = target.squeeze(0).unsqueeze(-1)
                fake_x = fake.detach().squeeze(0).unsqueeze(-1)
                
                real_score = critic(real_x, edge_index, edge_weight)
                fake_score = critic(fake_x, edge_index, edge_weight)
                
                # Compute gradient penalty
                gp = compute_gradient_penalty(
                    critic, real_x.unsqueeze(0), fake_x.unsqueeze(0), edge_index, edge_weight
                )
                
                d_loss = -real_score.mean() + fake_score.mean() + lambda_gp * gp
                d_loss.backward()
                self.critic_optimizer.step()
                
                total_d_loss += d_loss.item()
            
            # ---------------------
            # 2) Train Generator
            # ---------------------
            self.optimizer.zero_grad()
            print(f"train_epoch: {epoch} batch_idx: {batch_idx}")

            # 打印shape
            print(f"tau shape: {tau.shape}")
            print(f"amy shape: {amy.shape}")
            print(f"edge_index shape: {edge_index.shape}")
            print(f"edge_weight shape: {edge_weight.shape}")

            fake = generator(tau, amy, edge_index, edge_weight)

            print(f"fake shape: {fake.shape}")
            fake_x = fake.squeeze(0).unsqueeze(-1)
            fake_score = critic(fake_x, edge_index, edge_weight)

            print(f"fake_score: {fake_score}")
            print(f"fake_score.mean(): {fake_score.mean()}")
            print(f"fake values: {fake}")
            print(f"target values: {target}")
            print(f"lambda_rec: {lambda_rec}")
            print(f"L1 loss: {F.l1_loss(fake, target)}")

            g_loss = -fake_score.mean() + lambda_rec * F.l1_loss(fake, target)
            #  lambda_rec * F.l1_loss(fake, target)

            g_loss.backward()
            self.optimizer.step()
            
            # Calculate MAE
            mae = F.l1_loss(fake, target)

            print(f"g_loss: {g_loss.item()}")
            print(f"mae: {mae.item()}")

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

    
        # 打印total_d_loss 
        print(f"total_d_loss: {total_d_loss}")
        print(f"total_g_loss: {total_g_loss}")
        print(f"total_mae: {total_mae}")

        # Compute average metrics
        avg_d_loss = total_d_loss / (batches * critic_iter)
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
        
        # Get validation parameters from config
        lambda_rec = self.config.get("lambda_rec", 10.0)
        
        # No gradient computation during validation
        with torch.no_grad():
            for batch in self.val_loader:
                batches += 1
                
                # Move data to device
                tau = batch['input_tau'].to(self.device)
                target = batch['output_tau'].to(self.device)
                amy = batch['amyloid'].to(self.device)
                
                # Prepare edge information
                if isinstance(batch['edge_index'], list):
                    edge_index = batch['edge_index'][0].squeeze(0).to(self.device)
                else:
                    edge_index = batch['edge_index'].squeeze(0).to(self.device)
            
                if isinstance(batch['edge_weight'], list):
                    edge_weight = batch['edge_weight'][0].squeeze(0).to(self.device)
                else:
                    edge_weight = batch['edge_weight'].squeeze(0).to(self.device)
                
                # Generate predictions
                fake = generator(tau, amy, edge_index, edge_weight)
                
                # Calculate validation losses
                real_x = target.squeeze(0).unsqueeze(-1)
                fake_x = fake.squeeze(0).unsqueeze(-1)
                
                real_score = critic(real_x, edge_index, edge_weight)
                fake_score = critic(fake_x, edge_index, edge_weight)
                
                val_d_loss = -real_score.mean() + fake_score.mean()
                val_g_loss = -fake_score.mean() + lambda_rec * F.l1_loss(fake, target)
                
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