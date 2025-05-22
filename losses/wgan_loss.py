import torch
import torch.nn as nn
import torch.autograd as autograd

class WGANGeneratorLoss(nn.Module):
    """WGAN loss for generator.
    
    The generator aims to maximize the critic's score for generated samples,
    which is equivalent to minimizing the negative of that score.
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, critic_fake_outputs: torch.Tensor) -> torch.Tensor:
        """Calculate generator loss.
        
        Args:
            critic_fake_outputs: Critic scores for fake samples.
                
        Returns:
            Generator loss.
        """
        # We want to maximize critic output for fake samples
        # So we minimize -E[critic(fake)]
        return -torch.mean(critic_fake_outputs)

class WGANDiscriminatorLoss(nn.Module):
    """WGAN loss for discriminator (critic).
    
    The critic aims to maximize the Wasserstein distance between real and fake samples.
    This is equivalent to maximizing E[critic(real)] - E[critic(fake)],
    or minimizing E[critic(fake)] - E[critic(real)]
    
    Additionally, we add a gradient penalty term to enforce Lipschitz constraint.
    """
    
    def __init__(self, gradient_penalty_weight: float = 10.0):
        super().__init__()
        self.gradient_penalty_weight = gradient_penalty_weight
    
    def forward(self, critic_real_outputs: torch.Tensor, 
                critic_fake_outputs: torch.Tensor,
                real_samples: torch.Tensor, 
                fake_samples: torch.Tensor, 
                critic: nn.Module) -> torch.Tensor:
        """Calculate critic loss.
        
        Args:
            critic_real_outputs: Critic scores for real samples.
            critic_fake_outputs: Critic scores for fake samples.
            real_samples: Real samples.
            fake_samples: Fake samples.
            critic: Critic model.
                
        Returns:
            Critic loss.
        """
        # Earth Mover (EM) distance
        em_loss = torch.mean(critic_fake_outputs) - torch.mean(critic_real_outputs)
        
        # Calculate gradient penalty
        gradient_penalty = self._gradient_penalty(critic, real_samples, fake_samples)
        
        # Return total loss
        return em_loss + self.gradient_penalty_weight * gradient_penalty
    
    def _gradient_penalty(self, critic: nn.Module, 
                        real_samples: torch.Tensor, 
                        fake_samples: torch.Tensor) -> torch.Tensor:
        """Calculate gradient penalty for WGAN-GP.
        
        Args:
            critic: Critic model.
            real_samples: Real samples.
            fake_samples: Fake samples.
                
        Returns:
            Gradient penalty.
        """
        # Get device
        device = real_samples.device
        
        # Get batch size
        batch_size = real_samples.size(0)
        
        # Create random interpolation factors
        alpha = torch.rand(batch_size, 1, device=device)
        
        # Interpolate between real and fake samples
        interpolates = alpha * real_samples + (1 - alpha) * fake_samples
        interpolates.requires_grad_(True)
        
        # Get critic outputs for interpolated samples
        critic_interpolates = critic(interpolates)
        
        # Create gradient outputs (ones)
        gradients_output = torch.ones(critic_interpolates.size(), device=device)
        
        # Calculate gradients
        gradients = autograd.grad(
            outputs=critic_interpolates,
            inputs=interpolates,
            grad_outputs=gradients_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        # Calculate gradient penalty
        gradients = gradients.view(batch_size, -1)
        gradient_norm = gradients.norm(2, dim=1)
        return torch.mean((gradient_norm - 1) ** 2) 