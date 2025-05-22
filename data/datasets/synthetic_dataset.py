import numpy as np
import torch
from torch.utils.data import Dataset

class SyntheticDataset(Dataset):
    """Synthetic dataset for testing.
    
    This dataset generates random data points and labels:
    - For classification: random features and integer labels
    - For regression: random features and continuous target values
    - For GANs: random samples from a distribution
    """
    
    def __init__(
        self,
        num_samples: int = 1000,
        input_dim: int = 10,
        output_dim: int = 1,
        task_type: str = "regression",
        noise_level: float = 0.1,
        seed: int = 42
    ):
        """Initialize dataset.
        
        Args:
            num_samples: Number of samples.
            input_dim: Input dimension.
            output_dim: Output dimension.
            task_type: Task type (regression, classification, gan).
            noise_level: Noise level.
            seed: Random seed.
        """
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        self.num_samples = num_samples
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.task_type = task_type
        self.noise_level = noise_level
        
        self._generate_data()
    
    def _generate_data(self):
        """Generate synthetic data."""
        # Generate input features
        self.features = torch.randn(self.num_samples, self.input_dim)
        
        if self.task_type == "regression":
            # Generate random weights and bias
            weights = torch.randn(self.input_dim, self.output_dim)
            bias = torch.randn(self.output_dim)
            
            # Generate target values with noise
            self.targets = torch.matmul(self.features, weights) + bias
            noise = torch.randn_like(self.targets) * self.noise_level
            self.targets += noise
            
        elif self.task_type == "classification":
            # Generate random weights and bias for logits
            weights = torch.randn(self.input_dim, self.output_dim)
            bias = torch.randn(self.output_dim)
            
            # Generate logits
            logits = torch.matmul(self.features, weights) + bias
            
            # Convert to probabilities and then to class labels
            if self.output_dim == 1:
                # Binary classification
                probs = torch.sigmoid(logits)
                self.targets = (probs > 0.5).float()
            else:
                # Multi-class classification
                probs = torch.softmax(logits, dim=1)
                self.targets = torch.argmax(probs, dim=1).view(-1, 1)
                
        elif self.task_type == "gan":
            # For GAN tasks, we only need the features (real samples)
            # Create samples from a mixture of Gaussians for more interesting data
            n_centers = 5
            centers = torch.randn(n_centers, self.input_dim) * 5
            indices = torch.randint(0, n_centers, (self.num_samples,))
            
            # Generate samples around centers
            self.features = centers[indices] + torch.randn(self.num_samples, self.input_dim)
            self.targets = torch.zeros(self.num_samples, 1)  # Dummy targets
        
        else:
            raise ValueError(f"Unknown task type: {self.task_type}")
    
    def __len__(self) -> int:
        """Return number of samples."""
        return self.num_samples
    
    def __getitem__(self, idx: int) -> tuple:
        """Get sample by index."""
        return self.features[idx], self.targets[idx] 