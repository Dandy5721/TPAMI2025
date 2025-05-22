import numpy as np
from sklearn.linear_model import Ridge
from dataclasses import dataclass
import torch

@dataclass
class RidgeConfig:
    alpha: float = 1.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

class RidgeModel:
    def __init__(self, config: RidgeConfig):
        """
        Initialize Ridge regression model
        
        Args:
            config (RidgeConfig): Configuration object containing alpha and device
        """
        self.config = config
        self.model = Ridge(alpha=config.alpha)
        
    def forward(self, X, y):
        """
        Fit the Ridge model and return predictions
        
        Args:
            X (np.ndarray): Input features
            y (np.ndarray): Target values
            
        Returns:
            np.ndarray: Predicted values
        """
        self.model.fit(X, y)
        return self.model.predict(X)
        
    def eval(self, X):
        """
        Make predictions using the fitted model
        
        Args:
            X (np.ndarray): Input features
            
        Returns:
            np.ndarray: Predicted values
        """
        return self.model.predict(X)
        
    def get_params(self):
        """
        Get model parameters
        
        Returns:
            dict: Model parameters
        """
        return self.model.get_params()
        
    def set_params(self, **params):
        """
        Set model parameters
        
        Args:
            **params: Parameters to set
        """
        self.model.set_params(**params) 