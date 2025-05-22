from typing import Dict, Any, Tuple, List, Optional, Callable
import os
import numpy as np
from sklearn.model_selection import KFold

import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, Subset

from interfaces import ConfigDict, DatasetProviderProtocol
from utils.data_func import custom_collate_fn
class BaseDatasetProvider(DatasetProviderProtocol):
    """Base dataset provider for K-Fold cross validation."""
    
    def __init__(self, dataset: Dataset):
        self.dataset = dataset
    
    def get_fold(self, fold_idx: int, num_folds: int) -> Tuple[Dataset, Dataset]:
        """Get train and validation datasets for a specific fold.
        
        Args:
            fold_idx: Fold index.
            num_folds: Number of folds.
                
        Returns:
            Tuple of (train_dataset, val_dataset).
        """
        # Create indices for K-Fold
        indices = list(range(len(self.dataset)))
        
        # Create K-Fold split
        kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
        
        # Convert to list of tuples (train_idx, val_idx)
        splits = list(kf.split(indices))
        
        # Get train and validation indices for the current fold
        train_indices, val_indices = splits[fold_idx]
        
        # Create train and validation subsets
        train_dataset = Subset(self.dataset, train_indices)
        val_dataset = Subset(self.dataset, val_indices)
        
        return train_dataset, val_dataset
    
    def get_train_val_dataloader(self, train_dataset: Dataset, val_dataset: Dataset, 
                               batch_size: int, num_workers: int = 4) -> Tuple[DataLoader, DataLoader]:
        """Get train and validation dataloaders.
        
        Args:
            train_dataset: Training dataset.
            val_dataset: Validation dataset.
            batch_size: Batch size.
            num_workers: Number of workers.
                
        Returns:
            Tuple of (train_dataloader, val_dataloader).
        """
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=custom_collate_fn
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=custom_collate_fn
        )
        
        return train_loader, val_loader

class DataLoaderFactory:
    """Factory for creating dataloaders from config."""
    
    def __init__(self):
        self._dataset_registry = {}
    
    def register_dataset(self, name: str, dataset_fn: Callable[..., Dataset]):
        """Register a dataset constructor function."""
        self._dataset_registry[name] = dataset_fn
    
    def create(self, config: ConfigDict) -> DatasetProviderProtocol:
        """Create dataset provider from config.
        
        Args:
            config: Configuration dict.
                
        Returns:
            Dataset provider.
        """
        dataset_name = config.get("dataset", {}).get("name")
        if dataset_name not in self._dataset_registry:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        # Create dataset
        dataset_config = config.get("dataset", {})
        dataset = self._dataset_registry[dataset_name](**dataset_config.get("params", {}))
        
        return BaseDatasetProvider(dataset) 