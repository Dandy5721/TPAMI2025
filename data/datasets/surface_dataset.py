import os
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import load_npz
import torch
from torch.utils.data import Dataset

class SurfaceDataset(Dataset):
    """Surface dataset implementation.
    
    This class loads Tau, Amyloid, and structural data from local files.


    Sample:
        Input Tau shape: torch.Size([163842])
        Output Tau shape: torch.Size([163842])
        Amyloid shape: torch.Size([163842])
        Edge Index shape: torch.Size([2, 1146882])
        Edge Weight shape: torch.Size([1146882])

    Testing DataLoader:
        Batch size: B
        Input Tau shape: torch.Size([B, 163842])
        Output Tau shape: torch.Size([B, 163842])
        Amyloid shape: torch.Size([B, 163842])
        Edge Index shape: torch.Size([B, 2, 1146882])
        Edge Weight shape: torch.Size([B, 1146882])
    """
    
    def __init__(
        self,
        data_dir: str,                # Data root directory path
        transform = None,             # Data transformation
        preload: bool = True          # Whether to preload data
    ):
        """Initialize the dataset.
        
        Args:
            data_dir: Path to the data root directory
            transform: Data transformation function
            preload: Whether to preload all data into memory
        """
        self.transform = transform
        self.data_dir = data_dir
        self.tau_dir = os.path.join(data_dir, 'surface/surface_T')
        self.amyloid_dir = os.path.join(data_dir, 'surface/surface_A')
        self.structural_dir = os.path.join(data_dir, 'surface/structural')
        self.preload = preload
        self.data_cache = {}  # Cache for loaded data
        
        # Get all Tau data files
        self.tau_files = [f for f in os.listdir(self.tau_dir) if f.endswith('.csv')]
        
        # Data pairs for training
        self.data_pairs = []
        
        # Preload all data into memory if requested
        if preload:
            print("Preloading all data into memory...")
            self._preload_data()
        else:
            print("Not preloading data into memory...")

    def _preload_data(self):
        """Preload all data into memory"""
        # Preload all Tau data
        tau_data_cache = {}
        for tau_file in self.tau_files:
            case_id = tau_file.split('.')[0]
            tau_data_cache[case_id] = pd.read_csv(os.path.join(self.tau_dir, tau_file), header=None)
        
        # Preload all Amyloid data
        amyloid_data_cache = {}
        for case_id in tau_data_cache:
            amyloid_file = os.path.join(self.amyloid_dir, f'{case_id}.csv')
            if os.path.exists(amyloid_file):
                amyloid_data_cache[case_id] = pd.read_csv(amyloid_file, header=None).iloc[0].values
        
        # Preload all Structural data
        structural_data_cache = {}
        for case_id in tau_data_cache:
            for f in os.listdir(self.structural_dir):
                if f.startswith(case_id):
                    structural_file = os.path.join(self.structural_dir, f)

                    structural_data_cache[case_id] = load_npz(structural_file)
                    break

        print("########################")
        print(f"Tau data: {len(tau_data_cache)} samples")
        print(f"Amyloid data: {len(amyloid_data_cache)} samples")
        print(f"Structural data: {len(structural_data_cache)} samples")
        print("########################")

        # Build data pairs
        for case_id, tau_data in tau_data_cache.items():
            for i in range(len(tau_data) - 1):
                if case_id in amyloid_data_cache and case_id in structural_data_cache:
                    values = structural_data_cache[case_id].data
                    indices = structural_data_cache[case_id].indices
                    indptr = structural_data_cache[case_id].indptr
                    shape = structural_data_cache[case_id].shape

                    if structural_data_cache[case_id].format == 'csr':
                        matrix = sparse.csr_matrix((values, indices, indptr), shape=shape)
                    elif structural_data_cache[case_id].format == 'csc':
                        matrix = sparse.csc_matrix((values, indices, indptr), shape=shape)
                    else:
                        raise ValueError(f"Unsupported sparse matrix format: {structural_data_cache[case_id].format}")
                    
                    matrix_coo = matrix.tocoo()
                    edge_index = torch.tensor([matrix_coo.row, matrix_coo.col], dtype=torch.long)
                    edge_weight = torch.tensor(matrix_coo.data, dtype=torch.float)

                    self.data_pairs.append({
                        'input_tau': tau_data.iloc[i].values,
                        'output_tau': tau_data.iloc[i + 1].values,
                        'amyloid': amyloid_data_cache[case_id],
                        'edge_index': edge_index,
                        'edge_weight': edge_weight,
                        'case_id': case_id + '_' + str(i)
                    })
    
        self.data_pairs.sort(key=lambda x: x['case_id'])
        print(f"Total training samples created: {len(self.data_pairs)}")

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.data_pairs)
    
    def __getitem__(self, idx):
        """Get a sample."""
        data = self.data_pairs[idx]

        if self.preload:
            # Data is already preloaded, use directly
            sample = {
                'input_tau': torch.tensor(data['input_tau'], dtype=torch.float),
                'output_tau': torch.tensor(data['output_tau'], dtype=torch.float),
                'amyloid': torch.tensor(data['amyloid'], dtype=torch.float),
                'edge_index': data['edge_index'],
                'edge_weight': data['edge_weight'],
                'case_id': data['case_id']
            }
        else:
            # Lazy loading implementation
            case_id, index = data['case_id'].rsplit('_', 1)
            index = int(index)
            
            # Use cache to avoid repeated loading
            if case_id not in self.data_cache:
                tau_file = os.path.join(self.tau_dir, f'{case_id}.csv')
                amyloid_file = os.path.join(self.amyloid_dir, f'{case_id}.csv')
                
                # Find corresponding structural data file
                structural_file = None
                for f in os.listdir(self.structural_dir):
                    if f.startswith(case_id):
                        structural_file = os.path.join(self.structural_dir, f)
                        break
                
                if not os.path.exists(tau_file) or not os.path.exists(amyloid_file) or structural_file is None:
                    raise FileNotFoundError(f"Cannot find complete data for {case_id}")
                
                tau_data = pd.read_csv(tau_file, header=None)
                amyloid_data = pd.read_csv(amyloid_file, header=None).iloc[0].values
                structural_data = load_npz(structural_file)
                matrix = sparse.csr_matrix(structural_data)
                matrix_coo = matrix.tocoo()
                edge_index = torch.tensor([matrix_coo.row, matrix_coo.col], dtype=torch.long)
                edge_weight = torch.tensor(matrix_coo.data, dtype=torch.float)

                self.data_cache[case_id] = {
                    'tau_data': tau_data,
                    'amyloid_data': amyloid_data,
                    'structural_data': structural_data,
                    'edge_index': edge_index,
                    'edge_weight': edge_weight
                }
            
            cached_data = self.data_cache[case_id]
            
            sample = {
                'input_tau': torch.tensor(cached_data['tau_data'].iloc[index].values, dtype=torch.float),
                'output_tau': torch.tensor(cached_data['tau_data'].iloc[index + 1].values, dtype=torch.float),
                'amyloid': torch.tensor(cached_data['amyloid_data'], dtype=torch.float),
                'edge_index': cached_data['edge_index'],
                'edge_weight': cached_data['edge_weight'],
                'case_id': data['case_id']
            }
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample 