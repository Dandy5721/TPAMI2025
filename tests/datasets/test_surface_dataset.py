import os
import argparse
import torch
from torch.utils.data import DataLoader
from data.datasets.surface_dataset import SurfaceDataset


def custom_collate_fn(batch):
    """Custom collate function to handle variable-sized edge indices and weights."""
    return {
        'input_tau': torch.stack([item['input_tau'] for item in batch]),
        'output_tau': torch.stack([item['output_tau'] for item in batch]),
        'case_id': [item['case_id'] for item in batch],
        'edge_index': [item['edge_index'] for item in batch],
        'edge_weight': [item['edge_weight'] for item in batch],
        'structural': torch.stack([item['structural'] for item in batch]),
        'amyloid': torch.stack([item['amyloid'] for item in batch])
    }


def test_surface_dataset(data_dir):
    """Test the functionality of SurfaceDataset class"""
    print(f"Using data directory: {data_dir}")
    
    # Create dataset instance
    dataset = SurfaceDataset(data_dir=data_dir, preload=True)
    
    # Print dataset size
    print(f"Dataset size: {len(dataset)}")
    
    # Check several samples
    for i in [0, len(dataset)//2, len(dataset)-1]:
        if i < len(dataset):
            sample = dataset[i]
            print(f"\nSample {i}:")
            print(f"  Case ID: {sample['case_id']}")
            print(f"  Input Tau shape: {sample['input_tau'].shape}")
            print(f"  Output Tau shape: {sample['output_tau'].shape}")
            print(f"  Amyloid shape: {sample['amyloid'].shape}")
            print(f"  Edge Index shape: {sample['edge_index'].shape}")
            print(f"  Edge Weight shape: {sample['edge_weight'].shape}")
    
    # Test DataLoader
    if len(dataset) > 0:
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=custom_collate_fn)
        print("\nTesting DataLoader:")
        batch = next(iter(dataloader))
        print(f"  Batch size: {len(batch['input_tau'])}")
        print(f"  Input Tau shape: {batch['input_tau'].shape}")
        print(f"  Output Tau shape: {batch['output_tau'].shape}")
        print(f"  Amyloid shape: {batch['amyloid'].shape}")
        print(f"  Number of edge indices: {len(batch['edge_index'])}")
        print(f"  Edge index shapes: {[idx.shape for idx in batch['edge_index']]}")
        print(f"  Edge weight shapes: {[w.shape for w in batch['edge_weight']]}")
        print(f"  Case IDs: {batch['case_id']}")

def main():
    parser = argparse.ArgumentParser(description="Test Surface Dataset")
    parser.add_argument("--data_dir", type=str, required=True, 
                      help="Path to the data root directory, e.g., /path/to/data/")
    args = parser.parse_args()
    
    test_surface_dataset(args.data_dir)

if __name__ == "__main__":
    main() 