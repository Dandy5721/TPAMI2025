import torch

def custom_collate_fn(batch):
    """Custom collate function to handle variable-sized edge indices and weights."""
    return {
        'input_tau': torch.stack([item['input_tau'] for item in batch]),
        'output_tau': torch.stack([item['output_tau'] for item in batch]),
        'amyloid': torch.stack([item['amyloid'] for item in batch]),
        'case_id': [item['case_id'] for item in batch],
        'edge_index': [item['edge_index'] for item in batch],  # Keep as list
        'edge_weight': [item['edge_weight'] for item in batch]  # Keep as list
    }