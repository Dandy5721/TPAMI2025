# MFG4AD

A modular and extendable deep learning training framework that supports:
- Multiple model types (MLP, GCN, GAN, WGAN and so on ...check the comparative methods in Table 1)
- K-fold cross-validation
- Configuration-driven training
- Advanced monitoring and logging

## Features

- **Modular Design**: Clear separation between models, losses, optimizers, trainers, etc.
- **K-Fold Cross Validation**: Support for K-fold training with proper metric aggregation
- **Configuration-Driven**: All aspects of training controlled through YAML configuration files
- **Multiple Model Support**: Ready to use implementations for MLP, GAN, and WGAN ...
- **Flexible Dataset Handling**: Support for various dataset types with a common interface
- **Advanced Logging**: JSON logging with support for metrics tracking
- **Hooks System**: Customizable training hooks for early stopping, model checkpointing, etc.

## Project Structure

```
project/
├── configs/                 # YAML configuration files
│   ├── mlp_exp.yaml        # MLP experiment configuration
│   ├── wgan_exp.yaml       # WGAN experiment configuration
│   ├── ***_exp.yaml       # *** experiment configuration     
│   └── mfg4ad_exp.yaml       # MFG4AD experiment configuration

├── data/                    # Data loading modules
│   ├── dataloader_factory.py    # Factory for creating dataloaders
│   └── datasets/                # Dataset implementations
│       └── synthetic_dataset.py # Synthetic dataset for testing
├── models/                  # Model implementations
│   ├── model_factory.py     # Factory for creating models
│   ├── mlp.py               # MLP model implementation
│   ├── gan.py               # GAN model implementation
│   ├── ***.py               # *** model implementation
│   └── mfg_4ad.py           # mfg_4ad model implementation
├── losses/                  # Loss functions
│   ├── loss_factory.py      # Factory for creating loss functions
│   └── wgan_loss.py         # WGAN loss implementations
├── optim/                   # Optimization modules
│   └── optim_factory.py     # Factory for creating optimizers and schedulers
├── trainers/                # Trainer implementations
│   ├── trainer_factory.py   # Factory for creating trainers
│   ├── base_trainer.py      # Base trainer implementation
│   ├── mlp_trainer.py       # MLP trainer implementation
│   ├── ***_trainer.py       # *** trainer implementation
│   └── mfg_4ad.py           # mfg_4ad trainer implementation
├── hooks/                   # Training hooks
│   ├── base_hook.py         # Base hook implementation
│   ├── early_stopping.py    # Early stopping hook
│   ├── best_tracker.py      # Best model tracker hook
│   └── logger.py            # Logger hook
├── utils/                   # Utility functions
├── inference/               # Inference modules
│   └── predict.py           # Model prediction script
├── interfaces.py            # Protocol definitions for interfaces
├── main.py                  # Main training script
├── brain_network_con.py     # Brain network construction script
└── README.md                # This file
```

## Usage

### Training

To train a model, create a configuration file in the `configs` directory and run:

```bash
python main.py --config configs/mlp_exp1.yaml --device cuda
```

This will:
1. Load the configuration
2. Setup the necessary components (model, loss, optimizer, etc.)
3. Run K-fold cross-validation
4. Save the trained models and metrics

### Inference

To make predictions with a trained model, run:

```bash
python inference/predict.py --config configs/mlp_exp.yaml --checkpoint checkpoints/mlp_exp/fold_0/best.pt --device cuda
```

For WGAN, you can generate samples:

```bash
python inference/predict.py --config configs/wgan_exp.yaml --checkpoint checkpoints/wgan_exp/fold_0/best.pt --num_samples 100 --device cuda
```

## Configuration

The system is designed to be fully configurable through YAML files. Here's an example configuration for a simple MLP model:

```yaml
# MLP Experiment Configuration
exp_name: mlp_exp

# Model configuration
model:
  type: mlp
  input_dim: 10
  hidden_dims: [64, 32]
  output_dim: 1
  activation: relu
  dropout: 0.1
  batch_norm: true

# Loss configuration
loss:
  type: mse
  reduction: mean

# Optimizer configuration
optim:
  optimizer:
    type: adam
    lr: 0.001
    beta1: 0.9
    beta2: 0.999
    weight_decay: 0.0001
  scheduler:
    type: reduce_on_plateau
    mode: min
    factor: 0.1
    patience: 5
    min_lr: 0.00001

# Training configuration
train:
  num_folds: 3
  batch_size: 32
  num_workers: 4

# Trainer configuration
trainer:
  type: mlp
  epochs: 50
  log_interval: 5
```

## Extending the System

### Adding a New Model

1. Create a new model implementation in the `models` directory
2. Register the model in `models/model_factory.py`
3. Create a corresponding trainer if necessary

### Adding a New Loss Function

1. Create a new loss implementation in the `losses` directory
2. Register the loss in `losses/loss_factory.py`

### Adding a New Dataset

1. Create a new dataset implementation in the `data/datasets` directory
2. Register the dataset in `main.py` using `data_factory.register_dataset()`

### Adding a New Hook

1. Create a new hook implementation in the `hooks` directory
2. Add the hook in `setup_hooks()` function in `main.py`

## Requirements

- Python 3.7+
- PyTorch 1.7+
- NumPy
- PyYAML

## License

This project is licensed under the MIT License. 
