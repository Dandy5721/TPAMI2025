from typing import Dict, Any, Type

import torch
import torch.nn as nn
from interfaces import ModelProtocol, ConfigDict
from models.mlp import MLP, MLPConfig
from models.ode import GraphNeuralODE, ODEConfig
from models.gcn import GCN, GCNConfig
from models.grand import GRAND, GRANDConfig
from models.gread import GREAD, GREADConfig
from models.gcnii import GCNII, GCNIIConfig
from models.ridge import RidgeModel, RidgeConfig
from models.mfg_4ad import MFG4ADGenerator, MFG4ADGeneratorConfig, MFG4ADCritic, MFG4ADCriticConfig
from models.deep_symbolic import SymbolicNetL0, SymbolicNetConfig
from models.gan import Generator, GeneratorConfig, Discriminator, DiscriminatorConfig

class ModelFactory:
    """Factory for creating models from config."""

    def __init__(self):
        self._model_registry = {
            "mlp": self._create_mlp,
            "ode": self._create_ode,
            "gcn": self._create_gcn,
            "grand": self._create_grand,
            "gread": self._create_gread,
            "gcnii": self._create_gcnii,
            "ridge": self._create_ridge,
            "mfg4ad_generator": self._create_mfg4ad_generator,
            "mfg4ad_critic": self._create_mfg4ad_critic,
            "symbolic_net": self._create_symbolic_net,
            "raw_generator": self._create_raw_generator,
            "raw_discriminator": self._create_raw_discriminator,
        }
    

    def create(self, config: ConfigDict) -> ModelProtocol:
        """Create model from config."""
        model_type = config.get("type")

        print(f"Creating model: {model_type}")
        if model_type not in self._model_registry:
            raise ValueError(f"Unknown model type: {model_type}")
        
        return self._model_registry[model_type](config)
    
    def _create_mlp(self, config: ConfigDict) -> MLP:
        """Create MLP model from config."""
        mlp_config = MLPConfig(
            input_dim=config.get("input_dim"),
            hidden_dims=config.get("hidden_dims", []),
            output_dim=config.get("output_dim"),
            activation=config.get("activation", "relu"),
            dropout=config.get("dropout", 0.0),
            batch_norm=config.get("batch_norm", False),
        )
        return MLP(mlp_config)
    
    def _create_ode(self, config: ConfigDict) -> GraphNeuralODE:
        """Create ODE model from config."""
        ode_config = ODEConfig(
            input_dim=config.get("input_dim"),
            hidden_dim=config.get("hidden_dim"),
            output_dim=config.get("output_dim"),
        )
        return GraphNeuralODE(ode_config)
    

    

    

    
    def _create_gcn(self, config: ConfigDict) -> GCN:
        """Create GCN model from config."""
        gcn_config = GCNConfig(
            input_dim=config.get("input_dim"),
            hidden_dim=config.get("hidden_dim", 16),
            output_dim=config.get("output_dim", 1),
        )
        return GCN(gcn_config) 
    
    def _create_grand(self, config: ConfigDict) -> GRAND:
        """Create GRAND model from config."""
        grand_config = GRANDConfig(
            input_dim=config.get("input_dim"),
            hidden_dim=config.get("hidden_dim", 16),
            output_dim=config.get("output_dim", 1),
            n_layers=config.get("n_layers", 30),
            dropout=config.get("dropout", 0.5),
            activation=config.get("activation", "relu"),
            layer_norm=config.get("layer_norm", False),
            residual=config.get("residual", False),
            feat_norm=config.get("feat_norm", None),
        )

        return GRAND(grand_config)
    
    def _create_gread(self, config: ConfigDict) -> GREAD:
        """Create GREAD model from config."""

        gread_config = GREADConfig(
            input_dim=config.get("input_dim", 1),
            hidden_dim=config.get("hidden_dim", 16),
            output_dim=config.get("output_dim", 1),
            device= config.get("device", "cuda" if torch.cuda.is_available() else "cpu"),
        )
        return GREAD(gread_config)
    
    def _create_gcnii(self, config: ConfigDict) -> GCNII:
        """Create GCNII model from config."""
        gcnii_config = GCNIIConfig(
            input_dim=config.get("input_dim"),
            hidden_dim=config.get("hidden_dim", 16),
            output_dim=config.get("output_dim", 1),
            num_layers=config.get("num_layers", 3),
            alpha=config.get("alpha", 0.1),
            lambda_val=config.get("lambda_val", 0.5),
            dropout=config.get("dropout", 0.5),
            device= config.get("device", "cuda" if torch.cuda.is_available() else "cpu"),
        )
        return GCNII(gcnii_config)
    
    def _create_ridge(self, config: ConfigDict) -> RidgeModel:
        """Create Ridge model from config."""
        ridge_config = RidgeConfig(
            alpha=config.get("alpha", 1.0),
            device= config.get("device", "cuda" if torch.cuda.is_available() else "cpu"),
        )
        return RidgeModel(ridge_config)

    def _create_mfg4ad_generator(self, config: ConfigDict) -> MFG4ADGenerator:
        """Create MFG4AD generator from config."""
        mfg4ad_config = MFG4ADGeneratorConfig(
            num_nodes=config.get("num_nodes", 163842),
            input_dim=config.get("input_dim", 1),
            dt=config.get("dt", 1.0),
            hidden_gcn=config.get("hidden_gcn", [8]),
            hidden_flow=config.get("hidden_flow", [16, 8]),
            mlp_hidden=config.get("mlp_hidden", [400, 320, 240, 200]),
            ablation=config.get("ablation", "all"),
            device= config.get("device", "cuda" if torch.cuda.is_available() else "cpu"),
        )

        return MFG4ADGenerator(mfg4ad_config)
    
    def _create_mfg4ad_critic(self, config: ConfigDict) -> MFG4ADCritic:
        """Create MFG4AD critic from config."""
        mfg4ad_config = MFG4ADCriticConfig(
            input_dim=config.get("input_dim"),
            hidden_dims=config.get("hidden_dims", [16, 8]),
            mlp_dims=config.get("mlp_dims", [16, 8]),
            device= config.get("device", "cuda" if torch.cuda.is_available() else "cpu"),
        )

        return MFG4ADCritic(mfg4ad_config)
    
    def _create_symbolic_net(self, config: ConfigDict) -> SymbolicNetL0:
        """Create SymbolicNet model from config."""
        return SymbolicNetL0(
            symbolic_depth=config.get("symbolic_depth", 2),
            in_dim=config.get("in_dim", 1)
        )
    
    def _create_raw_generator(self, config: ConfigDict) -> Generator:
        """Create Generator model from config."""
        generator_config = GeneratorConfig(
            input_dim=config.get("input_dim", 1),
            hidden_dims=config.get("hidden_dims", [16, 8]),
            output_dim=config.get("output_dim", 1),
            dropout_rate=config.get("dropout_rate", 0.2),
        )
        return Generator(generator_config)

    def _create_raw_discriminator(self, config: ConfigDict) -> Discriminator:
        """Create Discriminator model from config."""
        discriminator_config = DiscriminatorConfig(
            input_dim=config.get("input_dim", 1),
            hidden_dims=config.get("hidden_dims", [16, 8]),
            dropout_rate=config.get("dropout_rate", 0.2),
        )
        return Discriminator(discriminator_config)

