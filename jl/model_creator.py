from jl.models import Resnet, MLP, MultiLinear, KPolynomial
from jl.parallel_models import (
    ResnetH,
    ResnetDModel,
    MLPH,
)
from jl.config import Config
from jl.feature_experiments.dropout import create_dropout_list
from torch import nn


def create_resnet(c: Config):
    """
    Factory function to create the appropriate Resnet subclass based on the configuration.
    
    Args:
        c: Configuration object containing model parameters including width_varyer
        
    Returns:
        nn.Module: The appropriate Resnet subclass instance
        
    Raises:
        ValueError: If width_varyer is not a recognized value
    """
    # Determine d_model for dropout list
    d_model = c.d if c.d_model is None else c.d_model
    
    # Create list of dropouts, one per layer
    dropouts = create_dropout_list(
        num_layers=c.num_layers,
        dropout_prob=c.dropout_prob,
        is_hashed_dropout=c.is_hashed_dropout,
        d_model=d_model,
    )
    
    if c.width_varyer is None:
        return Resnet(c, dropouts=dropouts)
    elif c.width_varyer == "h":
        return ResnetH(c, dropouts=dropouts)
    elif c.width_varyer == "d_model":
        return ResnetDModel(c, dropouts=dropouts)
    else:
        raise ValueError(f"Invalid width_varyer: {c.width_varyer}. Must be None, 'h', or 'd_model'.")


def create_mlp(c: Config):
    """
    Factory function to create the appropriate MLP subclass based on the configuration.
    
    Args:
        c: Configuration object containing model parameters including width_varyer
        
    Returns:
        nn.Module: The appropriate MLP subclass instance
        
    Raises:
        ValueError: If width_varyer is not a recognized value or if num_layers=0 with non-None width_varyer
    """
    # Validate that num_layers=0 is only supported with width_varyer=None
    if c.num_layers == 0 and c.width_varyer is not None:
        raise ValueError(f"num_layers=0 is only supported when width_varyer=None. Got width_varyer={c.width_varyer}")
    
    # Create list of dropouts, one per hidden layer (skip first layer, so num_layers - 1)
    # MLP doesn't support hashed dropout
    dropouts = create_dropout_list(
        num_layers=max(0, c.num_layers - 1),
        dropout_prob=c.dropout_prob,
        is_hashed_dropout=False,
        d_model=None,
    )
    
    if c.width_varyer is None:
        return MLP(c, dropouts=dropouts)
    elif c.width_varyer == "h":
        return MLPH(c, dropouts=dropouts)
    else:
        raise ValueError(f"Invalid width_varyer for MLP: {c.width_varyer}. Must be None or 'h'.")


def create_model(c: Config):
    """
    Factory function to create the appropriate model based on the configuration.
    
    Args:
        c: Configuration object containing model parameters including model_type and width_varyer
        
    Returns:
        nn.Module: The appropriate model instance (Resnet, MLP, MultiLinear, or KPolynomial variant)
        
    Raises:
        ValueError: If model_type or width_varyer is not a recognized value
    """
    if c.model_type == 'resnet':
        return create_resnet(c)
    elif c.model_type == 'mlp':
        return create_mlp(c)
    elif c.model_type == 'multi-linear':
        # Create list of dropouts, one per hidden layer (skip first layer, so num_layers - 1)
        # MultiLinear doesn't support hashed dropout
        dropouts = create_dropout_list(
            num_layers=max(0, c.num_layers - 1),
            dropout_prob=c.dropout_prob,
            is_hashed_dropout=False,
            d_model=None,
        )
        return MultiLinear(c, dropouts=dropouts)
    elif c.model_type == 'k-polynomial':
        return KPolynomial(c)
    else:
        raise ValueError(f"Invalid model_type: {c.model_type}. Must be 'resnet', 'mlp', 'multi-linear', or 'k-polynomial'.")

