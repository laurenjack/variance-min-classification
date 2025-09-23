from torch import nn
import torch
import torch.nn.functional as F
from typing import Optional

from src.learned_dropout.model_tracker_standard import ResNetStandardTracker
from src.learned_dropout.config import Config


class RMSNorm(nn.Module):
    """
    Custom implementation of RMSNorm
    Root Mean Square Layer Normalization: https://arxiv.org/abs/1910.07467
    """
    def __init__(self, dim, eps=1e-6, has_parameters=True):
        super().__init__()
        self.eps = eps
        self.has_parameters = has_parameters
        if has_parameters:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_buffer('weight', torch.ones(dim))

    def forward(self, x):
        # Calculate RMS
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        # Normalize and scale
        return x / rms * self.weight


def create_norm_layer(layer_norm: str, dim: int, has_parameters: bool = True):
    """
    Create a normalization layer based on the specified type.
    
    Args:
        layer_norm: "layer_norm" | "rms_norm"
        dim: The dimension for the normalization layer
        has_parameters: Whether the normalization layer should have learnable parameters
    
    Returns:
        nn.Module: The appropriate normalization layer
    """
    if layer_norm == "layer_norm":
        if has_parameters:
            return nn.LayerNorm(dim)
        else:
            return nn.LayerNorm(dim, elementwise_affine=False)
    elif layer_norm == "rms_norm":
        return RMSNorm(dim, has_parameters=has_parameters)
    else:
        raise ValueError(f"Invalid layer_norm: {layer_norm}. Must be 'layer_norm' or 'rms_norm'.")


class StandardResidualBlock(nn.Module):
    def __init__(self, d, h, layer_norm: str):
        """
        d: Dimension of the residual stream.
        h: Hidden dimension for the block.
        layer_norm: "layer_norm" | "rms_norm".
            - "layer_norm": apply LayerNorm(d)
            - "rms_norm": apply RMSNorm(d)
        """
        super(StandardResidualBlock, self).__init__()
        # configure normalization
        self.layer_norm = create_norm_layer(layer_norm, d)

        self.weight_in = nn.Linear(d, h, bias=False)
        self.weight_out = nn.Linear(h, d, bias=False)

    def forward(self, x, h_mask: Optional[torch.Tensor] = None):
        x = self.layer_norm(x)
        hidden = self.weight_in(x)
        hidden = F.relu(hidden)
        
        # Apply h_mask to hidden layer if provided
        if h_mask is not None:
            hidden = hidden * h_mask.unsqueeze(0)  # Broadcast across batch dimension
            
        out = self.weight_out(hidden)
        return out


class ResNetStandard(nn.Module):
    def __init__(self, c: Config):
        """
        c: Configuration object containing all model parameters
        """
        super(ResNetStandard, self).__init__()
        self.input_dim = c.d
        self.d_model = c.d if c.d_model is None else c.d_model
        self.down_rank_dim = c.down_rank_dim
        self.layer_norm = c.layer_norm
        self.l1_final = c.l1_final

        self.input_projection = None
        # Only use if a separate d_model is specified
        if c.d_model is not None:
            # First, project inputs from d -> d_model
            self.input_projection = nn.Linear(self.input_dim, self.d_model, bias=False)
            

        # Residual blocks operate in d_model space
        self.blocks = nn.ModuleList([
            StandardResidualBlock(self.d_model, c.h, layer_norm=c.layer_norm)
            for _ in range(c.num_layers)
        ])
        
        # Optionally add the down-ranking layer (in d_model space)
        if c.down_rank_dim is not None:
            self.down_rank_layer = nn.Linear(self.d_model, c.down_rank_dim, bias=False)
            self.final_layer = nn.Linear(c.down_rank_dim, 1, bias=False)
            final_norm_dim = c.down_rank_dim
        else:
            self.down_rank_layer = None
            self.final_layer = nn.Linear(self.d_model, 1, bias=False)
            final_norm_dim = self.d_model
        
        # Add final layer normalization (pre-logit) as in transformers
        # Applied after down-rank layer if it exists
        self.final_layer_norm = create_norm_layer(c.layer_norm, final_norm_dim, has_parameters=False)

    @staticmethod
    def get_tracker(track_weights):
        return ResNetStandardTracker(track_weights)
    
    def get_l1_regularization_loss(self):
        """
        Compute L1 regularization loss for the final layer.
        Returns 0 if l1_final is None, otherwise returns l1_final * L1_norm(final_layer_weights).
        """
        if self.l1_final is None:
            return 0.0
        return self.l1_final * torch.sum(torch.abs(self.final_layer.weight))

    def forward(self, x, h_mask: Optional[torch.Tensor] = None):
        """
        Forward pass with optional pre‑norm on each block:
        Each block optionally normalizes its input, then applies two linear layers
        (with an optional ReLU), and adds the result back into the residual stream.
        Finally, applies optional layer norm before the output projection.
        
        Args:
            x: Input tensor
            h_mask: Optional tensor of shape (h,) containing 0.0 or 1.0 values.
                   Applied to hidden layers in each residual block.
        """
        if self.input_projection is not None:
            # Project inputs to model dimension
            current = self.input_projection(x)
        else:
            current = x

        for block in self.blocks:
            block_out = block(current, h_mask=h_mask)
            current = current + block_out
        
        # Apply optional down-ranking layer
        if self.down_rank_layer is not None:
            current = self.down_rank_layer(current)
        
        # Apply final layer normalization (pre-logit)
        current = self.final_layer_norm(current)
            
        output = self.final_layer(current)
        return output.squeeze(1)
