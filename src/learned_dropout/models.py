from torch import nn
import torch
import torch.nn.functional as F
from typing import Optional

from src.learned_dropout.model_tracker import ResnetTracker
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


class MaskedRMSNorm(RMSNorm):
    """
    RMSNorm that supports width-masking of the feature dimension.
    If a mask is provided, the RMS is computed only over active (mask==1) dims,
    and the output is zeroed over inactive dims to simulate an effective width.
    """
    def __init__(self, dim, eps=1e-6, has_parameters=True):
        super().__init__(dim, eps=eps, has_parameters=has_parameters)

    def forward(self, x, mask: Optional[torch.Tensor] = None):
        if mask is None:
            return super().forward(x)
        # Compute mean square over active dims only
        # mask shape: [d]; broadcast to batch
        mask_b = mask.unsqueeze(0)
        # active per-sample (broadcasted); shape [batch, 1]
        active = mask_b.sum(dim=-1, keepdim=True)
        mean_sq = ((x ** 2) * mask_b).sum(dim=-1, keepdim=True) / active
        rms = torch.sqrt(mean_sq + self.eps)
        out = (x / rms) * self.weight
        # Zero-out inactive dims to enforce effective width
        out = out * mask_b
        return out


class ResidualBlock(nn.Module):
    def __init__(self, d, h):
        """
        d: Dimension of the residual stream.
        h: Hidden dimension for the block.
        """
        super(ResidualBlock, self).__init__()

        self.rms_norm = RMSNorm(d)

        self.weight_in = nn.Linear(d, h, bias=False)
        self.weight_out = nn.Linear(h, d, bias=False)

    def forward(self, x, width_mask: Optional[torch.Tensor] = None):
        # Base ResidualBlock ignores width_mask
        x = self.rms_norm(x)
        hidden = self.weight_in(x)
        hidden = F.relu(hidden)
        out = self.weight_out(hidden)
        return out


class ResidualBlockH(nn.Module):
    def __init__(self, d, h):
        """
        ResidualBlockH that handles width_mask for the h parameter.
        
        d: Dimension of the residual stream.
        h: Hidden dimension for the block.
        """
        super(ResidualBlockH, self).__init__()
        self.rms_norm = RMSNorm(d)

        self.weight_in = nn.Linear(d, h, bias=False)
        self.weight_out = nn.Linear(h, d, bias=False)

    def forward(self, x, width_mask: Optional[torch.Tensor] = None):
        x = self.rms_norm(x)
        hidden = self.weight_in(x)
        hidden = F.relu(hidden)
        
        # Apply width_mask to hidden layer if provided
        if width_mask is not None:
            hidden = hidden * width_mask.unsqueeze(0)  # Broadcast across batch dimension
            
        out = self.weight_out(hidden)
        return out


class Resnet(nn.Module):
    def __init__(self, c: Config, block_class=None):
        """
        Base Resnet class that can use different residual block types.
        c: Configuration object containing all model parameters
        block_class: Class to use for residual blocks (defaults to ResidualBlock)
        """
        if block_class is None:
            block_class = ResidualBlock
        super(Resnet, self).__init__()
        self.input_dim = c.d
        self.d_model = c.d if c.d_model is None else c.d_model
        self.down_rank_dim = c.down_rank_dim
        self.l1_final = c.l1_final

        self.input_projection = None
        # Only use if a separate d_model is specified
        if c.d_model is not None:
            # First, project inputs from d -> d_model
            self.input_projection = nn.Linear(self.input_dim, self.d_model, bias=False)
            

        # Residual blocks operate in d_model space
        self.blocks = nn.ModuleList([
            block_class(self.d_model, c.h)
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
        self.final_rms_norm = RMSNorm(final_norm_dim, has_parameters=False)

    @staticmethod
    def get_tracker(track_weights):
        return ResnetTracker(track_weights)
    
    def get_l1_regularization_loss(self):
        """
        Compute L1 regularization loss for the final layer.
        Returns 0 if l1_final is None, otherwise returns l1_final * L1_norm(final_layer_weights).
        """
        if self.l1_final is None:
            return 0.0
        return self.l1_final * torch.sum(torch.abs(self.final_layer.weight))

    def forward(self, x, width_mask: Optional[torch.Tensor] = None):
        """
        Forward pass with optional pre‑norm on each block.
        Passes width_mask to blocks (blocks can choose to ignore it).
        
        Args:
            x: Input tensor
            width_mask: Optional mask tensor passed to blocks
        """
        if self.input_projection is not None:
            # Project inputs to model dimension
            current = self.input_projection(x)
        else:
            current = x

        for block in self.blocks:
            block_out = block(current, width_mask=width_mask)
            current = current + block_out
        
        # Apply optional down-ranking layer
        if self.down_rank_layer is not None:
            current = self.down_rank_layer(current)
        
        # Apply final layer normalization (pre-logit)
        current = self.final_rms_norm(current)
            
        output = self.final_layer(current)
        return output.squeeze(1)


class ResnetH(Resnet):
    def __init__(self, c: Config):
        """
        ResnetH class that uses width_mask to vary the hidden dimension h.
        Uses ResidualBlockH and inherits all behavior from base Resnet class.
        """
        super(ResnetH, self).__init__(c, block_class=ResidualBlockH)


class ResnetDownRankDim(Resnet):
    def __init__(self, c: Config):
        """
        ResnetDownRankDim class that uses width_mask to vary the down_rank_dim.
        The width_mask is applied to the down-ranking layer.
        """
        super(ResnetDownRankDim, self).__init__(c)
        self.final_rms_norm = MaskedRMSNorm(self.down_rank_dim, has_parameters=False)
        
    def forward(self, x, width_mask: Optional[torch.Tensor] = None):
        """
        Forward pass with width_mask applied to the down-ranking layer.
        
        Args:
            x: Input tensor
            width_mask: Optional tensor of shape (down_rank_dim,) containing 0.0 or 1.0 values.
                       Applied to the down-ranking layer output.
        """
        if self.input_projection is not None:
            # Project inputs to model dimension
            current = self.input_projection(x)
        else:
            current = x

        for block in self.blocks:
            block_out = block(current)  # No width_mask applied to blocks
            current = current + block_out
        
        # Apply down-ranking layer (must exist in this model)
        current = self.down_rank_layer(current)
        
        # Apply width_mask to down-ranking layer output
        if width_mask is not None:
            current = current * width_mask.unsqueeze(0)  # Broadcast across batch dimension
        
        # Apply final layer normalization (pre-logit) using mask over down-rank dims
        current = self.final_rms_norm(current, mask=width_mask)
            
        output = self.final_layer(current)
        return output.squeeze(1)


class ResidualBlockDModel(nn.Module):
    def __init__(self, d, h):
        """
        Special ResidualBlock for ResnetDModel that can handle d_model width masking.
        
        d: Maximum dimension of the residual stream.
        h: Hidden dimension for the block.
        """
        super(ResidualBlockDModel, self).__init__()
        # Use MaskedRMSNorm to compute RMS over active d_model* only
        self.masked_rms_norm = MaskedRMSNorm(d)

        self.weight_in = nn.Linear(d, h, bias=False)
        self.weight_out = nn.Linear(h, d, bias=False)

    def forward(self, x, d_model_mask: Optional[torch.Tensor] = None):
        # Normalize using only active d_model* dims; zero-out inactive dims
        x_normed = self.masked_rms_norm(x, mask=d_model_mask)

        hidden = self.weight_in(x_normed)
        hidden = F.relu(hidden)
        out = self.weight_out(hidden)
        
        # Apply masking to output as well
        if d_model_mask is not None:
            out = out * d_model_mask.unsqueeze(0)
            
        return out


class ResnetDModel(Resnet):
    def __init__(self, c: Config):
        """
        ResnetDModel class that uses width_mask to vary the d_model dimension.
        The width_mask is applied to the model dimension throughout the network.
        """
        super(Resnet, self).__init__()  # Skip Resnet.__init__ to customize initialization
        self.input_dim = c.d
        self.d_model = c.d if c.d_model is None else c.d_model
        self.down_rank_dim = c.down_rank_dim
        self.l1_final = c.l1_final

        # We always create an input projection to the maximum d_model, so we can mask
        self.input_projection = nn.Linear(self.input_dim, self.d_model, bias=False)

        # Use special ResidualBlockDModel that can handle d_model masking
        self.blocks = nn.ModuleList([
            ResidualBlockDModel(self.d_model, c.h)
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
        self.final_rms_norm = MaskedRMSNorm(final_norm_dim, has_parameters=False)
        
    def forward(self, x, width_mask: Optional[torch.Tensor] = None):
        """
        Forward pass with width_mask applied to d_model dimensions.
        
        Args:
            x: Input tensor
            width_mask: Optional tensor of shape (d_model,) containing 0.0 or 1.0 values.
                       Applied to d_model dimensions throughout the network.
        """
        # Always project to max d_model and then mask to simulate d_model*
        current = self.input_projection(x)
        # Apply mask to projected input
        if width_mask is not None:
            current = current * width_mask.unsqueeze(0)  # Broadcast across batch

        for block in self.blocks:
            block_out = block(current, d_model_mask=width_mask)
            current = current + block_out
        
        # Apply optional down-ranking layer
        if self.down_rank_layer is not None:
            current = self.down_rank_layer(current)
        
        # Apply final layer normalization (pre-logit)
        # Pass mask only if we are still in d_model space (no down-rank)
        if self.down_rank_layer is None:
            current = self.final_rms_norm(current, mask=width_mask)
        else:
            current = self.final_rms_norm(current)
            
        output = self.final_layer(current)
        return output.squeeze(1)


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
    if c.width_varyer is None:
        return Resnet(c)
    elif c.width_varyer == "h":
        return ResnetH(c)
    elif c.width_varyer == "down_rank_dim":
        return ResnetDownRankDim(c)
    elif c.width_varyer == "d_model":
        return ResnetDModel(c)
    else:
        raise ValueError(f"Invalid width_varyer: {c.width_varyer}. Must be None, 'h', 'down_rank_dim', or 'd_model'.")
