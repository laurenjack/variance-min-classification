from torch import nn
import torch
import torch.nn.functional as F
from typing import Optional

from jl.model_tracker import ResnetTracker
from jl.model_tracker import MLPTracker
from jl.config import Config


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
    def __init__(self, d, h, is_norm=True):
        """
        d: Dimension of the residual stream.
        h: Hidden dimension for the block.
        is_norm: Whether to apply RMSNorm (default True).
        """
        super(ResidualBlock, self).__init__()

        self.is_norm = is_norm
        if is_norm:
            self.rms_norm = RMSNorm(d)
        else:
            self.rms_norm = None

        self.weight_in = nn.Linear(d, h, bias=False)
        self.weight_out = nn.Linear(h, d, bias=False)

    def forward(self, x, width_mask: Optional[torch.Tensor] = None):
        # Base ResidualBlock ignores width_mask
        if self.is_norm:
            x = self.rms_norm(x)
        hidden = self.weight_in(x)
        hidden = F.relu(hidden)
        out = self.weight_out(hidden)
        return out


class ResidualBlockH(nn.Module):
    def __init__(self, d, h, is_norm=True):
        """
        ResidualBlockH that handles width_mask for the h parameter.
        
        d: Dimension of the residual stream.
        h: Hidden dimension for the block.
        is_norm: Whether to apply RMSNorm (default True).
        """
        super(ResidualBlockH, self).__init__()
        self.is_norm = is_norm
        if is_norm:
            self.rms_norm = RMSNorm(d)
        else:
            self.rms_norm = None

        self.weight_in = nn.Linear(d, h, bias=False)
        self.weight_out = nn.Linear(h, d, bias=False)

    def forward(self, x, width_mask: Optional[torch.Tensor] = None):
        if self.is_norm:
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

        self.input_projection = None
        # Only use if a separate d_model is specified
        if c.d_model is not None:
            # First, project inputs from d -> d_model
            self.input_projection = nn.Linear(self.input_dim, self.d_model, bias=False)
            

        # Residual blocks operate in d_model space
        self.blocks = nn.ModuleList([
            block_class(self.d_model, c.h, is_norm=c.is_norm)
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
        self.is_norm = c.is_norm
        if c.is_norm:
            self.final_rms_norm = RMSNorm(final_norm_dim, has_parameters=False)
        else:
            self.final_rms_norm = None

    @staticmethod
    def get_tracker(track_weights):
        return ResnetTracker(track_weights)

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
        if self.is_norm:
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
        if c.is_norm:
            self.final_rms_norm = MaskedRMSNorm(self.down_rank_dim, has_parameters=False)
        else:
            self.final_rms_norm = None
        
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
        if self.is_norm:
            current = self.final_rms_norm(current, mask=width_mask)
            
        output = self.final_layer(current)
        return output.squeeze(1)


class ResidualBlockDModel(nn.Module):
    def __init__(self, d, h, is_norm=True):
        """
        Special ResidualBlock for ResnetDModel that can handle d_model width masking.
        
        d: Maximum dimension of the residual stream.
        h: Hidden dimension for the block.
        is_norm: Whether to apply RMSNorm (default True).
        """
        super(ResidualBlockDModel, self).__init__()
        # Use MaskedRMSNorm to compute RMS over active d_model* only
        self.is_norm = is_norm
        if is_norm:
            self.masked_rms_norm = MaskedRMSNorm(d)
        else:
            self.masked_rms_norm = None

        self.weight_in = nn.Linear(d, h, bias=False)
        self.weight_out = nn.Linear(h, d, bias=False)

    def forward(self, x, d_model_mask: Optional[torch.Tensor] = None):
        # Normalize using only active d_model* dims; zero-out inactive dims
        if self.is_norm:
            x_normed = self.masked_rms_norm(x, mask=d_model_mask)
        else:
            x_normed = x

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

        # We always create an input projection to the maximum d_model, so we can mask
        self.input_projection = nn.Linear(self.input_dim, self.d_model, bias=False)

        # Use special ResidualBlockDModel that can handle d_model masking
        self.blocks = nn.ModuleList([
            ResidualBlockDModel(self.d_model, c.h, is_norm=c.is_norm)
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
        self.is_norm = c.is_norm
        if c.is_norm:
            self.final_rms_norm = MaskedRMSNorm(final_norm_dim, has_parameters=False)
        else:
            self.final_rms_norm = None
        
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
        if self.is_norm:
            if self.down_rank_layer is None:
                current = self.final_rms_norm(current, mask=width_mask)
            else:
                current = self.final_rms_norm(current)
            
        output = self.final_layer(current)
        return output.squeeze(1)


class MLP(nn.Module):
    def __init__(self, c: Config):
        """
        Multi-Layer Perceptron (MLP) without residual connections.
        
        Args:
            c: Configuration object containing model parameters
        """
        super(MLP, self).__init__()
        self.input_dim = c.d
        self.d_model = c.d_model if c.d_model is not None else c.d
        self.down_rank_dim = c.down_rank_dim
        self.num_layers = c.num_layers
        self.is_norm = c.is_norm
        
        # Build MLP layers
        layers = []
        
        # Input layer (only add if num_layers > 0, otherwise just identity)
        if self.num_layers > 0:
            layers.append(nn.Linear(self.input_dim, self.d_model, bias=False))
            if c.is_norm:
                layers.append(RMSNorm(self.d_model))
            layers.append(nn.ReLU())
            
            # Hidden layers
            for _ in range(self.num_layers - 1):
                layers.append(nn.Linear(self.d_model, self.d_model, bias=False))
                if c.is_norm:
                    layers.append(RMSNorm(self.d_model))
                layers.append(nn.ReLU())
        
        self.layers = nn.Sequential(*layers) if layers else nn.Identity()
        
        # Optionally add the down-ranking layer
        if c.down_rank_dim is not None:
            # When num_layers=0, down_rank takes input directly from input_dim
            down_rank_input = self.input_dim if self.num_layers == 0 else self.d_model
            self.down_rank_layer = nn.Linear(down_rank_input, c.down_rank_dim, bias=False)
            self.final_layer = nn.Linear(c.down_rank_dim, 1, bias=False)
            final_norm_dim = c.down_rank_dim
        else:
            self.down_rank_layer = None
            # When num_layers=0, final layer takes input directly from input_dim
            final_input = self.input_dim if self.num_layers == 0 else self.d_model
            self.final_layer = nn.Linear(final_input, 1, bias=False)
            final_norm_dim = final_input
        
        # Add final layer normalization (pre-logit)
        if c.is_norm:
            self.final_rms_norm = RMSNorm(final_norm_dim, has_parameters=False)
        else:
            self.final_rms_norm = None
    
    @staticmethod
    def get_tracker(track_weights):
        return MLPTracker(track_weights)
    
    def forward(self, x, width_mask: Optional[torch.Tensor] = None):
        """
        Forward pass through MLP.
        Base MLP ignores width_mask.
        
        Args:
            x: Input tensor
            width_mask: Optional mask tensor (ignored in base MLP)
        """
        current = self.layers(x)
        
        # Apply optional down-ranking layer
        if self.down_rank_layer is not None:
            current = self.down_rank_layer(current)
        
        # Apply final layer normalization (pre-logit)
        if self.is_norm:
            current = self.final_rms_norm(current)
        
        output = self.final_layer(current)
        return output.squeeze(1)


class MLPDownRankDim(nn.Module):
    def __init__(self, c: Config):
        """
        MLP that uses width_mask to vary the down_rank_dim.
        The width_mask is applied to the down-ranking layer.
        """
        super(MLPDownRankDim, self).__init__()
        self.input_dim = c.d
        self.d_model = c.d_model if c.d_model is not None else c.d
        self.down_rank_dim = c.down_rank_dim
        self.num_layers = c.num_layers
        self.is_norm = c.is_norm
        
        # Build MLP layers
        layers = []
        
        # Input layer
        layers.append(nn.Linear(self.input_dim, self.d_model, bias=False))
        if c.is_norm:
            layers.append(RMSNorm(self.d_model))
        layers.append(nn.ReLU())
        
        # Hidden layers
        for _ in range(self.num_layers - 1):
            layers.append(nn.Linear(self.d_model, self.d_model, bias=False))
            if c.is_norm:
                layers.append(RMSNorm(self.d_model))
            layers.append(nn.ReLU())
        
        self.layers = nn.Sequential(*layers)
        
        # Down-ranking layer (must exist in this model)
        self.down_rank_layer = nn.Linear(self.d_model, c.down_rank_dim, bias=False)
        self.final_layer = nn.Linear(c.down_rank_dim, 1, bias=False)
        
        # Add final layer normalization (pre-logit) using mask
        if c.is_norm:
            self.final_rms_norm = MaskedRMSNorm(c.down_rank_dim, has_parameters=False)
        else:
            self.final_rms_norm = None
    
    def forward(self, x, width_mask: Optional[torch.Tensor] = None):
        """
        Forward pass with width_mask applied to the down-ranking layer.
        
        Args:
            x: Input tensor
            width_mask: Optional tensor of shape (down_rank_dim,) containing 0.0 or 1.0 values.
                       Applied to the down-ranking layer output.
        """
        current = self.layers(x)
        
        # Apply down-ranking layer
        current = self.down_rank_layer(current)
        
        # Apply width_mask to down-ranking layer output
        if width_mask is not None:
            current = current * width_mask.unsqueeze(0)  # Broadcast across batch dimension
        
        # Apply final layer normalization (pre-logit) using mask
        if self.is_norm:
            current = self.final_rms_norm(current, mask=width_mask)
        
        output = self.final_layer(current)
        return output.squeeze(1)


class MLPDModel(nn.Module):
    def __init__(self, c: Config):
        """
        MLP that uses width_mask to vary the d_model dimension.
        The width_mask is applied to the model dimension throughout the network.
        """
        super(MLPDModel, self).__init__()
        self.input_dim = c.d
        self.d_model = c.d_model if c.d_model is not None else c.d
        self.down_rank_dim = c.down_rank_dim
        self.num_layers = c.num_layers
        self.is_norm = c.is_norm
        
        # Build MLP layers manually to support masking
        self.input_layer = nn.Linear(self.input_dim, self.d_model, bias=False)
        if c.is_norm:
            self.input_norm = MaskedRMSNorm(self.d_model)
        else:
            self.input_norm = None
        
        # Hidden layers
        self.hidden_layers = nn.ModuleList()
        self.hidden_norms = nn.ModuleList()
        for _ in range(self.num_layers - 1):
            self.hidden_layers.append(nn.Linear(self.d_model, self.d_model, bias=False))
            if c.is_norm:
                self.hidden_norms.append(MaskedRMSNorm(self.d_model))
            else:
                self.hidden_norms.append(None)
        
        # Optionally add the down-ranking layer
        if c.down_rank_dim is not None:
            self.down_rank_layer = nn.Linear(self.d_model, c.down_rank_dim, bias=False)
            self.final_layer = nn.Linear(c.down_rank_dim, 1, bias=False)
            final_norm_dim = c.down_rank_dim
        else:
            self.down_rank_layer = None
            self.final_layer = nn.Linear(self.d_model, 1, bias=False)
            final_norm_dim = self.d_model
        
        # Add final layer normalization (pre-logit)
        if c.is_norm:
            self.final_rms_norm = MaskedRMSNorm(final_norm_dim, has_parameters=False)
        else:
            self.final_rms_norm = None
    
    def forward(self, x, width_mask: Optional[torch.Tensor] = None):
        """
        Forward pass with width_mask applied to d_model dimensions.
        
        Args:
            x: Input tensor
            width_mask: Optional tensor of shape (d_model,) containing 0.0 or 1.0 values.
                       Applied to d_model dimensions throughout the network.
        """
        # Input layer with masking
        current = self.input_layer(x)
        if width_mask is not None:
            current = current * width_mask.unsqueeze(0)
        if self.is_norm:
            current = self.input_norm(current, mask=width_mask)
        current = F.relu(current)
        
        # Hidden layers with masking
        for hidden_layer, hidden_norm in zip(self.hidden_layers, self.hidden_norms):
            current = hidden_layer(current)
            if width_mask is not None:
                current = current * width_mask.unsqueeze(0)
            if self.is_norm:
                current = hidden_norm(current, mask=width_mask)
            current = F.relu(current)
        
        # Apply optional down-ranking layer
        if self.down_rank_layer is not None:
            current = self.down_rank_layer(current)
            # No mask after down-rank
            if self.is_norm:
                current = self.final_rms_norm(current)
        else:
            # Apply final layer normalization with mask if still in d_model space
            if self.is_norm:
                current = self.final_rms_norm(current, mask=width_mask)
        
        output = self.final_layer(current)
        return output.squeeze(1)


class KPolynomial(nn.Module):
    def __init__(self, c: Config):
        """
        K-degree polynomial model with learned coefficients for independent polynomials per dimension.
        
        Computes: logit = Σᵢ Σⱼ₌₁ᵏ wᵢⱼ * xᵢʲ
        
        Args:
            c: Configuration object containing model parameters
        """
        super(KPolynomial, self).__init__()
        self.input_dim = c.d
        self.k = c.k
        
        # Initialize coefficient matrix of shape (d, k) for learning wᵢⱼ coefficients
        # Each row i contains coefficients for dimension i: [wᵢ₁, wᵢ₂, ..., wᵢₖ]
        self.coefficients = nn.Parameter(torch.randn(self.input_dim, self.k) * 0.01)
    
    @staticmethod
    def get_tracker(track_weights):
        from jl.model_tracker import PolynomialTracker
        return PolynomialTracker(track_weights)
    
    def forward(self, x, width_mask: Optional[torch.Tensor] = None):
        """
        Forward pass through polynomial model.
        
        Args:
            x: Input tensor of shape (batch_size, d)
            width_mask: Optional mask tensor (ignored in polynomial model)
        
        Returns:
            Logits of shape (batch_size,)
        """
        # x has shape (batch_size, d)
        batch_size, d = x.shape
        
        # Compute polynomial features: [x, x², x³, ..., xᵏ]
        # powers will have shape (batch_size, d, k)
        powers = torch.stack([x ** (j + 1) for j in range(self.k)], dim=2)
        
        # Apply learned coefficients: element-wise multiply and sum
        # coefficients has shape (d, k), unsqueeze to (1, d, k) for broadcasting
        # Result: (batch_size, d, k) * (1, d, k) -> (batch_size, d, k)
        weighted = powers * self.coefficients.unsqueeze(0)
        
        # Sum over dimensions d and polynomial degrees k to get scalar logit per sample
        logits = weighted.sum(dim=(1, 2))  # Shape: (batch_size,)
        
        return logits


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
    
    if c.width_varyer is None:
        return MLP(c)
    elif c.width_varyer == "down_rank_dim":
        return MLPDownRankDim(c)
    elif c.width_varyer == "d_model":
        return MLPDModel(c)
    else:
        raise ValueError(f"Invalid width_varyer for MLP: {c.width_varyer}. Must be None, 'down_rank_dim', or 'd_model'.")


def create_model(c: Config):
    """
    Factory function to create the appropriate model based on the configuration.
    
    Args:
        c: Configuration object containing model parameters including model_type and width_varyer
        
    Returns:
        nn.Module: The appropriate model instance (Resnet, MLP, or KPolynomial variant)
        
    Raises:
        ValueError: If model_type or width_varyer is not a recognized value
    """
    if c.model_type == 'resnet':
        return create_resnet(c)
    elif c.model_type == 'mlp':
        return create_mlp(c)
    elif c.model_type == 'k-polynomial':
        return KPolynomial(c)
    else:
        raise ValueError(f"Invalid model_type: {c.model_type}. Must be 'resnet', 'mlp', or 'k-polynomial'.")
