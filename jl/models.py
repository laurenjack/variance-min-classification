from torch import nn
import torch
import torch.nn.functional as F
from typing import Optional

from jl.model_tracker import TrackerInterface
from jl.model_tracker import ResnetTracker
from jl.model_tracker import MLPTracker
from jl.model_tracker import MultiLinearTracker
from jl.model_tracker import PolynomialTracker
from jl.config import Config
from jl.feature_experiments.dropout import Dropout, DropoutModules
from jl.feature_experiments.scaled_regularization import ScaledRegLinear


def _make_linear(in_features, out_features, scaled_reg_k):
    """Create nn.Linear or ScaledRegLinear based on whether scaled_reg_k is set."""
    if scaled_reg_k is not None:
        return ScaledRegLinear(in_features, out_features, scaled_reg_k)
    return nn.Linear(in_features, out_features, bias=False)



class RMSNorm(nn.Module):
    """
    Custom implementation of RMSNorm
    Root Mean Square Layer Normalization: https://arxiv.org/abs/1910.07467
    """
    def __init__(self, dim, eps=1e-6, learnable_norm_parameters=True):
        super().__init__()
        self.eps = eps
        self.learnable_norm_parameters = learnable_norm_parameters
        if learnable_norm_parameters:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_buffer('weight', torch.ones(dim))

    def forward(self, x):
        # Calculate RMS
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        # Normalize and scale
        return x / rms * self.weight


class ResidualBlock(nn.Module):
    def __init__(self, d, h, is_norm=True, learnable_norm_parameters=True, scaled_reg_k=None):
        """
        d: Dimension of the residual stream.
        h: Hidden dimension for the block.
        is_norm: Whether to apply RMSNorm (default True).
        learnable_norm_parameters: Whether RMSNorm weights are learnable.
        scaled_reg_k: If set, use ScaledRegLinear instead of nn.Linear.
        """
        super(ResidualBlock, self).__init__()

        self.is_norm = is_norm
        if is_norm:
            self.rms_norm = RMSNorm(d, learnable_norm_parameters=learnable_norm_parameters)
        else:
            self.rms_norm = None

        self.weight_in = _make_linear(d, h, scaled_reg_k)
        self.weight_out = _make_linear(h, d, scaled_reg_k)

    def forward(self, x, width_mask: Optional[torch.Tensor] = None):
        # Base ResidualBlock ignores width_mask
        if self.is_norm:
            x = self.rms_norm(x)
        hidden = self.weight_in(x)
        hidden = F.relu(hidden)
        out = self.weight_out(hidden)
        return out


class Resnet(nn.Module):
    def __init__(self, c: Config, dropout_modules: DropoutModules, block_class=None):
        """
        Base Resnet class that can use different residual block types.
        c: Configuration object containing all model parameters
        dropout_modules: DropoutModules container with dropouts for blocks and final layer
        block_class: Class to use for residual blocks (defaults to ResidualBlock)
        """
        if block_class is None:
            block_class = ResidualBlock
        super(Resnet, self).__init__()
        self.input_dim = c.d
        self.d_model = c.d if c.d_model is None else c.d_model
        self.num_class = c.num_class
        
        # Determine output dimension: 1 for binary, num_class for multi-class
        output_dim = 1 if c.num_class == 2 else c.num_class

        self.input_projection = None
        # Only use if a separate d_model is specified
        if c.d_model is not None:
            # First, project inputs from d -> d_model
            self.input_projection = _make_linear(self.input_dim, self.d_model, c.scaled_reg_k)


        # Residual blocks operate in d_model space
        self.blocks = nn.ModuleList([
            block_class(self.d_model, c.h, is_norm=c.is_norm, learnable_norm_parameters=c.learnable_norm_parameters, scaled_reg_k=c.scaled_reg_k)
            for _ in range(c.num_layers)
        ])

        # Final layer connects directly from d_model to output
        self.final_layer = _make_linear(self.d_model, output_dim, c.scaled_reg_k)
        
        # Add final layer normalization (pre-logit) as in transformers
        self.is_norm = c.is_norm
        if c.is_norm:
            self.final_rms_norm = RMSNorm(self.d_model, learnable_norm_parameters=c.learnable_norm_parameters)
        else:
            self.final_rms_norm = None
    
        # Convert dropout list to ModuleList and store final dropout separately
        self.dropouts = nn.ModuleList(dropout_modules.dropouts)
        self.dropout_final = dropout_modules.dropout_final

    def get_tracker(self, c: Config):
        if c.weight_tracker is None:
            return TrackerInterface()
        return ResnetTracker(
            c=c,
            num_layers=len(self.blocks)
        )

    def forward(self, x, x_indices: Optional[torch.Tensor] = None, width_mask: Optional[torch.Tensor] = None):
        """
        Forward pass with optional pre‑norm on each block.
        Passes width_mask to blocks (blocks can choose to ignore it).
        
        Args:
            x: Input tensor
            x_indices: Optional tensor of point indices for hashed dropout
            width_mask: Optional mask tensor passed to blocks
        """
        if self.input_projection is not None:
            # Project inputs to model dimension
            current = self.input_projection(x)
        else:
            current = x

        for block, dropout in zip(self.blocks, self.dropouts):
            block_out = block(current, width_mask=width_mask)
            # Apply dropout to block output before adding to residual stream
            block_out = dropout(block_out, x_indices)
            current = current + block_out
        
        # Apply final layer normalization (pre-logit)
        if self.is_norm:
            current = self.final_rms_norm(current)
        
        # Apply dropout before final layer (pre-logits)
        current = self.dropout_final(current, x_indices)
            
        output = self.final_layer(current)
        # Squeeze only for binary classification (output dim is 1)
        if self.num_class == 2:
            output = output.squeeze(1)
        return output


class MLP(nn.Module):
    def __init__(self, c: Config, dropout_modules: DropoutModules):
        """
        Multi-Layer Perceptron (MLP) without residual connections.
        Uses pre-norm architecture: Norm → Linear → ReLU → Linear per hidden layer.
        
        Args:
            c: Configuration object containing model parameters
            dropout_modules: DropoutModules container with dropouts for hidden layers and final layer
        """
        super(MLP, self).__init__()
        self.input_dim = c.d
        self.h = c.h if c.h is not None else c.d
        self.num_layers = c.num_layers
        self.is_norm = c.is_norm
        self.num_class = c.num_class
        
        # Determine output dimension: 1 for binary, num_class for multi-class
        output_dim = 1 if c.num_class == 2 else c.num_class
        
        # Build MLP hidden layers manually to track each component
        # Each hidden layer: Norm → Linear → ReLU → Linear
        self.hidden_norms = nn.ModuleList()
        self.hidden_linear1 = nn.ModuleList()
        self.hidden_linear2 = nn.ModuleList()
        
        for i in range(self.num_layers):
            # Norm
            if c.is_norm:
                norm_dim = self.input_dim if i == 0 else self.h
                self.hidden_norms.append(RMSNorm(norm_dim, learnable_norm_parameters=c.learnable_norm_parameters))
            else:
                self.hidden_norms.append(None)

            # Linear 1: input_dim→h for first layer, h→h for rest
            if i == 0:
                self.hidden_linear1.append(_make_linear(self.input_dim, self.h, c.scaled_reg_k))
            else:
                self.hidden_linear1.append(_make_linear(self.h, self.h, c.scaled_reg_k))

            # Linear 2: h→h
            self.hidden_linear2.append(_make_linear(self.h, self.h, c.scaled_reg_k))

        # Final layer connects directly from h (or input_dim if num_layers=0) to output
        final_input = self.input_dim if self.num_layers == 0 else self.h
        self.final_layer = _make_linear(final_input, output_dim, c.scaled_reg_k)
        
        # Add final layer normalization (pre-logit)
        if c.is_norm:
            self.final_rms_norm = RMSNorm(final_input, learnable_norm_parameters=c.learnable_norm_parameters)
        else:
            self.final_rms_norm = None
        
        # Convert dropout list to ModuleList and store final dropout separately
        self.dropouts = nn.ModuleList(dropout_modules.dropouts)
        self.dropout_final = dropout_modules.dropout_final
    
    def get_tracker(self, c: Config):
        if c.weight_tracker is None:
            return TrackerInterface()
        return MLPTracker(
            c=c,
            num_layers=self.num_layers
        )
    
    def forward(self, x, width_mask: Optional[torch.Tensor] = None):
        """
        Forward pass through MLP.
        Base MLP ignores width_mask.
        
        Args:
            x: Input tensor
            width_mask: Optional mask tensor (ignored in base MLP)
        """
        current = x
        
        # Process hidden layers: Norm → Linear → ReLU → Linear
        for i in range(self.num_layers):
            # Apply norm
            if self.is_norm and self.hidden_norms[i] is not None:
                current = self.hidden_norms[i](current)
            
            # Apply first linear
            current = self.hidden_linear1[i](current)
            
            # Apply ReLU
            current = F.relu(current)
            
            # Apply second linear
            current = self.hidden_linear2[i](current)
            
            # Apply dropout after hidden_linear2 (skip first layer)
            if i > 0:
                current = self.dropouts[i - 1](current)
        
        # Apply final norm
        if self.is_norm and self.final_rms_norm is not None:
            current = self.final_rms_norm(current)
        
        # Apply dropout before final layer (pre-logits)
        current = self.dropout_final(current)
        
        output = self.final_layer(current)
        # Squeeze only for binary classification (output dim is 1)
        if self.num_class == 2:
            output = output.squeeze(1)
        return output


class MultiLinear(nn.Module):
    def __init__(self, c: Config, dropout_modules: DropoutModules):
        """
        Multi-Linear model without activations (no ReLU).
        Just a stack of linear layers with optional normalization.
        
        Args:
            c: Configuration object containing model parameters
            dropout_modules: DropoutModules container with dropouts for hidden layers and final layer
        """
        super(MultiLinear, self).__init__()
        self.input_dim = c.d
        self.h = c.h if c.h is not None else c.d
        self.num_layers = c.num_layers
        self.is_norm = c.is_norm
        self.num_class = c.num_class
        
        # Determine output dimension: 1 for binary, num_class for multi-class
        output_dim = 1 if c.num_class == 2 else c.num_class
        
        # Build MultiLinear layers manually to support dropout (no activations)
        # Using pre-norm architecture: Norm → Linear → Norm → Linear → ...
        self.hidden_norms = nn.ModuleList()
        self.hidden_linears = nn.ModuleList()
        
        # Input layer (only add if num_layers > 0)
        if self.num_layers > 0:
            if c.is_norm:
                self.hidden_norms.append(RMSNorm(self.input_dim, learnable_norm_parameters=c.learnable_norm_parameters))
            else:
                self.hidden_norms.append(None)
            self.hidden_linears.append(_make_linear(self.input_dim, self.h, c.scaled_reg_k))

            # Hidden layers (no ReLU)
            for _ in range(self.num_layers - 1):
                if c.is_norm:
                    self.hidden_norms.append(RMSNorm(self.h, learnable_norm_parameters=c.learnable_norm_parameters))
                else:
                    self.hidden_norms.append(None)
                self.hidden_linears.append(_make_linear(self.h, self.h, c.scaled_reg_k))

        # Final layer connects directly from h (or input_dim if num_layers=0) to output
        final_input = self.input_dim if self.num_layers == 0 else self.h
        self.final_layer = _make_linear(final_input, output_dim, c.scaled_reg_k)
        
        # Add final layer normalization (pre-logit)
        if c.is_norm:
            self.final_rms_norm = RMSNorm(final_input, learnable_norm_parameters=c.learnable_norm_parameters)
        else:
            self.final_rms_norm = None
    
        # Convert dropout list to ModuleList and store final dropout separately
        self.dropouts = nn.ModuleList(dropout_modules.dropouts)
        self.dropout_final = dropout_modules.dropout_final
    
    def get_tracker(self, c: Config):
        if c.weight_tracker is None:
            return TrackerInterface()
        return MultiLinearTracker(
            c=c,
            num_layers=self.num_layers
        )
    
    def forward(self, x, width_mask: Optional[torch.Tensor] = None):
        """
        Forward pass through MultiLinear model.
        Base MultiLinear ignores width_mask.
        
        Args:
            x: Input tensor
            width_mask: Optional mask tensor (ignored in base MultiLinear)
        """
        current = x
        
        # Process hidden layers: Norm → Linear → (Dropout)
        for i, (norm, linear) in enumerate(zip(self.hidden_norms, self.hidden_linears)):
            if self.is_norm and norm is not None:
                current = norm(current)
            current = linear(current)
            # Apply dropout only after first layer (skip i=0)
            if i > 0:
                current = self.dropouts[i - 1](current)
        
        # Apply final layer normalization (pre-logit)
        if self.is_norm:
            current = self.final_rms_norm(current)
        
        # Apply dropout before final layer (pre-logits)
        current = self.dropout_final(current)
        
        output = self.final_layer(current)
        # Squeeze only for binary classification (output dim is 1)
        if self.num_class == 2:
            output = output.squeeze(1)
        return output



class KPolynomial(nn.Module):
    def __init__(self, c: Config):
        """
        K-degree polynomial model with learned coefficients for independent polynomials per dimension.
        
        Computes: logit = Σᵢ Σⱼ₌₁ᵏ wᵢⱼ * xᵢʲ
        
        Args:
            c: Configuration object containing model parameters
        """
        super(KPolynomial, self).__init__()
        
        # KPolynomial does not support dropout
        if c.dropout_prob is not None:
            raise ValueError("dropout_prob cannot be set for 'k-polynomial' model_type")
        
        self.input_dim = c.d
        self.k = c.k
        
        # Initialize coefficient matrix of shape (d, k) for learning wᵢⱼ coefficients
        # Each row i contains coefficients for dimension i: [wᵢ₁, wᵢ₂, ..., wᵢₖ]
        self.coefficients = nn.Parameter(torch.randn(self.input_dim, self.k) * 0.01)
    
    def get_tracker(self, c: Config):
        if c.weight_tracker is None:
            return TrackerInterface()
        return PolynomialTracker(c)
    
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
