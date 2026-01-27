from torch import nn
import torch
import torch.nn.functional as F
from typing import Optional

from jl.models import RMSNorm, ResidualBlock, Resnet, MLP, SimpleMLP
from jl.config import Config
from jl.feature_experiments.dropout import DropoutModules


class MaskedRMSNorm(RMSNorm):
    """
    RMSNorm that supports width-masking of the feature dimension.
    If a mask is provided, the RMS is computed only over active (mask==1) dims,
    and the output is zeroed over inactive dims to simulate an effective width.
    """
    def __init__(self, dim, eps=1e-6, learnable_norm_parameters=True):
        super().__init__(dim, eps=eps, learnable_norm_parameters=learnable_norm_parameters)

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


class ResidualBlockH(nn.Module):
    def __init__(self, d, h, is_norm=True, learnable_norm_parameters=True):
        """
        ResidualBlockH that handles width_mask for the h parameter.
        
        d: Dimension of the residual stream.
        h: Hidden dimension for the block.
        is_norm: Whether to apply RMSNorm (default True).
        learnable_norm_parameters: Whether RMSNorm weights are learnable.
        """
        super(ResidualBlockH, self).__init__()
        self.is_norm = is_norm
        if is_norm:
            self.rms_norm = RMSNorm(d, learnable_norm_parameters=learnable_norm_parameters)
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


class ResidualBlockDModel(nn.Module):
    def __init__(self, d, h, is_norm=True, learnable_norm_parameters=True):
        """
        Special ResidualBlock for ResnetDModel that can handle d_model width masking.
        
        d: Maximum dimension of the residual stream.
        h: Hidden dimension for the block.
        is_norm: Whether to apply RMSNorm (default True).
        learnable_norm_parameters: Whether RMSNorm weights are learnable.
        """
        super(ResidualBlockDModel, self).__init__()
        # Use MaskedRMSNorm to compute RMS over active d_model* only
        self.is_norm = is_norm
        if is_norm:
            self.masked_rms_norm = MaskedRMSNorm(d, learnable_norm_parameters=learnable_norm_parameters)
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


class ResnetH(Resnet):
    def __init__(self, c: Config, dropout_modules: DropoutModules):
        """
        ResnetH class that uses width_mask to vary the hidden dimension h.
        Uses ResidualBlockH and inherits all behavior from base Resnet class.
        """
        super(ResnetH, self).__init__(c, dropout_modules=dropout_modules, block_class=ResidualBlockH)


class ResnetDModel(Resnet):
    def __init__(self, c: Config, dropout_modules: DropoutModules):
        """
        ResnetDModel class that uses width_mask to vary the d_model dimension.
        The width_mask is applied to the model dimension throughout the network.
        """
        super(Resnet, self).__init__()  # Skip Resnet.__init__ to customize initialization
        self.input_dim = c.d
        self.d_model = c.d if c.d_model is None else c.d_model
        self.num_class = c.num_class
        
        # Determine output dimension: 1 for binary, num_class for multi-class
        output_dim = 1 if c.num_class == 2 else c.num_class

        # We always create an input projection to the maximum d_model, so we can mask
        self.input_projection = nn.Linear(self.input_dim, self.d_model, bias=False)

        # Use special ResidualBlockDModel that can handle d_model masking
        self.blocks = nn.ModuleList([
            ResidualBlockDModel(self.d_model, c.h, is_norm=c.is_norm, learnable_norm_parameters=c.learnable_norm_parameters)
            for _ in range(c.num_layers)
        ])
        
        # Final layer connects directly from d_model to output
        self.final_layer = nn.Linear(self.d_model, output_dim, bias=False)
        
        # Add final layer normalization (pre-logit) as in transformers
        self.is_norm = c.is_norm
        if c.is_norm:
            self.final_rms_norm = MaskedRMSNorm(self.d_model, learnable_norm_parameters=c.learnable_norm_parameters)
        else:
            self.final_rms_norm = None
        
        # Convert dropout list to ModuleList and store final dropout separately
        self.dropouts = nn.ModuleList(dropout_modules.dropouts)
        self.dropout_final = dropout_modules.dropout_final
        
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

        for block, dropout in zip(self.blocks, self.dropouts):
            block_out = block(current, d_model_mask=width_mask)
            # Apply dropout to block output before adding to residual stream
            block_out = dropout(block_out)
            current = current + block_out
        
        # Apply final layer normalization (pre-logit) with mask
        if self.is_norm:
            current = self.final_rms_norm(current, mask=width_mask)
        
        # Apply dropout before final layer (pre-logits)
        # Note: ResnetDModel doesn't use x_indices in forward, so pass None
        current = self.dropout_final(current, x_indices=None)
            
        output = self.final_layer(current)
        # Squeeze only for binary classification (output dim is 1)
        if self.num_class == 2:
            output = output.squeeze(1)
        return output


class MLPH(nn.Module):
    def __init__(self, c: Config, dropout_modules: DropoutModules):
        """
        MLP that uses width_mask to vary the h dimension.
        The width_mask is applied to the hidden dimension throughout the network.
        """
        super(MLPH, self).__init__()
        self.input_dim = c.d
        self.h = c.h if c.h is not None else c.d
        self.num_layers = c.num_layers
        self.is_norm = c.is_norm
        self.num_class = c.num_class
        
        # Determine output dimension: 1 for binary, num_class for multi-class
        output_dim = 1 if c.num_class == 2 else c.num_class
        
        # Build MLP layers manually to support masking
        self.input_layer = nn.Linear(self.input_dim, self.h, bias=False)
        if c.is_norm:
            self.input_norm = MaskedRMSNorm(self.h, learnable_norm_parameters=c.learnable_norm_parameters)
        else:
            self.input_norm = None
        
        # Hidden layers
        self.hidden_layers = nn.ModuleList()
        self.hidden_norms = nn.ModuleList()
        for _ in range(self.num_layers - 1):
            self.hidden_layers.append(nn.Linear(self.h, self.h, bias=False))
            if c.is_norm:
                self.hidden_norms.append(MaskedRMSNorm(self.h, learnable_norm_parameters=c.learnable_norm_parameters))
            else:
                self.hidden_norms.append(None)
        
        # Final layer connects directly from h to output
        self.final_layer = nn.Linear(self.h, output_dim, bias=False)
        
        # Add final layer normalization (pre-logit)
        if c.is_norm:
            self.final_rms_norm = MaskedRMSNorm(self.h, learnable_norm_parameters=c.learnable_norm_parameters)
        else:
            self.final_rms_norm = None
    
        # Convert dropout list to ModuleList and store final dropout separately
        self.dropouts = nn.ModuleList(dropout_modules.dropouts)
        self.dropout_final = dropout_modules.dropout_final
    
    def get_tracker(self, track_weights):
        raise NotImplementedError("Weight tracking is not supported for MLPH. Use base MLP instead.")
    
    def forward(self, x, width_mask: Optional[torch.Tensor] = None):
        """
        Forward pass with width_mask applied to h dimensions.
        
        Args:
            x: Input tensor
            width_mask: Optional tensor of shape (h,) containing 0.0 or 1.0 values.
                       Applied to h dimensions throughout the network.
        """
        # Input layer with masking (no dropout on first layer)
        current = self.input_layer(x)
        if width_mask is not None:
            current = current * width_mask.unsqueeze(0)
        if self.is_norm:
            current = self.input_norm(current, mask=width_mask)
        current = F.relu(current)
        
        # Hidden layers with masking
        for i, (hidden_layer, hidden_norm) in enumerate(zip(self.hidden_layers, self.hidden_norms)):
            current = hidden_layer(current)
            if width_mask is not None:
                current = current * width_mask.unsqueeze(0)
            if self.is_norm:
                current = hidden_norm(current, mask=width_mask)
            current = F.relu(current)
            current = self.dropouts[i](current)
        
        # Apply final layer normalization with mask
        if self.is_norm:
            current = self.final_rms_norm(current, mask=width_mask)
        
        # Apply dropout before final layer (pre-logits)
        current = self.dropout_final(current)

        output = self.final_layer(current)
        # Squeeze only for binary classification (output dim is 1)
        if self.num_class == 2:
            output = output.squeeze(1)
        return output


class SimpleMLPH(nn.Module):
    def __init__(self, c: Config):
        """
        Width-varying SimpleMLP that applies width_mask to the hidden dimension h.
        No normalization, no dropout - same as SimpleMLP but with width masking support.

        0 layers: Linear(d, num_class) - no width masking possible
        1 layer:  Linear(d, h) -> mask -> ReLU -> Linear(h, num_class)
        N layers: Linear(d, h) -> mask -> ReLU -> [Linear(h, h) -> mask -> ReLU] * (N-1) -> Linear(h, num_class)
        """
        super(SimpleMLPH, self).__init__()
        self.num_layers = c.num_layers
        self.num_class = c.num_class
        self.h = c.h
        output_dim = 1 if c.num_class == 2 else c.num_class

        if self.num_layers == 0:
            self.hidden_layers = None
            self.final_layer = nn.Linear(c.d, output_dim, bias=False)
        else:
            layers = []
            layers.append(nn.Linear(c.d, c.h, bias=False))
            for _ in range(self.num_layers - 1):
                layers.append(nn.Linear(c.h, c.h, bias=False))
            self.hidden_layers = nn.ModuleList(layers)
            self.final_layer = nn.Linear(c.h, output_dim, bias=False)

    def get_tracker(self, c: Config):
        raise NotImplementedError("Weight tracking is not supported for SimpleMLPH. Use SimpleMLP instead.")

    def forward(self, x, width_mask: Optional[torch.Tensor] = None):
        """
        Forward pass with optional width masking on hidden dimension h.

        Args:
            x: Input tensor of shape [batch_size, d]
            width_mask: Optional tensor of shape [h] containing 0.0 or 1.0 values.
                       Applied to h dimensions throughout the network.
        """
        current = x
        if self.num_layers > 0:
            for layer in self.hidden_layers:
                current = layer(current)
                if width_mask is not None:
                    current = current * width_mask.unsqueeze(0)
                current = torch.relu(current)

        output = self.final_layer(current)
        if self.num_class == 2:
            output = output.squeeze(1)
        return output
