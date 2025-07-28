from torch import nn
import torch.nn.functional as F

from src.learned_dropout.model_tracker_standard import MLPStandardTracker, ResNetStandardTracker

##############################################
# Helper function for creating norm layers
##############################################

def create_norm_layer(layer_norm: str, dim: int):
    """
    Create a normalization layer based on the specified type.
    
    Args:
        layer_norm: "layer_norm" | "rms_norm"
        dim: The dimension for the normalization layer
    
    Returns:
        nn.Module: The appropriate normalization layer
    """
    if layer_norm == "layer_norm":
        return nn.LayerNorm(dim)
    elif layer_norm == "rms_norm":
        return nn.RMSNorm(dim)
    else:
        raise ValueError(f"Invalid layer_norm: {layer_norm}. Must be 'layer_norm' or 'rms_norm'.")

##############################################
# Standard MLP without dropout (MLPStandard)
##############################################

class MLPStandard(nn.Module):
    def __init__(self, d, h_list, relus, down_rank_dim=None, up_proj=False, layer_norm=False):
        """
        d: Input dimension.
        h_list: List of hidden layer dimensions. The final layer is added automatically.
        relus: If True, applies a ReLU after each hidden linear layer.
        down_rank_dim: If not None, introduce a rank reducing dim before the last layer.
        up_proj: If True, project the down-ranked pre-logits back to the original dimension
        (only applicable if down_rank_dim is not None)
        layer_norm: If True, applies LayerNorm before every layer except the first.
        """
        super(MLPStandard, self).__init__()
        self.relus = relus
        self.layer_norm = layer_norm
        self.layers = nn.ModuleList()
        # First layer: from input (d) to first hidden layer.
        self.layers.append(nn.Linear(d, h_list[0], bias=False))
        # Hidden layers
        for i in range(1, len(h_list)):
            self.layers.append(nn.Linear(h_list[i-1], h_list[i], bias=False))
        
        h_final = h_list[-1]
        self.down_rank_dim = down_rank_dim
        self.up_proj = up_proj
        # Optionally add the down-ranking layer
        if down_rank_dim is not None:
            self.layers.append(nn.Linear(h_final, down_rank_dim, bias=False))
            # Extend so layer norm also applies to both down-rank layers
            h_list.append(down_rank_dim)
            if up_proj:
                self.layers.append(nn.Linear(down_rank_dim, h_final, bias=False))
                h_list.append(h_final)
            else:
                h_final = down_rank_dim
             
        
        # Final layer: from last hidden layer to scalar output.
        self.layers.append(nn.Linear(h_final, 1, bias=False))

        if self.layer_norm:
            # Identity for first layer, then LayerNorm for each subsequent input dimension
            self.norms = nn.ModuleList([nn.Identity()] + [nn.LayerNorm(dim) for dim in h_list])



    @staticmethod
    def get_tracker(track_weights):
        return MLPStandardTracker(track_weights)

    def forward(self, x):
        layers_with_relus = len(self.layers) - 1
        if self.down_rank_dim is not None:
            layers_with_relus -= 1
            if self.up_proj:
                layers_with_relus -= 1


        current = x
        for i, layer in enumerate(self.layers):
            # Pre-norm for all layers except the first (identity handles the skip)
            if self.layer_norm:
                current = self.norms[i](current)
            current = layer(current)
            # Apply activation on all layers except the final.
            if i < layers_with_relus and self.relus:
                current = F.relu(current)
        return current.squeeze(1)


####################################################
# Standard Residual Block (used in ResNetStandard)
####################################################

class StandardResidualBlock(nn.Module):
    def __init__(self, d, h, relus, layer_norm: str):
        """
        d: Dimension of the residual stream.
        h: Hidden dimension for the block.
        relus: If True, apply a ReLU activation after the weight_in layer.
        layer_norm: "layer_norm" | "rms_norm".
            - "layer_norm": apply LayerNorm(d)
            - "rms_norm": apply RMSNorm(d)
        """
        super(StandardResidualBlock, self).__init__()
        self.relus = relus
        # configure normalization
        self.layer_norm = create_norm_layer(layer_norm, d)

        self.weight_in = nn.Linear(d, h, bias=False)
        self.weight_out = nn.Linear(h, d, bias=False)

    def forward(self, x):
        x = self.layer_norm(x)
        hidden = self.weight_in(x)
        if self.relus:
            hidden = F.relu(hidden)
        out = self.weight_out(hidden)
        return out


class ResNetStandard(nn.Module):
    def __init__(self, d, h_list, relus, layer_norm: str, down_rank_dim=None):
        """
        d: Dimension of the residual stream.
        h_list: List of hidden dimensions, one per residual block.
        relus: If True, applies a ReLU after the weight_in layer of each block.
        layer_norm: "layer_norm" | "rms_norm".
            - "layer_norm": apply LayerNorm at the start of every block and before final layer.
            - "rms_norm": apply RMSNorm at the start of every block and before final layer.
        down_rank_dim: If not None, introduce a rank reducing dim before the last layer.
        """
        super(ResNetStandard, self).__init__()
        self.d = d
        self.down_rank_dim = down_rank_dim
        self.blocks = nn.ModuleList([
            StandardResidualBlock(d, h, relus, layer_norm=layer_norm)
            for h in h_list
        ])
        
        # Optionally add the down-ranking layer
        if down_rank_dim is not None:
            # Add normalization before down-rank layer
            self.pre_down_rank_norm = create_norm_layer(layer_norm, d)
            
            self.down_rank_layer = nn.Linear(d, down_rank_dim, bias=False)
            self.final_layer = nn.Linear(down_rank_dim, 1, bias=False)
            final_norm_dim = down_rank_dim
        else:
            self.pre_down_rank_norm = None
            self.down_rank_layer = None
            self.final_layer = nn.Linear(d, 1, bias=False)
            final_norm_dim = d
        
        # Add final layer normalization (pre-logit) as in transformers
        # Applied after down-rank layer if it exists
        self.final_layer_norm = create_norm_layer(layer_norm, final_norm_dim)

    @staticmethod
    def get_tracker(track_weights):
        return ResNetStandardTracker(track_weights)

    def forward(self, x):
        """
        Forward pass with optional pre‑norm on each block:
        Each block optionally normalizes its input, then applies two linear layers
        (with an optional ReLU), and adds the result back into the residual stream.
        Finally, applies optional layer norm before the output projection.
        """
        current = x
        for block in self.blocks:
            block_out = block(current)
            current = current + block_out
        
        # Apply optional down-ranking layer with pre-normalization
        if self.down_rank_layer is not None:
            current = self.pre_down_rank_norm(current)
            current = self.down_rank_layer(current)
        
        # Apply final layer normalization (pre-logit)
        current = self.final_layer_norm(current)
            
        output = self.final_layer(current)
        return output.squeeze(1)
