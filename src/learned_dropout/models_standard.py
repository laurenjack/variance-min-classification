from torch import nn
import torch.nn.functional as F

from model_tracker_standard import MLPStandardTracker, ResNetStandardTracker

##############################################
# Standard MLP without dropout (MLPStandard)
##############################################

class MLPStandard(nn.Module):
    def __init__(self, d, h_list, relus):
        """
        d: Input dimension.
        h_list: List of hidden layer dimensions. The final layer is added automatically.
        relus: If True, applies a ReLU after each hidden linear layer.
        """
        super(MLPStandard, self).__init__()
        self.relus = relus
        self.layers = nn.ModuleList()
        # First layer: from input (d) to first hidden layer.
        self.layers.append(nn.Linear(d, h_list[0], bias=False))
        # Hidden layers (if any)
        for i in range(1, len(h_list)):
            self.layers.append(nn.Linear(h_list[i - 1], h_list[i], bias=False))
        # Final layer: from last hidden layer to scalar output.
        self.layers.append(nn.Linear(h_list[-1], 1, bias=False))

    @staticmethod
    def get_tracker(track_weights):
        return MLPStandardTracker(track_weights)

    def forward(self, x):
        """
        Forward pass for the standard MLP.
        """
        current = x
        for i, layer in enumerate(self.layers):
            current = layer(current)
            # Apply activation on all layers except the final.
            if i < len(self.layers) - 1 and self.relus:
                current = F.relu(current)
        return current.squeeze(1)


####################################################
# Standard Residual Block (used in ResNetStandard)
####################################################

class StandardResidualBlock(nn.Module):
    def __init__(self, d, h, relus, layer_norm=False):
        """
        d: Dimension of the residual stream.
        h: Hidden dimension for the block.
        relus: If True, apply a ReLU activation after the weight_in layer.
        layer_norm: If True, apply a LayerNorm(d) to the input before weight_in.
        """
        super(StandardResidualBlock, self).__init__()
        self.relus = relus
        self.layer_norm = layer_norm
        if self.layer_norm:
            self.layer_norm_layer = nn.LayerNorm(d)
        self.weight_in = nn.Linear(d, h, bias=False)
        self.weight_out = nn.Linear(h, d, bias=False)

    def forward(self, x):
        if self.layer_norm:
            x = self.layer_norm_layer(x)
        hidden = self.weight_in(x)
        if self.relus:
            hidden = F.relu(hidden)
        out = self.weight_out(hidden)
        return out


##############################################
# Standard ResNet without dropout (ResNetStandard)
# now accepts layer_norm flag
##############################################

class ResNetStandard(nn.Module):
    def __init__(self, d, h_list, relus, layer_norm=False):
        """
        d: Dimension of the residual stream.
        h_list: List of hidden dimensions, one per residual block.
        relus: If True, applies a ReLU after the weight_in layer of each block.
        layer_norm: If True, applies LayerNorm at the start of every block.
        """
        super(ResNetStandard, self).__init__()
        self.d = d
        self.layer_norm = layer_norm
        self.blocks = nn.ModuleList([
            StandardResidualBlock(d, h, relus, layer_norm=self.layer_norm)
            for h in h_list
        ])
        self.final_layer = nn.Linear(d, 1, bias=False)

    @staticmethod
    def get_tracker(track_weights):
        return ResNetStandardTracker(track_weights)

    def forward(self, x):
        """
        Forward pass with optional pre‑norm:
        Each block optionally normalizes its input, then applies the two linear layers
        (with an optional ReLU), and adds the result back into the residual stream.
        """
        current = x
        for block in self.blocks:
            block_out = block(current)
            current = current + block_out
        output = self.final_layer(current)
        return output.squeeze(1)