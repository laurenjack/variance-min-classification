from torch import nn
import torch
from typing import Optional, List
from dataclasses import dataclass


class Dropout(nn.Module):
    """
    Base class for dropout implementations.
    Provides a common interface for all dropout variants.
    """
    def __init__(self, p: float = 0.5, inplace: bool = False):
        """
        Args:
            p: Dropout probability
            inplace: Whether to do inplace operation
        """
        super(Dropout, self).__init__()
        self.p = p
        self.inplace = inplace
    
    def forward(self, x: torch.Tensor, x_indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through dropout.
        
        Args:
            x: Input tensor
            x_indices: Optional tensor of point indices (used by HashedDropout)
            
        Returns:
            Output tensor
        """
        raise NotImplementedError("Subclasses must implement forward()")


class NoOpDropout(Dropout):
    """
    Null Object Pattern implementation of Dropout.
    Always returns the input unchanged, regardless of training/eval mode.
    """
    def __init__(self, p: float = 0.5, inplace: bool = False):
        """
        Args:
            p: Dropout probability (ignored, kept for API compatibility)
            inplace: Whether to do inplace operation (ignored, kept for API compatibility)
        """
        super(NoOpDropout, self).__init__(p=p, inplace=inplace)
    
    def forward(self, x: torch.Tensor, x_indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass - always returns input unchanged.
        
        Args:
            x: Input tensor
            x_indices: Optional tensor of point indices (ignored)
            
        Returns:
            Input tensor unchanged
        """
        return x


class StandardDropout(Dropout):
    """
    Standard dropout implementation wrapping PyTorch's nn.Dropout.
    """
    def __init__(self, p: float = 0.5, inplace: bool = False):
        """
        Args:
            p: Dropout probability
            inplace: Whether to do inplace operation
        """
        super(StandardDropout, self).__init__(p=p, inplace=inplace)
        self._dropout = nn.Dropout(p=p, inplace=inplace)
    
    def forward(self, x: torch.Tensor, x_indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through PyTorch dropout.
        
        Args:
            x: Input tensor
            x_indices: Optional tensor of point indices (ignored for standard dropout)
            
        Returns:
            Output tensor with dropout applied
        """
        return self._dropout(x)


class HashedDropout(Dropout):
    """
    Deterministic dropout based on hashing (point_index, node_index) pairs.
    
    Each (point, node) pair always produces the same dropout mask, ensuring
    that across epochs, each node is trained on the same set of points.
    
    Behavior is identical in training and evaluation modes.
    When x_indices is None, all neurons are used (no dropout applied).
    """
    
    # Large primes for hash mixing
    P1 = 73856093
    P2 = 19349669
    M = 2**31 - 1
    
    def __init__(self, p: float = 0.5, layer_index: int = 0, d_model: int = 1, inplace: bool = False):
        """
        Args:
            p: Dropout probability
            layer_index: Index of the layer this dropout belongs to
            d_model: Model dimension (used to compute node indices)
            inplace: Whether to do inplace operation (ignored for HashedDropout)
        """
        super(HashedDropout, self).__init__(p=p, inplace=inplace)
        self.layer_index = layer_index
        self.d_model = d_model
    
    def forward(self, x: torch.Tensor, x_indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with deterministic hashed dropout.
        
        Args:
            x: Input tensor of shape [batch_size, d_model]
            x_indices: Tensor of point indices, shape [batch_size]. If None, no dropout applied.
            
        Returns:
            Output tensor with hashed dropout applied
        """
        if x_indices is None:
            # No indices provided, use all neurons (no dropout)
            return x
        
        batch_size = x.shape[0]
        d = x.shape[1]
        
        # Compute node indices: layer_index * d_model + j for j in [0, d)
        # node_indices shape: [d]
        node_indices = self.layer_index * self.d_model + torch.arange(d, device=x.device)
        
        # Expand for broadcasting: point_indices [batch_size, 1], node_indices [1, d]
        # Result: [batch_size, d]
        point_indices = x_indices.unsqueeze(1).long()  # [batch_size, 1]
        node_indices = node_indices.unsqueeze(0)  # [1, d]
        
        # Compute hash values in [0, 1]
        combined = (point_indices * self.P1 + node_indices * self.P2) % self.M
        hash_values = combined.float() / self.M
        
        # Create dropout mask: keep where hash >= p (so p fraction gets dropped)
        keep_mask = (hash_values >= self.p).float()
        
        # Scale by 1/(1-p) to maintain expected value, same as standard dropout
        scale = 1.0 / (1.0 - self.p)
        
        return x * keep_mask * scale


@dataclass
class DropoutModules:
    """
    Container for dropout modules used in models.
    
    Attributes:
        dropouts: List of Dropout modules for hidden/block layers
        dropout_final: Dropout module for the final layer (pre-logits)
    """
    dropouts: List[Dropout]
    dropout_final: Dropout


def create_dropout_modules(
    num_layers: int,
    dropout_prob: Optional[float],
    is_hashed_dropout: bool = False,
    d_model: Optional[int] = None,
    final_dim: Optional[int] = None,
) -> DropoutModules:
    """
    Factory function to create dropout modules for a model.
    
    Creates dropouts for hidden/block layers and a separate dropout for the final layer.
    
    Args:
        num_layers: Number of hidden/block layers (number of dropouts to create for these layers)
        dropout_prob: Dropout probability. If None, creates NoOpDropout instances.
        is_hashed_dropout: If True, creates HashedDropout instances (requires d_model).
        d_model: Model dimension for hidden/block layers, required when is_hashed_dropout=True.
        final_dim: Dimension for final layer dropout. If None, uses d_model (for Resnet) or 
                   must be provided separately (for MLP/MultiLinear).
    
    Returns:
        DropoutModules: Container with dropouts for hidden/block layers and final layer
    """
    dropouts = []
    
    # Create dropouts for hidden/block layers
    for layer_idx in range(num_layers):
        if dropout_prob is None:
            dropouts.append(NoOpDropout())
        elif is_hashed_dropout:
            if d_model is None:
                raise ValueError("d_model must be provided when is_hashed_dropout=True")
            dropouts.append(HashedDropout(p=dropout_prob, layer_index=layer_idx, d_model=d_model))
        else:
            dropouts.append(StandardDropout(p=dropout_prob))
    
    # Create final layer dropout
    # For hashed dropout, final layer uses layer_index=num_layers (blocks are 0 to num_layers-1)
    if dropout_prob is None:
        dropout_final = NoOpDropout()
    elif is_hashed_dropout:
        if d_model is None:
            raise ValueError("d_model must be provided when is_hashed_dropout=True")
        # Final layer uses num_layers as layer_index (blocks use 0 to num_layers-1)
        dropout_final = HashedDropout(p=dropout_prob, layer_index=num_layers, d_model=d_model)
    else:
        dropout_final = StandardDropout(p=dropout_prob)
    
    return DropoutModules(dropouts=dropouts, dropout_final=dropout_final)
