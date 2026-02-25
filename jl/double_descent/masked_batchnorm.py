"""Masked BatchNorm2d for channel-masked parallel training."""

import torch
import torch.nn as nn
from typing import Optional


class MaskedBatchNorm2d(nn.Module):
    """BatchNorm2d that computes statistics only over active (masked) channels.

    When a channel mask is provided, running mean/var are computed only over
    channels where mask == 1, and output is zeroed for inactive channels.
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
    ):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.ones(num_features))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_buffer('running_mean', None)
            self.register_buffer('running_var', None)
            self.register_buffer('num_batches_tracked', None)

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with optional channel mask.

        Args:
            x: Input tensor of shape [N, C, H, W]
            mask: Optional channel mask of shape [C] with 1 for active channels, 0 for inactive

        Returns:
            Normalized tensor of shape [N, C, H, W]
        """
        if mask is None:
            # Standard batch norm
            return self._standard_forward(x)

        # Masked batch norm
        return self._masked_forward(x, mask)

    def _standard_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard batch norm forward pass."""
        if self.training:
            # Compute batch statistics
            mean = x.mean(dim=(0, 2, 3))
            var = x.var(dim=(0, 2, 3), unbiased=False)

            # Update running stats
            if self.track_running_stats:
                with torch.no_grad():
                    self.num_batches_tracked += 1
                    if self.momentum is None:
                        exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                    else:
                        exponential_average_factor = self.momentum

                    self.running_mean = (1 - exponential_average_factor) * self.running_mean + exponential_average_factor * mean
                    self.running_var = (1 - exponential_average_factor) * self.running_var + exponential_average_factor * var
        else:
            mean = self.running_mean
            var = self.running_var

        # Normalize
        x_norm = (x - mean.view(1, -1, 1, 1)) / torch.sqrt(var.view(1, -1, 1, 1) + self.eps)

        # Scale and shift
        if self.affine:
            x_norm = x_norm * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)

        return x_norm

    def _masked_forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Masked batch norm forward - only compute stats over active channels."""
        # mask shape: [C], x shape: [N, C, H, W]
        mask_4d = mask.view(1, -1, 1, 1)  # [1, C, 1, 1]

        # Zero out inactive channels before computing stats
        x_masked = x * mask_4d

        if self.training:
            # Compute statistics only over active spatial locations
            # For each channel, compute mean and var across N, H, W
            # But we only care about channels where mask == 1

            # Sum over N, H, W dimensions
            n_elements = x.shape[0] * x.shape[2] * x.shape[3]

            # Mean per channel (only active channels matter, but compute all)
            mean = x_masked.sum(dim=(0, 2, 3)) / n_elements

            # Variance per channel
            var = ((x_masked - mean.view(1, -1, 1, 1) * mask_4d) ** 2).sum(dim=(0, 2, 3)) / n_elements

            # Update running stats only for active channels
            if self.track_running_stats:
                with torch.no_grad():
                    self.num_batches_tracked += 1
                    if self.momentum is None:
                        exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                    else:
                        exponential_average_factor = self.momentum

                    # Only update active channels
                    new_running_mean = (1 - exponential_average_factor) * self.running_mean + exponential_average_factor * mean
                    new_running_var = (1 - exponential_average_factor) * self.running_var + exponential_average_factor * var

                    # Preserve running stats for inactive channels
                    self.running_mean = torch.where(mask.bool(), new_running_mean, self.running_mean)
                    self.running_var = torch.where(mask.bool(), new_running_var, self.running_var)
        else:
            mean = self.running_mean
            var = self.running_var

        # Normalize (add eps to avoid division by zero for inactive channels)
        x_norm = (x - mean.view(1, -1, 1, 1)) / torch.sqrt(var.view(1, -1, 1, 1) + self.eps)

        # Scale and shift
        if self.affine:
            x_norm = x_norm * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)

        # Zero out inactive channels
        x_norm = x_norm * mask_4d

        return x_norm
