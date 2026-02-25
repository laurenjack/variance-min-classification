"""ResNet18 with channel masking for parallel width training.

Based on PreActResNet from https://github.com/kuangliu/pytorch-cifar
Modified to support channel masking for parallel training across width values.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List

from jl.double_descent.masked_batchnorm import MaskedBatchNorm2d


class MaskedPreActBlock(nn.Module):
    """Pre-activation ResNet block with channel masking support."""
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.bn1 = MaskedBatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = MaskedBatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )

        self.shortcut = None
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Conv2d(
                in_planes, self.expansion * planes,
                kernel_size=1, stride=stride, bias=False
            )

    def forward(
        self,
        x: torch.Tensor,
        in_mask: Optional[torch.Tensor] = None,
        out_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with optional channel masks.

        Args:
            x: Input tensor [N, C_in, H, W]
            in_mask: Mask for input channels [C_in], 1 for active, 0 for inactive
            out_mask: Mask for output channels [C_out], 1 for active, 0 for inactive

        Returns:
            Output tensor [N, C_out, H, W]
        """
        # Pre-activation: BN -> ReLU
        out = F.relu(self.bn1(x, in_mask))

        # Shortcut uses the activated input
        if self.shortcut is not None:
            shortcut = self.shortcut(out)
            if out_mask is not None:
                shortcut = shortcut * out_mask.view(1, -1, 1, 1)
        else:
            shortcut = x

        # First conv
        out = self.conv1(out)
        if out_mask is not None:
            out = out * out_mask.view(1, -1, 1, 1)

        # Second pre-activation and conv
        out = self.conv2(F.relu(self.bn2(out, out_mask)))
        if out_mask is not None:
            out = out * out_mask.view(1, -1, 1, 1)

        # Residual connection
        out = out + shortcut
        return out


class MaskedPreActResNet(nn.Module):
    """Pre-activation ResNet with channel masking for parallel width training.

    Architecture uses layer widths [k, 2k, 4k, 8k] with strides [1, 2, 2, 2].
    The model is built with k=k_max channels, and a width mask can be applied
    to simulate training with smaller k values.
    """

    def __init__(
        self,
        num_blocks: List[int],
        num_classes: int = 10,
        init_channels: int = 64,
    ):
        super().__init__()
        self.init_channels = init_channels
        self.in_planes = init_channels

        # Initial conv layer: 3 -> k channels
        self.conv1 = nn.Conv2d(
            3, init_channels, kernel_size=3, stride=1, padding=1, bias=False
        )

        # Four stages with increasing channels
        self.layer1 = self._make_layer(init_channels, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(2 * init_channels, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(4 * init_channels, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(8 * init_channels, num_blocks[3], stride=2)

        # Final linear layer: 8k -> num_classes
        self.linear = nn.Linear(8 * init_channels, num_classes)

    def _make_layer(
        self,
        planes: int,
        num_blocks: int,
        stride: int,
    ) -> nn.ModuleList:
        """Create a stage with multiple blocks."""
        strides = [stride] + [1] * (num_blocks - 1)
        layers = nn.ModuleList()
        for s in strides:
            layers.append(MaskedPreActBlock(self.in_planes, planes, s))
            self.in_planes = planes * MaskedPreActBlock.expansion
        return layers

    def forward(
        self,
        x: torch.Tensor,
        width_masks: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Forward pass with optional width masks.

        Args:
            x: Input images [N, 3, 32, 32]
            width_masks: List of 4 masks for each stage's channels:
                [mask_k, mask_2k, mask_4k, mask_8k]
                Each mask is shape [num_channels] with 1s for active channels.
                If None, all channels are used (full width).

        Returns:
            Logits [N, num_classes]
        """
        if width_masks is None:
            # No masking - use all channels
            mask1 = mask2 = mask4 = mask8 = None
        else:
            mask1, mask2, mask4, mask8 = width_masks

        # Initial conv
        out = self.conv1(x)
        if mask1 is not None:
            out = out * mask1.view(1, -1, 1, 1)

        # Stage 1: k channels
        in_mask = mask1
        for block in self.layer1:
            out = block(out, in_mask, mask1)
            in_mask = mask1

        # Stage 2: 2k channels
        for i, block in enumerate(self.layer2):
            out_mask = mask2
            out = block(out, in_mask, out_mask)
            in_mask = mask2

        # Stage 3: 4k channels
        for i, block in enumerate(self.layer3):
            out_mask = mask4
            out = block(out, in_mask, out_mask)
            in_mask = mask4

        # Stage 4: 8k channels
        for i, block in enumerate(self.layer4):
            out_mask = mask8
            out = block(out, in_mask, out_mask)
            in_mask = mask8

        # Global average pooling
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)

        # Mask the features before linear layer
        if mask8 is not None:
            out = out * mask8

        # Linear layer
        out = self.linear(out)

        return out


def make_resnet18k(k: int = 64, num_classes: int = 10) -> MaskedPreActResNet:
    """Create a ResNet18 with width parameter k.

    Args:
        k: Width multiplier. Layer channels will be [k, 2k, 4k, 8k].
           k=64 is standard ResNet18.
        num_classes: Number of output classes.

    Returns:
        MaskedPreActResNet model.
    """
    return MaskedPreActResNet(
        num_blocks=[2, 2, 2, 2],
        num_classes=num_classes,
        init_channels=k,
    )


def create_width_masks(k: int, k_max: int, device: torch.device) -> List[torch.Tensor]:
    """Create channel masks for a given width k within a k_max model.

    Args:
        k: Target width (1 to k_max)
        k_max: Maximum width the model was built with
        device: Device for the tensors

    Returns:
        List of 4 masks for [k, 2k, 4k, 8k] channel stages.
    """
    masks = []
    for multiplier in [1, 2, 4, 8]:
        active_channels = k * multiplier
        total_channels = k_max * multiplier
        mask = torch.zeros(total_channels, device=device)
        mask[:active_channels] = 1.0
        masks.append(mask)
    return masks
