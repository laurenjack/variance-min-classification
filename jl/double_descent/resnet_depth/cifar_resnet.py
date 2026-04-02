"""CIFAR-10 ResNet with variable depth following He et al. (2015).

3-stage architecture with depth = 6n + 2, where n = blocks per stage.
Widths [16k, 32k, 64k] with optional width multiplier k.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from jl.double_descent.resnet18.resnet18k import PreActBlock


class CIFARResNet(nn.Module):
    """CIFAR-10 ResNet with 3 stages and variable depth.

    Architecture: conv(3->16k) -> stage1(16k) -> stage2(32k) -> stage3(64k) -> avgpool -> FC
    Depth = 6n + 2 where n = blocks per stage.
    """

    def __init__(self, n: int, k: int = 1, num_classes: int = 10):
        super().__init__()
        self.init_channels = 16 * k
        self.in_planes = self.init_channels

        # Initial conv layer: 3 -> 16k channels
        self.conv1 = nn.Conv2d(
            3, self.init_channels, kernel_size=3, stride=1, padding=1, bias=False
        )

        # Three stages with increasing channels
        self.layer1 = self._make_layer(16 * k, n, stride=1)
        self.layer2 = self._make_layer(32 * k, n, stride=2)
        self.layer3 = self._make_layer(64 * k, n, stride=2)

        # Final batch norm for the last pre-activation
        self.bn = nn.BatchNorm2d(64 * k)

        # Final linear layer: 64k -> num_classes
        self.linear = nn.Linear(64 * k, num_classes)

    def _make_layer(
        self,
        planes: int,
        num_blocks: int,
        stride: int,
    ) -> nn.Sequential:
        """Create a stage with multiple blocks."""
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(PreActBlock(self.in_planes, planes, s))
            self.in_planes = planes * PreActBlock.expansion
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Initial conv
        out = self.conv1(x)

        # Three stages
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        # Final pre-activation before pooling
        out = F.relu(self.bn(out))

        # Global average pooling (8x8 feature maps -> 1x1)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)

        # Linear layer
        out = self.linear(out)

        return out


def make_cifar_resnet(
    n: int,
    k: int = 1,
    num_classes: int = 10,
) -> CIFARResNet:
    """Create a CIFAR ResNet with depth = 6n + 2.

    Args:
        n: Blocks per stage. Depth = 6n + 2.
           n=3 -> ResNet-20, n=5 -> ResNet-32, n=9 -> ResNet-56, n=18 -> ResNet-110.
        k: Width multiplier. Layer channels will be [16k, 32k, 64k].
           k=1 is the standard CIFAR ResNet from He et al. (2015).
        num_classes: Number of output classes.

    Returns:
        CIFARResNet model.
    """
    return CIFARResNet(n=n, k=k, num_classes=num_classes)
