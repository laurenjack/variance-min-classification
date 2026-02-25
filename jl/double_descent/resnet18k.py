"""Standard Pre-activation ResNet18 with width parameter k.

Based on PreActResNet from https://github.com/kuangliu/pytorch-cifar
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class PreActBlock(nn.Module):
    """Pre-activation ResNet block."""
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Conv2d(
                in_planes, self.expansion * planes,
                kernel_size=1, stride=stride, bias=False
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-activation: BN -> ReLU
        out = F.relu(self.bn1(x))

        # Shortcut uses the activated input
        shortcut = self.shortcut(out)

        # First conv
        out = self.conv1(out)

        # Second pre-activation and conv
        out = self.conv2(F.relu(self.bn2(out)))

        # Residual connection
        out = out + shortcut
        return out


class PreActResNet(nn.Module):
    """Pre-activation ResNet with width parameter k.

    Architecture uses layer widths [k, 2k, 4k, 8k] with strides [1, 2, 2, 2].
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

        # Final batch norm for the last pre-activation
        self.bn = nn.BatchNorm2d(8 * init_channels)

        # Final linear layer: 8k -> num_classes
        self.linear = nn.Linear(8 * init_channels, num_classes)

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

        # Four stages
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        # Final pre-activation before pooling
        out = F.relu(self.bn(out))

        # Global average pooling
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)

        # Linear layer
        out = self.linear(out)

        return out


def make_resnet18k(
    k: int = 64,
    num_classes: int = 10,
) -> PreActResNet:
    """Create a ResNet18 with width parameter k.

    Args:
        k: Width multiplier. Layer channels will be [k, 2k, 4k, 8k].
           k=64 is standard ResNet18.
        num_classes: Number of output classes.

    Returns:
        PreActResNet model.
    """
    return PreActResNet(
        num_blocks=[2, 2, 2, 2],
        num_classes=num_classes,
        init_channels=k,
    )
