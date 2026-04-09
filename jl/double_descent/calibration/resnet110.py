"""ResNet-110 for CIFAR (32x32 images).

Follows He et al. 2015 "Deep Residual Learning for Image Recognition":
- Stem: 3x3 conv (3->16), BN, ReLU (no maxpool)
- 3 groups of basic blocks: 16->16, 16->32 (stride 2), 32->64 (stride 2)
- n=18 blocks per group -> 6*18 + 2 = 110 layers
- Global average pool -> 64-dim features -> linear classifier
- ~1.7M parameters
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """Two 3x3 convs with BN+ReLU and identity/projection shortcut."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet110(nn.Module):
    """ResNet-110 for CIFAR (32x32 input)."""

    def __init__(self, num_classes: int, n: int = 18):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        self.layer1 = self._make_group(16, 16, n, stride=1)
        self.layer2 = self._make_group(16, 32, n, stride=2)
        self.layer3 = self._make_group(32, 64, n, stride=2)

        self.fc = nn.Linear(64, num_classes)

        # Initialize weights (He initialization)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_group(
        self, in_channels: int, out_channels: int, num_blocks: int, stride: int
    ) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        current_channels = in_channels
        for s in strides:
            layers.append(BasicBlock(current_channels, out_channels, s))
            current_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract 64-dim features before the final linear layer."""
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        return out


def make_resnet110(num_classes: int) -> nn.Module:
    """Create ResNet-110 for CIFAR (32x32 images)."""
    return ResNet110(num_classes=num_classes)


def extract_features(model: nn.Module, images: torch.Tensor) -> torch.Tensor:
    """Extract 64-dim features before the final linear layer."""
    return model.extract_features(images)
