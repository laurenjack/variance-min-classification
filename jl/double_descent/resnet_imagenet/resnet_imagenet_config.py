"""Configuration for ResNet-50 ImageNet training (He et al. 2015 recipe)."""

from dataclasses import dataclass, field
from typing import List


@dataclass
class ResNetImageNetConfig:
    """He et al. 2015 recipe for ResNet-50 on ImageNet-1K.

    SGD with momentum 0.9, weight decay 1e-4, base LR 0.1, batch 256,
    step decay (divide by 10) at epochs 30, 60, 80, 90 epochs total.
    """

    # Training
    epochs: int = 90
    batch_size: int = 256
    learning_rate: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 1e-4

    # LR schedule
    lr_decay_epochs: List[int] = field(default_factory=lambda: [30, 60, 80])
    lr_decay_factor: float = 0.1

    # Data loading
    num_workers: int = 8
    data_augmentation: bool = True

    # Model
    num_classes: int = 1000

    # Mixed precision
    use_amp: bool = True

    # Logging
    log_interval: int = 1
