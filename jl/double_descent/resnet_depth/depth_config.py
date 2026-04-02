"""Configuration for CIFAR ResNet depth-varying double descent experiments."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class DepthDDConfig:
    """Configuration for depth double descent training.

    Uses the CIFAR-10 ResNet architecture from He et al. (2015) with 3 stages
    and variable depth (6n+2 layers, where n = blocks per stage).
    """

    # Depth parameter
    n_start: int = 4  # Starting n (blocks per stage), depth = 6n+2
    n_step: int = 4  # Increment between n values
    n_count: int = 16  # Number of n values to sweep
    k: int = 1  # Width multiplier (widths = [16k, 32k, 64k])

    # Training
    epochs: int = 800
    batch_size: int = 128
    learning_rate: float = 0.001
    optimizer: str = "adam_w"
    cosine_decay_epoch: Optional[int] = 100

    # Data
    label_noise: float = 0.15
    data_augmentation: bool = True

    # Logging
    log_interval: int = 1
