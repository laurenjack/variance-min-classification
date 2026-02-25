"""Configuration for Deep Double Descent experiments."""

from dataclasses import dataclass


@dataclass
class DDConfig:
    """Configuration for double descent training."""

    # Resnet k values (full range is 1-64)
    width_min: int = 1
    width_max: int = 4  # Start small for memory testing

    # Training (from paper)
    epochs: int = 10  # 4000 for full run
    batch_size: int = 128
    learning_rate: float = 0.0001
    optimizer: str = "adam"

    # Data
    label_noise: float = 0.15
    data_augmentation: bool = True

    # Logging
    log_interval: int = 1  # Every epoch
