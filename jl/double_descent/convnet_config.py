"""Configuration for Deep Double Descent experiments."""

from dataclasses import dataclass


@dataclass
class DDConfig:
    """Configuration for double descent training."""

    # Width parameter
    k_start: int = 9  # Starting k value, trains k, k+1, ..., k+7

    # Training
    # Paper used: epochs=4000, batch_size=128, lr=0.0001, optimizer=Adam
    # We use AdamW with 10x learning rate for faster convergence
    epochs: int = 10
    batch_size: int = 128
    learning_rate: float = 0.001
    optimizer: str = "adam_w"

    # Data
    label_noise: float = 0.15
    data_augmentation: bool = True

    # Logging
    log_interval: int = 1  # Every epoch
