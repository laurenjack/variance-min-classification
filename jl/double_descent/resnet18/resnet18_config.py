"""Configuration for Deep Double Descent experiments."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class DDConfig:
    """Configuration for double descent training."""

    # Width parameter
    k_start: int = 36  # Starting k value, trains k, k+4, k+8, ..., k+4*(N-1)

    # Training
    # Paper used: epochs=4000, batch_size=128, lr=0.0001, optimizer=Adam
    # We use AdamW with 10x learning rate for faster convergence
    epochs: int = 800
    batch_size: int = 128
    learning_rate: float = 0.001
    optimizer: str = "adam_w"
    cosine_decay_epoch: Optional[int] = 100  # Cosine decay LR to 10% from this epoch

    # Data
    label_noise: float = 0.15
    data_augmentation: bool = True

    # Logging
    log_interval: int = 1  # Every epoch
