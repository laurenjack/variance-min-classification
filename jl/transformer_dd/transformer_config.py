"""Configuration for Transformer Double Descent experiments."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class TDDConfig:
    """Configuration for Transformer double descent training."""

    # Model architecture (from paper)
    d_model_start: int = 128  # Starting embedding dim, trains d_model, d_model+8, ..., d_model+8*(N-1)
    n_layers: int = 6  # Encoder and decoder layers
    n_heads: int = 8  # Attention heads (always 8, even for small d_model)
    d_ff_multiplier: int = 4  # d_ff = d_ff_multiplier * d_model
    # Note: dropout hardcoded to 0.0 in model (per paper)

    # Training (from paper: 80K steps)
    max_steps: int = 1000  # Gradient steps
    max_tokens: int = 4096  # Tokens per batch (max-tokens batching)
    warmup_steps: int = 4000  # LR warmup steps
    optimizer: str = "adam_w"  # AdamW with Vaswani params (beta1=0.9, beta2=0.98, eps=1e-9)

    # Regularization
    label_smoothing: Optional[float] = 0.1  # None to disable

    # Data
    train_samples: int = 4000  # Subsample training set (paper uses 4K and 18K)
    subsample_seed: int = 42  # Fixed seed for reproducibility

    # Logging
    log_interval: int = 100  # Log train metrics every N steps
    eval_interval: int = 100  # Evaluate on valid set every N steps
