"""Configuration for Transformer Double Descent experiments."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class TDDConfig:
    """Configuration for Transformer double descent training.

    Note: d_model and train_samples are NOT configurable - they are
    hardcoded in transformer_main.py for the full experiment.
    """

    # Model architecture (from paper)
    n_layers: int = 6  # Encoder and decoder layers
    n_heads: int = 8  # Attention heads (always 8, even for small d_model)
    d_ff_multiplier: int = 4  # d_ff = d_ff_multiplier * d_model
    # Note: dropout hardcoded to 0.0 in model (per paper)

    # Training (from paper: 80K steps)
    max_steps: int = 80000  # Gradient steps
    max_tokens: int = 4096  # Tokens per batch (max-tokens batching)
    warmup_steps: int = 4000  # LR warmup steps
    learning_rate: float = 3e-4  # Peak learning rate after warmup
    optimizer: str = "adam_w"  # AdamW with beta1=0.9, beta2=0.98, eps=1e-9

    # Regularization
    label_smoothing: Optional[float] = None  # Label smoothing factor

    # Data
    subsample_seed: int = 674931  # Fixed seed for reproducibility

    # Variance mode: train num_splits independent models per d_model on
    # disjoint subsets of IWSLT train, with chunk num_splits reserved as
    # an in-distribution held-out test set. See TRANSFORMER_PLAN.md.
    variance: bool = False
    num_splits: int = 4
    samples_per_split: int = 32000  # 4 train + 1 held-out test = 160K (fits IWSLT-14)

    # Mixed precision: BF16 autocast on forward+backward, FP32 weights / grads
    # / optimizer state. Same protocol as resnet18 trainer.
    use_bf16: bool = True

    # Logging
    log_interval: int = 100  # Log train metrics every N steps
    eval_interval: int = 100  # Evaluate on valid set every N steps
