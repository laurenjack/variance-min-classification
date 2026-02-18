"""Shared configuration for reward model training."""

from dataclasses import dataclass


@dataclass
class RewardConfig:
    """Configuration for reward model training.

    Attributes:
        dataset: Which dataset to use ("anthropic-hh" or "helpsteer2")
        model_name: HuggingFace model identifier for the base LLM
        max_length: Maximum sequence length for tokenization
        train_batch_size: Batch size for training
        eval_batch_size: Batch size for evaluation
        learning_rate: Learning rate for AdamW optimizer
        weight_decay: Weight decay for AdamW optimizer
        num_epochs: Maximum number of training epochs
        log_interval: Log training metrics every N steps
        log_timing: Whether to log performance timing information
        warmup_steps: Number of steps for quadratic LR warmup
        min_lr_ratio: Minimum LR as fraction of initial LR (for cosine decay)
        smoke_test: If True, exit training early after smoke_test_steps steps
            (LR schedule is still computed from full dataset, this just cuts the run short)
        smoke_test_steps: Number of steps to run when smoke_test is True
    """
    dataset: str = "helpsteer2"
    model_name: str = "meta-llama/Llama-3.2-1B-Instruct"
    max_length: int = 1024
    train_batch_size: int = 128
    eval_batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    num_epochs: int = 30
    log_interval: int = 1
    log_timing: bool = True
    warmup_steps: int = 30
    min_lr_ratio: float = 0.1
    smoke_test: bool = False
    smoke_test_steps: int = 200
