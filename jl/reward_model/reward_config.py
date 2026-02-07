"""Shared configuration for reward model training."""

from dataclasses import dataclass


@dataclass
class RewardConfig:
    """Configuration for reward model training.
    
    Attributes:
        model_name: HuggingFace model identifier for the base LLM
        hf_dataset: HuggingFace dataset identifier for training data
        max_length: Maximum sequence length for tokenization
        train_batch_size: Batch size for training
        eval_batch_size: Batch size for evaluation
        learning_rate: Learning rate for AdamW optimizer
        weight_decay: Weight decay for AdamW optimizer
        num_epochs: Maximum number of training epochs
        log_interval: Log training metrics every N steps
        log_timing: Whether to log performance timing information
        smoke_test: If True, run only 50 steps for quick validation
        warmup_ratio: Fraction of total steps for linear LR warmup
        min_lr_ratio: Minimum LR as fraction of initial LR (for cosine decay)
    """
    model_name: str = "meta-llama/Llama-3.2-1B-Instruct"
    hf_dataset: str = "Anthropic/hh-rlhf"
    max_length: int = 1024
    # Batch size per GPU (total effective batch = train_batch_size * 8 GPUs)
    train_batch_size: int = 32
    eval_batch_size: int = 32
    learning_rate: float = 3e-6
    weight_decay: float = 0.01
    num_epochs: int = 1
    log_interval: int = 100
    log_timing: bool = True
    smoke_test: bool = False
    warmup_ratio: float = 0.20
    min_lr_ratio: float = 0.1

