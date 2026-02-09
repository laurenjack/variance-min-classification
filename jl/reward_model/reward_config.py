"""Shared configuration for reward model training."""

from dataclasses import dataclass


@dataclass
class RewardConfig:
    """Configuration for reward model training.
    
    Attributes:
        is_multi: If True, use DDP for multi-GPU training (requires torchrun).
            If False, run on a single GPU without DDP.
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
        warmup_ratio: Fraction of total steps for linear LR warmup
        min_lr_ratio: Minimum LR as fraction of initial LR (for cosine decay)
        smoke_test: If True, exit training early after smoke_test_steps steps
            (LR schedule is still computed from full dataset, this just cuts the run short)
        smoke_test_steps: Number of steps to run when smoke_test is True
    """
    is_multi: bool = False
    model_name: str = "meta-llama/Llama-3.2-1B-Instruct"
    hf_dataset: str = "Anthropic/hh-rlhf"
    max_length: int = 1024
    # Batch size per GPU (total effective batch = train_batch_size * 8 GPUs)
    train_batch_size: int = 32
    eval_batch_size: int = 32
    learning_rate: float = 3e-5
    weight_decay: float = 0.01
    num_epochs: int = 1
    log_interval: int = 100
    log_timing: bool = True
    warmup_ratio: float = 0.20
    min_lr_ratio: float = 0.1
    smoke_test: bool = True
    smoke_test_steps: int = 200

