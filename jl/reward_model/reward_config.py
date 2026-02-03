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
        early_stopping: Whether to use early stopping
        patience: Number of epochs without improvement before stopping
        log_timing: Whether to log performance timing information
        smoke_test: If True, run only 50 steps for quick validation
    """
    model_name: str = "meta-llama/Llama-3.2-1B-Instruct"
    hf_dataset: str = "Anthropic/hh-rlhf"
    max_length: int = 1024
    train_batch_size: int = 4
    eval_batch_size: int = 4
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    num_epochs: int = 3
    log_interval: int = 100
    early_stopping: bool = True
    patience: int = 2
    log_timing: bool = True
    smoke_test: bool = True

