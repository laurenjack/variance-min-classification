"""Single model training for double descent experiments.

Model-agnostic: accepts a model factory callable so any architecture can be trained.
"""

import json
import math
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from jl.double_descent.resnet18.resnet18_data import load_cifar10_split, load_cifar10_with_noise


def make_cosine_decay_scheduler(
    optimizer: torch.optim.Optimizer,
    cosine_decay_epoch: int,
    total_epochs: int,
    min_lr_ratio: float = 0.0,
) -> LambdaLR:
    """Create a scheduler that holds LR constant then cosine decays to min_lr_ratio.

    Args:
        optimizer: The optimizer to schedule.
        cosine_decay_epoch: Epoch at which to start cosine decay.
        total_epochs: Total number of training epochs.
        min_lr_ratio: Final LR as fraction of initial (default 0.0).

    Returns:
        LambdaLR scheduler.
    """
    decay_epochs = total_epochs - cosine_decay_epoch

    def lr_lambda(epoch: int) -> float:
        if epoch < cosine_decay_epoch:
            return 1.0
        # Cosine decay from 1.0 to min_lr_ratio over remaining epochs
        progress = (epoch - cosine_decay_epoch) / decay_epochs
        return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


def evaluate(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
    """Evaluate model on a dataset.

    Args:
        model: The model to evaluate.
        data_loader: Data loader for evaluation.
        device: Device to run on.

    Returns:
        Tuple of (error_rate, average_loss).
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            batch_size = images.shape[0]

            logits = model(images)
            loss = F.cross_entropy(logits, labels, reduction='sum')

            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += batch_size

    avg_loss = total_loss / total_samples
    error_rate = 1.0 - (total_correct / total_samples)

    model.train()
    return error_rate, avg_loss


def train_single_model(
    gpu_id: int,
    model_factory: Callable[[], nn.Module],
    model_label: str,
    model_params: Dict[str, Any],
    config,
    output_path: str,
    data_path: str,
    split_id: Optional[int] = None,
    num_splits: Optional[int] = None,
) -> None:
    """Train a single model on the specified GPU.

    This function is designed to be called from a multiprocessing worker.

    Args:
        gpu_id: GPU device ID (0-7).
        model_factory: Zero-arg callable returning an nn.Module (use functools.partial).
        model_label: Label for filenames, e.g. "k4" or "n3" (used in metrics_{label}.jsonl).
        model_params: Dict merged into every metrics line, e.g. {"k": 4} or {"n": 3}.
        config: Training configuration (duck-typed: needs epochs, batch_size, etc.).
        output_path: Directory to save metrics.
        data_path: Path to CIFAR-10 data.
        split_id: For variance mode, which disjoint split to use (0 to num_splits-1).
        num_splits: For variance mode, total number of disjoint splits.
    """
    device = torch.device(f"cuda:{gpu_id}")

    # Determine if this is variance mode
    variance_mode = split_id is not None and num_splits is not None

    if variance_mode:
        print(f"[GPU {gpu_id}] Training {model_label}, split={split_id} on {device}")
    else:
        print(f"[GPU {gpu_id}] Training {model_label} on {device}")

    # Load data (each process loads independently)
    if variance_mode:
        train_loader, test_loader = load_cifar10_split(
            split_id=split_id,
            num_splits=num_splits,
            noise_prob=config.label_noise,
            batch_size=config.batch_size,
            data_augmentation=config.data_augmentation,
            data_dir=data_path,
        )
    else:
        train_loader, test_loader = load_cifar10_with_noise(
            noise_prob=config.label_noise,
            batch_size=config.batch_size,
            data_augmentation=config.data_augmentation,
            data_dir=data_path,
        )

    # Fix initialization for variance mode: all splits for the same model config
    # get identical initial weights, so Jensen Gap measures only data variance.
    if variance_mode:
        torch.manual_seed(42)

    # Create model
    model = model_factory().to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"[GPU {gpu_id}] {model_label} has {num_params:,} parameters")

    # Optimizer
    if config.optimizer == "adam_w":
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    # Learning rate scheduler (optional cosine decay)
    scheduler = None
    if config.cosine_decay_epoch is not None:
        scheduler = make_cosine_decay_scheduler(
            optimizer, config.cosine_decay_epoch, config.epochs
        )
        print(f"[GPU {gpu_id}] Cosine decay from epoch {config.cosine_decay_epoch} to {config.epochs} (to 0)")

    # Metrics file path depends on mode
    os.makedirs(output_path, exist_ok=True)
    if variance_mode:
        metrics_path = Path(output_path) / f"metrics_{model_label}_split{split_id}.jsonl"
    else:
        metrics_path = Path(output_path) / f"metrics_{model_label}.jsonl"

    # Clear metrics file
    with open(metrics_path, 'w') as f:
        pass

    # Training loop
    model.train()
    for epoch in range(config.epochs):
        epoch_start = time.time()

        # Accumulate training metrics during training
        train_loss_sum = 0.0
        train_correct = 0
        train_samples = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            batch_size = images.shape[0]

            optimizer.zero_grad()
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()

            # Accumulate metrics (no extra forward pass needed)
            with torch.no_grad():
                train_loss_sum += loss.item() * batch_size
                preds = logits.argmax(dim=1)
                train_correct += (preds == labels).sum().item()
                train_samples += batch_size

        # Compute training metrics from accumulated values
        train_loss = train_loss_sum / train_samples
        train_error = 1.0 - (train_correct / train_samples)

        # Only evaluate on test set (separate pass required)
        test_error, test_loss = evaluate(model, test_loader, device)

        # Step the scheduler if using cosine decay
        if scheduler is not None:
            scheduler.step()

        # Log metrics
        epoch_time = time.time() - epoch_start
        metrics = {
            "epoch": epoch + 1,
            **model_params,
            "train_error": train_error,
            "test_error": test_error,
            "train_loss": train_loss,
            "test_loss": test_loss,
        }
        if variance_mode:
            metrics["split_id"] = split_id
        with open(metrics_path, 'a') as f:
            f.write(json.dumps(metrics) + "\n")

        # Print progress
        if (epoch + 1) % config.log_interval == 0 or epoch == 0:
            split_str = f", split={split_id}" if variance_mode else ""
            print(
                f"[GPU {gpu_id}] {model_label}{split_str} Epoch {epoch + 1:4d}/{config.epochs} | "
                f"train_err={train_error:.4f} test_err={test_error:.4f} | "
                f"train_loss={train_loss:.4f} test_loss={test_loss:.4f} | "
                f"{epoch_time:.1f}s"
            )

    # Save final model
    if variance_mode:
        model_path = Path(output_path) / f"model_{model_label}_split{split_id}.pt"
    else:
        model_path = Path(output_path) / f"model_{model_label}.pt"
    torch.save(model.state_dict(), model_path)
    split_str = f", split={split_id}" if variance_mode else ""
    print(f"[GPU {gpu_id}] {model_label}{split_str} training complete! Model saved to {model_path}")

    # Compute and save final evaluation metrics (main runs only)
    if not variance_mode:
        from jl.double_descent.resnet18.evaluation import compute_final_metrics
        eval_output = Path(output_path)
        compute_final_metrics(model, test_loader, metrics_path, eval_output, model_label, model_params, device)
