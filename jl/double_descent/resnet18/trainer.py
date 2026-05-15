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
from torch.amp import autocast
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from jl.double_descent.resnet18.resnet18_data import (
    load_cifar10_with_noise,
    load_cifar10_with_noise_val_split,
    load_cifar10_with_noise_val_split_gpu,
    load_cifar10_variance_split_gpu,
)


def make_cosine_decay_scheduler(
    optimizer: torch.optim.Optimizer,
    cosine_decay_epoch: int,
    total_epochs: int,
    min_lr_ratio: float = 0.0,
) -> LambdaLR:
    """Create a scheduler that holds LR constant then cosine decays to min_lr_ratio."""
    decay_epochs = total_epochs - cosine_decay_epoch

    def lr_lambda(epoch: int) -> float:
        if epoch < cosine_decay_epoch:
            return 1.0
        progress = (epoch - cosine_decay_epoch) / decay_epochs
        return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


def evaluate(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
    """Evaluate model on a dataset. Returns (error_rate, average_loss)."""
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
) -> None:
    """Train a single model on the specified GPU.

    Designed to be called from a multiprocessing worker.

    Args:
        gpu_id: GPU device ID.
        model_factory: Zero-arg callable returning an nn.Module (use functools.partial).
        model_label: Label for filenames, e.g. "k4" or "k4_split0".
        model_params: Dict merged into every metrics line, e.g. {"k": 4} or
            {"k": 4, "split_id": 0}.
        config: Training configuration (duck-typed).
        output_path: Directory to save metrics.
        data_path: Path to CIFAR-10 data.
    """
    device = torch.device(f"cuda:{gpu_id}")

    print(f"[GPU {gpu_id}] Training {model_label} on {device}")

    # Load data (each process loads independently). When val_split or
    # variance mode is on we use the GPU-resident pipeline: CIFAR-10 lives
    # on this worker's GPU as a normalized FP32 tensor, and RandomCrop/HFlip
    # are batched tensor ops on-device.
    val_loader = None
    split_id = model_params.get("split_id")
    if split_id is not None:
        # Variance mode: each (k, split_id) model gets a disjoint training
        # chunk. val + test are shared across all splits.
        train_loader, val_loader, test_loader = (
            load_cifar10_variance_split_gpu(
                noise_prob=config.label_noise,
                batch_size=config.batch_size,
                device=device,
                split_id=split_id,
                num_splits=config.num_splits,
                data_augmentation=config.data_augmentation,
                data_dir=data_path,
            )
        )
    elif getattr(config, "use_val_split", False):
        train_loader, val_loader, test_loader = (
            load_cifar10_with_noise_val_split_gpu(
                noise_prob=config.label_noise,
                batch_size=config.batch_size,
                device=device,
                data_augmentation=config.data_augmentation,
                data_dir=data_path,
            )
        )
    else:
        train_loader, test_loader = load_cifar10_with_noise(
            noise_prob=config.label_noise,
            batch_size=config.batch_size,
            data_augmentation=config.data_augmentation,
            data_dir=data_path,
        )

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

    os.makedirs(output_path, exist_ok=True)
    metrics_path = Path(output_path) / f"metrics_{model_label}.jsonl"

    # Early-stop checkpoint setup. Requires val tracking; fail-fast otherwise.
    if val_loader is None:
        raise RuntimeError(
            "Early-stop checkpoint saving requires --val-split (val_loader is None)."
        )
    es_dir = Path(output_path) / "early_stop"
    es_dir.mkdir(parents=True, exist_ok=True)
    es_model_path = es_dir / f"model_{model_label}.pt"
    best_val_loss = float("inf")
    best_val_epoch = 0

    use_bf16 = getattr(config, "use_bf16", True)
    if use_bf16:
        print(f"[GPU {gpu_id}] {model_label} BF16 autocast enabled")

    # Clear metrics file
    with open(metrics_path, 'w') as f:
        pass

    # Training loop
    model.train()
    for epoch in range(config.epochs):
        epoch_start = time.time()

        # Accumulate training metrics during training (no extra forward pass).
        train_loss_sum = 0.0
        train_correct = 0
        train_samples = 0

        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            batch_size = images.shape[0]

            optimizer.zero_grad()
            with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
                logits = model(images)
                loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                train_loss_sum += loss.item() * batch_size
                preds = logits.argmax(dim=1)
                train_correct += (preds == labels).sum().item()
                train_samples += batch_size

        train_loss = train_loss_sum / train_samples
        train_error = 1.0 - (train_correct / train_samples)

        test_error, test_loss = evaluate(model, test_loader, device)

        val_error: Optional[float] = None
        val_loss: Optional[float] = None
        if val_loader is not None:
            val_error, val_loss = evaluate(model, val_loader, device)

        if scheduler is not None:
            scheduler.step()

        epoch_time = time.time() - epoch_start
        metrics = {
            "epoch": epoch + 1,
            **model_params,
            "train_error": train_error,
            "test_error": test_error,
            "train_loss": train_loss,
            "test_loss": test_loss,
        }
        if val_loader is not None:
            metrics["val_error"] = val_error
            metrics["val_loss"] = val_loss
        with open(metrics_path, 'a') as f:
            f.write(json.dumps(metrics) + "\n")

        # Save early-stop checkpoint when val_loss improves.
        if val_loss is not None and val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_epoch = epoch + 1
            torch.save(model.state_dict(), es_model_path)

        if (epoch + 1) % config.log_interval == 0 or epoch == 0:
            val_str = (
                f" val_err={val_error:.4f} val_loss={val_loss:.4f} |"
                if val_loader is not None
                else ""
            )
            print(
                f"[GPU {gpu_id}] {model_label} Epoch {epoch + 1:4d}/{config.epochs} | "
                f"train_err={train_error:.4f} test_err={test_error:.4f} |{val_str} "
                f"train_loss={train_loss:.4f} test_loss={test_loss:.4f} | "
                f"{epoch_time:.1f}s"
            )

    # Save final model
    model_path = Path(output_path) / f"model_{model_label}.pt"
    torch.save(model.state_dict(), model_path)
    print(f"[GPU {gpu_id}] {model_label} training complete! Model saved to {model_path}")

    # Compute and save final evaluation metrics
    from jl.double_descent.resnet18.evaluation import compute_final_metrics
    eval_output = Path(output_path)
    compute_final_metrics(
        model, test_loader, metrics_path, eval_output, model_label,
        model_params, device, val_loader=val_loader,
        best_val_epoch=best_val_epoch, best_val_loss=best_val_loss,
    )

    # Also evaluate the early-stop checkpoint (mirrors the transformer pattern).
    print(
        f"[GPU {gpu_id}] {model_label} evaluating early-stop checkpoint "
        f"(best val_loss={best_val_loss:.4f} at epoch {best_val_epoch})"
    )
    es_model = model_factory().to(device)
    es_model.load_state_dict(
        torch.load(es_model_path, map_location=device, weights_only=True)
    )
    es_model.eval()
    compute_final_metrics(
        es_model, test_loader, metrics_path, es_dir, model_label,
        model_params, device, val_loader=val_loader,
        best_val_epoch=best_val_epoch, best_val_loss=best_val_loss,
    )
    del es_model
