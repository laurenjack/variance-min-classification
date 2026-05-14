"""Single model training for double descent experiments.

Model-agnostic: accepts a model factory callable so any architecture can be trained.
"""

import json
import math
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from jl.double_descent.resnet18.resnet18_data import (
    load_cifar10_with_noise,
    load_cifar10_with_noise_val_split,
)


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
    """
    device = torch.device(f"cuda:{gpu_id}")

    print(f"[GPU {gpu_id}] Training {model_label} on {device}")

    # Load data (each process loads independently)
    val_loader: Optional[DataLoader] = None
    if getattr(config, "use_val_split", False):
        train_loader, val_loader, test_loader, mislabel_mask = (
            load_cifar10_with_noise_val_split(
                noise_prob=config.label_noise,
                batch_size=config.batch_size,
                data_augmentation=config.data_augmentation,
                data_dir=data_path,
            )
        )
    else:
        train_loader, test_loader, mislabel_mask = load_cifar10_with_noise(
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

    # Early-stop checkpoint setup.  Requires val tracking; fail-fast otherwise.
    if val_loader is None:
        raise RuntimeError(
            "Early-stop checkpoint saving requires --val-split (val_loader is None)."
        )
    es_dir = Path(output_path) / "early_stop"
    es_dir.mkdir(parents=True, exist_ok=True)
    es_model_path = es_dir / f"model_{model_label}.pt"
    best_val_loss = float("inf")
    best_val_epoch = 0

    # Per-point residual accumulators: r_i = 1 - p(assigned class).
    #   residual_sum     : raw cumulative sum (per step the point participates).
    #   residual_ema     : Adam-style EMA with beta1: every step we apply
    #                      m <- beta1 * m, then add (1 - beta1) * r for points
    #                      in the batch.  Mirrors first-moment update with
    #                      sparse gradients.  Bias-corrected at save time by
    #                      dividing by (1 - beta1**total_steps).
    # Both vectors are snapshotted whenever val_loss improves so the ES
    # checkpoint captures contributions accumulated *up to* the early-stop epoch.
    n_train = len(train_loader.dataset)
    residual_sum = torch.zeros(n_train, device=device, dtype=torch.float64)
    residual_ema = torch.zeros(n_train, device=device, dtype=torch.float64)
    total_steps = 0
    residual_sum_es = torch.zeros_like(residual_sum)
    residual_ema_es = torch.zeros_like(residual_ema)
    total_steps_es = 0
    if (
        hasattr(optimizer, "param_groups")
        and "betas" in optimizer.param_groups[0]
    ):
        beta1 = float(optimizer.param_groups[0]["betas"][0])
    else:
        beta1 = 0.9

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

        # Accumulate training metrics during training
        train_loss_sum = 0.0
        train_correct = 0
        train_samples = 0

        for images, labels, indices in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            indices = indices.to(device, non_blocking=True)
            batch_size = images.shape[0]

            optimizer.zero_grad()
            with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
                logits = model(images)
                loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()

            # Accumulate metrics (no extra forward pass needed)
            with torch.no_grad():
                train_loss_sum += loss.item() * batch_size
                # Cast to FP32 before softmax for numerical safety in BF16.
                probs = F.softmax(logits.float(), dim=1)
                p_assigned = probs.gather(1, labels.unsqueeze(1)).squeeze(1)
                residual = (1.0 - p_assigned).double()
                residual_sum.index_add_(0, indices, residual)
                residual_ema.mul_(beta1)
                residual_ema.index_add_(0, indices, residual * (1.0 - beta1))
                total_steps += 1
                preds = logits.argmax(dim=1)
                train_correct += (preds == labels).sum().item()
                train_samples += batch_size

        # Compute training metrics from accumulated values
        train_loss = train_loss_sum / train_samples
        train_error = 1.0 - (train_correct / train_samples)

        # Only evaluate on test set (separate pass required)
        test_error, test_loss = evaluate(model, test_loader, device)

        # Val metrics (only when use_val_split is enabled)
        val_error: Optional[float] = None
        val_loss: Optional[float] = None
        if val_loader is not None:
            val_error, val_loss = evaluate(model, val_loader, device)

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
        if val_loader is not None:
            metrics["val_error"] = val_error
            metrics["val_loss"] = val_loss
        with open(metrics_path, 'a') as f:
            f.write(json.dumps(metrics) + "\n")

        # Save early-stop checkpoint when val_loss improves
        if val_loss is not None and val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_epoch = epoch + 1
            torch.save(model.state_dict(), es_model_path)
            residual_sum_es.copy_(residual_sum)
            residual_ema_es.copy_(residual_ema)
            total_steps_es = total_steps

        # Print progress
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

    # Save per-point average residuals (FINAL + ES), split by mislabel mask.
    # average = sum(1 - p_assigned) / number_of_training_steps_at_snapshot.
    residuals_dir = Path(output_path) / "residuals"
    residuals_dir.mkdir(parents=True, exist_ok=True)
    es_residuals_dir = es_dir / "residuals"
    es_residuals_dir.mkdir(parents=True, exist_ok=True)

    avg_final = (residual_sum / max(total_steps, 1)).cpu().numpy()
    avg_es = (residual_sum_es / max(total_steps_es, 1)).cpu().numpy()
    ema_bias_final = 1.0 - beta1 ** max(total_steps, 1)
    ema_bias_es = 1.0 - beta1 ** max(total_steps_es, 1)
    ema_final = (residual_ema / ema_bias_final).cpu().numpy()
    ema_es = (residual_ema_es / ema_bias_es).cpu().numpy()
    correct_mask = ~mislabel_mask

    np.save(residuals_dir / f"{model_label}_mislabel.npy", avg_final[mislabel_mask])
    np.save(residuals_dir / f"{model_label}_correct.npy", avg_final[correct_mask])
    np.save(residuals_dir / f"{model_label}_mislabel_ema.npy", ema_final[mislabel_mask])
    np.save(residuals_dir / f"{model_label}_correct_ema.npy", ema_final[correct_mask])
    np.save(es_residuals_dir / f"{model_label}_mislabel.npy", avg_es[mislabel_mask])
    np.save(es_residuals_dir / f"{model_label}_correct.npy", avg_es[correct_mask])
    np.save(es_residuals_dir / f"{model_label}_mislabel_ema.npy", ema_es[mislabel_mask])
    np.save(es_residuals_dir / f"{model_label}_correct_ema.npy", ema_es[correct_mask])

    meta = {
        **model_params,
        "n_train": int(n_train),
        "n_mislabel": int(mislabel_mask.sum()),
        "n_correct": int(correct_mask.sum()),
        "total_steps": int(total_steps),
        "total_steps_es": int(total_steps_es),
        "best_val_epoch": int(best_val_epoch),
        "best_val_loss": float(best_val_loss),
        "beta1": beta1,
    }
    with open(residuals_dir / f"{model_label}_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(
        f"[GPU {gpu_id}] {model_label} residuals saved: "
        f"FINAL avg mis={avg_final[mislabel_mask].mean():.4f} correct={avg_final[correct_mask].mean():.4f}; "
        f"FINAL ema mis={ema_final[mislabel_mask].mean():.4f} correct={ema_final[correct_mask].mean():.4f}; "
        f"ES avg mis={avg_es[mislabel_mask].mean():.4f} correct={avg_es[correct_mask].mean():.4f}; "
        f"ES ema mis={ema_es[mislabel_mask].mean():.4f} correct={ema_es[correct_mask].mean():.4f}"
    )

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
