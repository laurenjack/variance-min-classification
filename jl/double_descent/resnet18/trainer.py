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
    load_cifar10_with_noise_val_split_gpu,
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

    # Load data (each process loads independently).  When val_split is on
    # we use the GPU-resident pipeline: CIFAR-10 lives on this worker's GPU
    # as a normalized FP32 tensor, and RandomCrop/HFlip are batched tensor
    # ops on-device.  Removes the CPU augmentation bottleneck.
    val_loader = None
    if getattr(config, "use_val_split", False):
        train_loader, val_loader, test_loader, mislabel_mask = (
            load_cifar10_with_noise_val_split_gpu(
                noise_prob=config.label_noise,
                batch_size=config.batch_size,
                device=device,
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
    residuals_dir = Path(output_path) / "residuals"
    residuals_dir.mkdir(parents=True, exist_ok=True)
    es_residuals_dir = es_dir / "residuals"
    es_residuals_dir.mkdir(parents=True, exist_ok=True)
    correct_mask = ~mislabel_mask
    best_val_loss = float("inf")
    best_val_epoch = 0

    # Per-point + scalar residual accumulators.
    #   residual_sum : per-point raw cumulative sum (per step the point is in batch).
    #   residual_m   : per-point EMA(r) with beta1 (sparse update, decay every step).
    #   residual_v   : SCALAR EMA over time of mean_i(r_i^2) across the batch,
    #                  decayed every step — v_t = beta2 * v_{t-1} +
    #                  (1 - beta2) * mean(g^2).  Shared across points,
    #                  mirroring Adam's batch-averaged second-moment signal.
    #   tau_accum    : per-point cumulative sum of tau_it = m_hat_it /
    #                  (sqrt(v_hat_t) + eps), accumulated only at steps where
    #                  the point appears in the batch.  This is the
    #                  Adam-style normalized contribution accumulated over
    #                  training — different sampling times see different v_t,
    #                  so v does NOT cancel in the share (unlike a snapshot
    #                  m_T / sqrt(v_T)).
    # All accumulators are snapshotted at every val_loss improvement.
    n_train = len(train_loader.dataset)
    residual_sum = torch.zeros(n_train, device=device, dtype=torch.float64)
    residual_m = torch.zeros(n_train, device=device, dtype=torch.float64)
    residual_v = torch.zeros((), device=device, dtype=torch.float64)
    tau_accum = torch.zeros(n_train, device=device, dtype=torch.float64)
    total_steps = 0
    residual_sum_es = torch.zeros_like(residual_sum)
    residual_m_es = torch.zeros_like(residual_m)
    residual_v_es = torch.zeros_like(residual_v)
    tau_accum_es = torch.zeros_like(tau_accum)
    total_steps_es = 0
    pg = optimizer.param_groups[0]
    betas = pg.get("betas", (0.9, 0.999))
    beta1 = float(betas[0])
    beta2 = float(betas[1])
    adam_eps = float(pg.get("eps", 1e-8))

    use_bf16 = getattr(config, "use_bf16", True)
    if use_bf16:
        print(f"[GPU {gpu_id}] {model_label} BF16 autocast enabled")

    def _save_residual_snapshot(
        out_dir: Path,
        sum_t: torch.Tensor,
        tau_t: torch.Tensor,
        v_scalar_t: torch.Tensor,
        steps: int,
        snapshot_kind: str,
        snapshot_epoch: int,
    ) -> None:
        """Write 4 .npy files (raw avg + accumulated tau, split by mislabel mask) + meta.json."""
        avg = (sum_t / max(steps, 1)).cpu().numpy()
        # tau_t is already the cumulative sum of per-step tau_it across the
        # point's batch visits.  Save as-is; share = sum(tau_mis) / sum(tau_total)
        # is not invariant under the scalar v_t (varies across the visited steps).
        tau = tau_t.cpu().numpy()
        np.save(out_dir / f"{model_label}_mislabel.npy", avg[mislabel_mask])
        np.save(out_dir / f"{model_label}_correct.npy", avg[correct_mask])
        np.save(out_dir / f"{model_label}_mislabel_adam.npy", tau[mislabel_mask])
        np.save(out_dir / f"{model_label}_correct_adam.npy", tau[correct_mask])
        v_value = float(v_scalar_t.item())
        meta = {
            **model_params,
            "n_train": int(n_train),
            "n_mislabel": int(mislabel_mask.sum()),
            "n_correct": int(correct_mask.sum()),
            "total_steps": int(steps),
            "snapshot_kind": snapshot_kind,
            "snapshot_epoch": int(snapshot_epoch),
            "best_val_epoch": int(best_val_epoch),
            "best_val_loss": float(best_val_loss),
            "beta1": beta1,
            "beta2": beta2,
            "adam_eps": adam_eps,
            "v_scalar": v_value,
        }
        with open(out_dir / f"{model_label}_meta.json", "w") as f:
            json.dump(meta, f, indent=2)

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
                residual_m.mul_(beta1)
                residual_m.index_add_(0, indices, residual * (1.0 - beta1))
                # Scalar v: EMA of mean(g^2) across the batch.
                batch_mean_sq = (residual * residual).mean()
                residual_v.mul_(beta2).add_(batch_mean_sq, alpha=1.0 - beta2)
                total_steps += 1
                # Accumulate per-batch-point Adam-style tau_it = m_hat / sqrt(v_hat).
                bc_m = 1.0 - beta1 ** total_steps
                bc_v = 1.0 - beta2 ** total_steps
                m_hat_batch = residual_m.index_select(0, indices) / bc_m
                v_hat = residual_v / bc_v
                tau_step = m_hat_batch / (v_hat.sqrt() + adam_eps)
                tau_accum.index_add_(0, indices, tau_step)
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

        # Save early-stop checkpoint when val_loss improves.  Mirrors the
        # residual accumulators to disk in the same step so we always have a
        # consistent on-disk ES snapshot (model.pt + residuals) — useful if
        # training is killed before completion.
        if val_loss is not None and val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_epoch = epoch + 1
            torch.save(model.state_dict(), es_model_path)
            residual_sum_es.copy_(residual_sum)
            residual_m_es.copy_(residual_m)
            residual_v_es.copy_(residual_v)
            tau_accum_es.copy_(tau_accum)
            total_steps_es = total_steps
            _save_residual_snapshot(
                es_residuals_dir,
                residual_sum_es, tau_accum_es, residual_v_es,
                total_steps_es, snapshot_kind="ES",
                snapshot_epoch=best_val_epoch,
            )

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

    # Save FINAL per-point residuals.  ES residuals are already on disk —
    # we re-write them once more here so the meta.json reflects the final
    # best_val_epoch/best_val_loss values.
    _save_residual_snapshot(
        residuals_dir,
        residual_sum, tau_accum, residual_v, total_steps,
        snapshot_kind="FINAL", snapshot_epoch=config.epochs,
    )
    _save_residual_snapshot(
        es_residuals_dir,
        residual_sum_es, tau_accum_es, residual_v_es, total_steps_es,
        snapshot_kind="ES", snapshot_epoch=best_val_epoch,
    )
    avg_final = (residual_sum / max(total_steps, 1)).cpu().numpy()
    avg_es = (residual_sum_es / max(total_steps_es, 1)).cpu().numpy()
    tau_final_np = tau_accum.cpu().numpy()
    tau_es_np = tau_accum_es.cpu().numpy()
    print(
        f"[GPU {gpu_id}] {model_label} residuals saved: "
        f"FINAL avg mis={avg_final[mislabel_mask].mean():.4f} correct={avg_final[correct_mask].mean():.4f}; "
        f"FINAL tau mis={tau_final_np[mislabel_mask].mean():.4f} correct={tau_final_np[correct_mask].mean():.4f}; "
        f"ES avg mis={avg_es[mislabel_mask].mean():.4f} correct={avg_es[correct_mask].mean():.4f}; "
        f"ES tau mis={tau_es_np[mislabel_mask].mean():.4f} correct={tau_es_np[correct_mask].mean():.4f}"
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
