"""ResNet18 trainer that excludes mislabeled (high-entropy) training images.

Companion to bucket_shadow_trainer.py: trains the same architecture under
the same recipe, but on only the clean subset of the 45k train pool
(images whose noisy label equals the original CIFAR label, ~85% by
construction at noise_prob=0.15).

Same val + test as the standard pipeline. Same epoch budget as the
shadow run for an apples-to-apples comparison ("same recipe, just
less data"). No shadow tracking.
"""

import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast

from jl.double_descent.resnet18.resnet18_data import (
    GPUTrainLoader,
    GPUEvalLoader,
    _apply_label_noise_tensor,
    _build_gpu_cifar_tensors,
    compute_val_split_indices,
    DEFAULT_NOISE_SEED,
    DEFAULT_VAL_SIZE,
    VAL_SPLIT_SEED,
)
from jl.double_descent.resnet18.trainer import (
    evaluate,
    make_cosine_decay_scheduler,
)


def _build_clean_only_loaders(
    noise_prob: float,
    batch_size: int,
    device: torch.device,
    data_dir: str,
    data_augmentation: bool,
    noise_seed: int = DEFAULT_NOISE_SEED,
    val_size: int = DEFAULT_VAL_SIZE,
    val_split_seed: int = VAL_SPLIT_SEED,
):
    """Build train (clean-only), val, test loaders.

    train loader covers only images in the 45k train pool whose noisy
    label == original label. val + test labels follow the standard
    pipeline.
    """
    train_images, train_orig, test_images, test_labels = (
        _build_gpu_cifar_tensors(data_dir, device)
    )
    train_noisy = _apply_label_noise_tensor(train_orig, noise_prob, noise_seed)
    mis_mask_50k = (train_noisy != train_orig).long()  # [50000] on device

    train_indices, val_indices = compute_val_split_indices(
        total_samples=train_images.size(0),
        val_size=val_size,
        seed=val_split_seed,
    )
    train_idx_t = torch.from_numpy(train_indices).to(device)
    val_idx_t = torch.from_numpy(val_indices).to(device)

    # Filter the train subset to clean only.
    train_buckets_sub = mis_mask_50k.index_select(0, train_idx_t)
    keep_mask = (train_buckets_sub == 0)
    keep_idx = train_idx_t[keep_mask]

    train_sub_imgs = train_images.index_select(0, keep_idx).contiguous()
    train_sub_labels = train_noisy.index_select(0, keep_idx).contiguous()
    val_sub_imgs = train_images.index_select(0, val_idx_t).contiguous()
    val_sub_labels = train_noisy.index_select(0, val_idx_t).contiguous()

    train_loader = GPUTrainLoader(
        train_sub_imgs, train_sub_labels, batch_size,
        augment=data_augmentation, drop_last=True,
    )
    val_loader = GPUEvalLoader(val_sub_imgs, val_sub_labels, batch_size)
    test_loader = GPUEvalLoader(test_images, test_labels, batch_size)
    n_kept = int(keep_idx.size(0))
    n_full = int(train_idx_t.size(0))
    return train_loader, val_loader, test_loader, n_kept, n_full


def train_single_model_clean_only(
    gpu_id: int,
    model_factory: Callable[[], nn.Module],
    model_label: str,
    model_params: Dict[str, Any],
    config,
    output_path: str,
    data_path: str,
) -> None:
    """Train a model on clean-only train data (same recipe as the shadow
    run, minus the mislabeled images).

    Outputs in output_path:
      - metrics_{label}.jsonl
      - model_{label}.pt
      - early_stop/model_{label}.pt
    """
    device = torch.device(f"cuda:{gpu_id}")
    # Seed-locked init keyed by the width-style integer in model_params so a
    # given (k or d_model) reproduces across runs.
    seed_key = model_params.get("k", model_params.get("d_model", 0))
    torch.manual_seed(42 + int(seed_key))

    print(f"[GPU {gpu_id}] [clean-only] Training {model_label} on {device}")

    train_loader, val_loader, test_loader, n_clean, n_full = (
        _build_clean_only_loaders(
            noise_prob=config.label_noise,
            batch_size=config.batch_size,
            device=device,
            data_dir=data_path,
            data_augmentation=config.data_augmentation,
        )
    )
    print(
        f"[GPU {gpu_id}] {model_label} clean-only data: "
        f"{n_clean}/{n_full} kept ({n_clean / n_full:.3f})"
    )

    model = model_factory().to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"[GPU {gpu_id}] {model_label} has {num_params:,} parameters")

    if config.optimizer == "adam_w":
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    scheduler = None
    if config.cosine_decay_epoch is not None:
        scheduler = make_cosine_decay_scheduler(
            optimizer, config.cosine_decay_epoch, config.epochs
        )

    output_path_p = Path(output_path)
    output_path_p.mkdir(parents=True, exist_ok=True)
    metrics_path = output_path_p / f"metrics_{model_label}.jsonl"
    metrics_path.write_text("")

    es_dir = output_path_p / "early_stop"
    es_dir.mkdir(parents=True, exist_ok=True)
    es_model_path = es_dir / f"model_{model_label}.pt"
    best_val_loss = float("inf")
    best_val_epoch = 0

    use_bf16 = getattr(config, "use_bf16", True)

    model.train()
    for epoch in range(config.epochs):
        epoch_start = time.time()
        train_loss_sum, train_correct, train_samples = 0.0, 0, 0

        for images, labels in train_loader:
            B = images.size(0)
            optimizer.zero_grad()
            with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
                logits = model(images)
                loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                train_loss_sum += loss.item() * B
                preds = logits.argmax(dim=1)
                train_correct += int((preds == labels).sum().item())
                train_samples += B

        train_loss = train_loss_sum / max(1, train_samples)
        train_error = 1.0 - (train_correct / max(1, train_samples))
        test_error, test_loss = evaluate(model, test_loader, device)
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
            "val_error": val_error,
            "val_loss": val_loss,
        }
        with open(metrics_path, "a") as f:
            f.write(json.dumps(metrics) + "\n")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_epoch = epoch + 1
            torch.save(model.state_dict(), es_model_path)

        if (epoch + 1) % config.log_interval == 0 or epoch == 0:
            print(
                f"[GPU {gpu_id}] {model_label} Epoch {epoch + 1:4d}/{config.epochs} | "
                f"train_err={train_error:.4f} test_err={test_error:.4f} "
                f"val_err={val_error:.4f} val_loss={val_loss:.4f} | "
                f"train_loss={train_loss:.4f} test_loss={test_loss:.4f} | "
                f"{epoch_time:.1f}s"
            )

    model_path = output_path_p / f"model_{model_label}.pt"
    torch.save(model.state_dict(), model_path)
    print(
        f"[GPU {gpu_id}] {model_label} DONE clean-only. "
        f"best_val_loss={best_val_loss:.4f} at epoch {best_val_epoch}"
    )

    from jl.double_descent.resnet18.evaluation import compute_final_metrics
    compute_final_metrics(
        model, test_loader, metrics_path, output_path_p, model_label,
        model_params, device, val_loader=val_loader,
        best_val_epoch=best_val_epoch, best_val_loss=best_val_loss,
    )
    if es_model_path.exists():
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
