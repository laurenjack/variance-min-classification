"""ResNet18 trainer that decomposes the AdamW trajectory into 2 buckets:
bucket 0 = clean (correctly-labeled) images, bucket 1 = mislabeled images.

Bucket assignment is the ground-truth mislabel mask (we know which images
we flipped). For each training step we run:

    g_t = g_t^0 + g_t^1                          (full-batch gradient)
    m_t^b = beta1 * m_{t-1}^b + (1-beta1) * g_t^b   (per-bucket first moment)
    v_t   = beta2 * v_{t-1} + (1-beta2) * g_t^2     (shared second moment)
    Delta W_t = -lr * m_hat_t / (sqrt(v_hat_t) + eps)
              = sum_b [ -lr * m_hat_t^b / (sqrt(v_hat_t) + eps) ]
              = sum_b Delta W_t^b
    shadow_b += Delta W_t^b

After training, ||shadow_b||_2 (global L2 over all params) is bucket b's
cumulative push to W via the AdamW update path. Weight decay
(W -= lr * wd * W) is W's self-shrinkage and is NOT attributed to either
bucket.

Save format mirrors the transformer's bucket_shadow_trainer:
  - bucket_shadows_{label}.pt : {shadows[2], param_names, bucket_sizes}
  - bucket_shares_{label}.json : norms + shares for diagnostic
  - model_{label}.pt           : final model state_dict
  - early_stop/model_{label}.pt: best-val checkpoint (val_loader required)
"""

import json
import logging
import math
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast

from jl.double_descent.resnet18.resnet18_data import (
    _apply_label_noise_tensor,
    _build_gpu_cifar_tensors,
    _normalize_buffers,
    compute_val_split_indices,
    gpu_random_crop_flip,
    DEFAULT_NOISE_SEED,
    DEFAULT_VAL_SIZE,
    VAL_SPLIT_SEED,
)
from jl.double_descent.resnet18.trainer import (
    evaluate,
    make_cosine_decay_scheduler,
)


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# GPU-resident loader that also yields per-image bucket id
# ---------------------------------------------------------------------------


class _GPUBucketTrainLoader:
    """Yields (images, labels, buckets) per batch. CIFAR + buckets stay on GPU."""

    def __init__(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        buckets: torch.Tensor,
        batch_size: int,
        augment: bool = True,
        drop_last: bool = True,
    ):
        assert images.size(0) == labels.size(0) == buckets.size(0)
        self.images = images
        self.labels = labels
        self.buckets = buckets
        self.batch_size = batch_size
        self.augment = augment
        self.drop_last = drop_last
        self.n = images.size(0)
        self._mean, self._std = _normalize_buffers(images.device)

    def __len__(self) -> int:
        if self.drop_last:
            return self.n // self.batch_size
        return (self.n + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        perm = torch.randperm(self.n, device=self.images.device)
        end = (self.n // self.batch_size) * self.batch_size if self.drop_last else self.n
        for start in range(0, end, self.batch_size):
            idx = perm[start:start + self.batch_size]
            imgs = self.images[idx]
            if self.augment:
                imgs = gpu_random_crop_flip(imgs)
            imgs = (imgs - self._mean) / self._std
            yield imgs, self.labels[idx], self.buckets[idx]


def _build_train_val_test_buckets(
    noise_prob: float,
    batch_size: int,
    device: torch.device,
    data_dir: str,
    data_augmentation: bool,
    noise_seed: int = DEFAULT_NOISE_SEED,
    val_size: int = DEFAULT_VAL_SIZE,
    val_split_seed: int = VAL_SPLIT_SEED,
):
    """Build train (with buckets), val, and test loaders.

    Bucket 0 = clean (label unchanged by noise), bucket 1 = mislabeled.
    val + test labels are noisy / clean respectively per the standard
    pipeline (val uses train-noisy labels for the val-loss signal to
    track interpolation; test is the unmodified CIFAR test set).
    """
    train_images, train_orig, test_images, test_labels = (
        _build_gpu_cifar_tensors(data_dir, device)
    )
    train_noisy = _apply_label_noise_tensor(train_orig, noise_prob, noise_seed)

    # Mislabel mask over all 50K: 1 where noisy != orig
    mis_mask_50k = (train_noisy != train_orig).long()  # [50000] on device

    train_indices, val_indices = compute_val_split_indices(
        total_samples=train_images.size(0),
        val_size=val_size,
        seed=val_split_seed,
    )
    train_idx_t = torch.from_numpy(train_indices).to(device)
    val_idx_t = torch.from_numpy(val_indices).to(device)

    train_sub_imgs = train_images.index_select(0, train_idx_t).contiguous()
    train_sub_labels = train_noisy.index_select(0, train_idx_t).contiguous()
    train_sub_buckets = mis_mask_50k.index_select(0, train_idx_t).contiguous()
    val_sub_imgs = train_images.index_select(0, val_idx_t).contiguous()
    val_sub_labels = train_noisy.index_select(0, val_idx_t).contiguous()

    train_loader = _GPUBucketTrainLoader(
        train_sub_imgs, train_sub_labels, train_sub_buckets, batch_size,
        augment=data_augmentation, drop_last=True,
    )
    # Re-use the existing GPUEvalLoader pattern by importing it directly.
    from jl.double_descent.resnet18.resnet18_data import GPUEvalLoader
    val_loader = GPUEvalLoader(val_sub_imgs, val_sub_labels, batch_size)
    test_loader = GPUEvalLoader(test_images, test_labels, batch_size)
    return train_loader, val_loader, test_loader, train_sub_buckets


# ---------------------------------------------------------------------------
# Shadow allocation + diagnostics
# ---------------------------------------------------------------------------


def _init_shadows(model: nn.Module, n_bins: int, device: torch.device):
    return [
        [torch.zeros_like(p, device=device) for p in model.parameters()]
        for _ in range(n_bins)
    ]


def _shadow_global_l2(shadow_list) -> float:
    s = 0.0
    for t in shadow_list:
        s += float(t.pow(2).sum().item())
    return math.sqrt(s)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


def train_single_model_bucket_shadow(
    gpu_id: int,
    model_factory: Callable[[], nn.Module],
    model_label: str,
    model_params: Dict[str, Any],
    config,
    output_path: str,
    data_path: str,
) -> None:
    """Train a single ResNet (or any classifier from model_factory) while
    tracking per-bucket AdamW shadows.

    Bucket definition: bucket 1 = mislabeled (true noisy labels differ from
    original), bucket 0 = clean. Mask is observable by construction.

    Outputs in output_path:
      - metrics_{label}.jsonl
      - model_{label}.pt
      - bucket_shadows_{label}.pt
      - bucket_shares_{label}.json
      - early_stop/model_{label}.pt (if val_loader available)
    """
    device = torch.device(f"cuda:{gpu_id}")
    print(f"[GPU {gpu_id}] [bucket-shadow] Training {model_label} on {device}")

    train_loader, val_loader, test_loader, train_buckets = (
        _build_train_val_test_buckets(
            noise_prob=config.label_noise,
            batch_size=config.batch_size,
            device=device,
            data_dir=data_path,
            data_augmentation=config.data_augmentation,
        )
    )
    n_train = int(train_buckets.size(0))
    n_mis = int(train_buckets.sum().item())
    n_clean = n_train - n_mis
    print(
        f"[GPU {gpu_id}] {model_label} bucket sizes: "
        f"clean={n_clean}, mislabeled={n_mis} "
        f"(mis rate {n_mis / n_train:.3f})"
    )

    model = model_factory().to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"[GPU {gpu_id}] {model_label} has {num_params:,} parameters")

    # Manual AdamW with per-bucket m, shared v. Match torch.optim.AdamW
    # defaults for the resnet path: betas=(0.9, 0.999), eps=1e-8, wd=0.01.
    BETA1, BETA2 = 0.9, 0.999
    ADAM_EPS = 1e-8
    WEIGHT_DECAY = getattr(config, "weight_decay", 0.01)

    n_bins = 2
    params = list(model.parameters())
    adam_m_per_bucket = [
        [torch.zeros_like(p, device=device) for p in params] for _ in range(n_bins)
    ]
    adam_v = [torch.zeros_like(p, device=device) for p in params]
    shadows = _init_shadows(model, n_bins, device)
    print(
        f"[GPU {gpu_id}] {model_label} allocated {n_bins} per-bucket m + "
        f"{n_bins} shadow tensors + shared v ({2 * n_bins + 1}x model size)"
    )

    base_lr = config.learning_rate
    if config.cosine_decay_epoch is not None:
        cosine_start = config.cosine_decay_epoch
        cosine_total = config.epochs - cosine_start
    else:
        cosine_start = None
        cosine_total = 0

    def lr_at_epoch(epoch: int) -> float:
        if cosine_start is None or epoch < cosine_start:
            return base_lr
        progress = (epoch - cosine_start) / max(1, cosine_total)
        return base_lr * 0.5 * (1 + math.cos(math.pi * progress))

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
    if use_bf16:
        print(f"[GPU {gpu_id}] {model_label} BF16 autocast enabled")

    step = 0
    model.train()
    for epoch in range(config.epochs):
        epoch_start = time.time()
        current_lr = lr_at_epoch(epoch)

        train_loss_sum = 0.0
        train_correct = 0
        train_samples = 0

        for images, labels, buckets in train_loader:
            B = images.size(0)
            # Autocast for the forward only; gradients flow through fp32 params.
            with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
                logits = model(images)
                per_sample_loss = F.cross_entropy(
                    logits, labels, reduction="none",
                )  # [B], dtype depends on autocast
            per_sample_loss = per_sample_loss.float()

            n_active = float(B)
            total_grad = [None] * len(params)

            # Per-bucket backward. Mirror the transformer shadow pattern.
            for b in range(n_bins):
                mask_b = (buckets == b)
                if not torch.any(mask_b):
                    # m_b still decays so the EMA progresses uniformly.
                    with torch.no_grad():
                        for p_idx in range(len(params)):
                            adam_m_per_bucket[b][p_idx].mul_(BETA1)
                    continue
                L_b = (per_sample_loss * mask_b.float()).sum() / n_active
                grads_b = torch.autograd.grad(
                    L_b, params,
                    retain_graph=(b < n_bins - 1),
                    allow_unused=True,
                )
                with torch.no_grad():
                    for p_idx, g in enumerate(grads_b):
                        if g is None:
                            adam_m_per_bucket[b][p_idx].mul_(BETA1)
                            continue
                        adam_m_per_bucket[b][p_idx].mul_(BETA1).add_(
                            g, alpha=(1.0 - BETA1)
                        )
                        if total_grad[p_idx] is None:
                            total_grad[p_idx] = g.detach().clone()
                        else:
                            total_grad[p_idx].add_(g)
                del grads_b

            step += 1
            bias_correction1 = 1.0 - BETA1 ** step
            bias_correction2 = 1.0 - BETA2 ** step

            with torch.no_grad():
                for p_idx, p in enumerate(params):
                    # Decoupled weight decay (NOT attributed to any bucket).
                    p.mul_(1.0 - current_lr * WEIGHT_DECAY)

                    g_total = total_grad[p_idx]
                    if g_total is None:
                        adam_v[p_idx].mul_(BETA2)
                        continue
                    adam_v[p_idx].mul_(BETA2).addcmul_(
                        g_total, g_total, value=(1.0 - BETA2)
                    )
                    denom = (adam_v[p_idx] / bias_correction2).sqrt_().add_(ADAM_EPS)
                    total_update = torch.zeros_like(p)
                    for b in range(n_bins):
                        m_hat_b = adam_m_per_bucket[b][p_idx] / bias_correction1
                        update_b = -current_lr * m_hat_b / denom
                        shadows[b][p_idx].add_(update_b)
                        total_update.add_(update_b)
                    p.add_(total_update)
                    del denom, total_update
            del total_grad

            with torch.no_grad():
                train_loss_sum += per_sample_loss.sum().item()
                preds = logits.argmax(dim=1)
                train_correct += int((preds == labels).sum().item())
                train_samples += B

        train_loss = train_loss_sum / max(1, train_samples)
        train_error = 1.0 - (train_correct / max(1, train_samples))

        test_error, test_loss = evaluate(model, test_loader, device)
        val_error, val_loss = (None, None)
        if val_loader is not None:
            val_error, val_loss = evaluate(model, val_loader, device)

        shadow_norms = [_shadow_global_l2(shadows[b]) for b in range(n_bins)]
        tot_norm = sum(shadow_norms) or 1.0
        shadow_shares = [v / tot_norm for v in shadow_norms]

        epoch_time = time.time() - epoch_start
        metrics = {
            "epoch": epoch + 1,
            **model_params,
            "lr": current_lr,
            "train_error": train_error,
            "test_error": test_error,
            "train_loss": train_loss,
            "test_loss": test_loss,
            "shadow_norms": shadow_norms,
            "shadow_shares": shadow_shares,
        }
        if val_loader is not None:
            metrics["val_error"] = val_error
            metrics["val_loss"] = val_loss
        with open(metrics_path, "a") as f:
            f.write(json.dumps(metrics) + "\n")

        if val_loss is not None and val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_epoch = epoch + 1
            torch.save(model.state_dict(), es_model_path)

        if (epoch + 1) % config.log_interval == 0 or epoch == 0:
            val_str = (
                f" val_err={val_error:.4f} val_loss={val_loss:.4f} |"
                if val_loss is not None else ""
            )
            print(
                f"[GPU {gpu_id}] {model_label} Epoch {epoch + 1:4d}/{config.epochs} | "
                f"train_err={train_error:.4f} test_err={test_error:.4f} |{val_str} "
                f"train_loss={train_loss:.4f} test_loss={test_loss:.4f} | "
                f"shares={[round(s, 3) for s in shadow_shares]} | "
                f"{epoch_time:.1f}s"
            )

    # --- save final model + shadows + diagnostics ---
    model_path = output_path_p / f"model_{model_label}.pt"
    torch.save(model.state_dict(), model_path)

    final_norms = [_shadow_global_l2(shadows[b]) for b in range(n_bins)]
    tot_final = sum(final_norms) or 1.0
    final_shares = [v / tot_final for v in final_norms]
    summary = {
        **model_params,
        "n_bins": n_bins,
        "n_train": n_train,
        "n_clean": n_clean,
        "n_mislabeled": n_mis,
        "mislabel_rate": n_mis / max(1, n_train),
        "shadow_norms": final_norms,
        "shadow_shares": final_shares,
        "best_val_loss": best_val_loss,
        "best_val_epoch": best_val_epoch,
    }
    shares_path = output_path_p / f"bucket_shares_{model_label}.json"
    shares_path.write_text(json.dumps(summary, indent=2))

    shadow_path = output_path_p / f"bucket_shadows_{model_label}.pt"
    torch.save(
        {
            "shadows": [[s.detach().cpu() for s in shadows[b]] for b in range(n_bins)],
            "param_names": [n for n, _ in model.named_parameters()],
            "n_clean": n_clean,
            "n_mislabeled": n_mis,
            "bucket_labels": ["clean", "mislabeled"],
        },
        shadow_path,
    )
    print(
        f"[GPU {gpu_id}] {model_label} saved {shadow_path.name} "
        f"({shadow_path.stat().st_size / 1e6:.1f} MB)"
    )

    # Evaluate final model + early-stop checkpoint via shared helper.
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

    print(
        f"[GPU {gpu_id}] {model_label} DONE. "
        f"shares={[round(s, 3) for s in final_shares]}"
    )
