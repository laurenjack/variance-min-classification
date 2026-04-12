"""Influence score computation via Yeh & Kim et al. (2018) decomposition.

Computes per-training-point influence scores measuring how much each
training point contributes to the model's test-time predictions.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def compute_residuals(
    features: torch.Tensor,
    labels: torch.Tensor,
    linear: nn.Linear,
) -> torch.Tensor:
    """Compute softmax residuals: softmax(logits) - one_hot(labels).

    Args:
        features: (N, d) penultimate features.
        labels: (N,) integer class labels.
        linear: Fine-tuned linear layer.

    Returns:
        (N, C) residual tensor.
    """
    with torch.no_grad():
        logits = features @ linear.weight.t() + linear.bias
        probs = F.softmax(logits, dim=1)
        one_hot = F.one_hot(labels, num_classes=linear.out_features).float()
        return probs - one_hot


def compute_influence_scores(
    phi_train: torch.Tensor,
    phi_test: torch.Tensor,
    residuals: torch.Tensor,
    chunk_size: int = 10000,
) -> torch.Tensor:
    """Compute normalized per-training-point influence scores.

    influence_i = ||r_i||_2 * mean_t(phi_train[i] . phi_test[t])
    Normalized so mean(influence) = 1.

    Args:
        phi_train: (N_train, d) training features.
        phi_test: (N_test, d) test features.
        residuals: (N_train, C) softmax residuals from fine-tuned model.
        chunk_size: Process test points in chunks of this size.

    Returns:
        (N_train,) normalized influence scores.
    """
    grad_norms = residuals.norm(dim=1)  # (N_train,)

    # Compute mean feature similarity to test set
    n_test = phi_test.size(0)
    mean_sim = torch.zeros(phi_train.size(0), device=phi_train.device)

    for start in range(0, n_test, chunk_size):
        end = min(start + chunk_size, n_test)
        # (N_train, chunk) -> sum along chunk dim
        dots = phi_train @ phi_test[start:end].t()
        mean_sim += dots.sum(dim=1)

    mean_sim /= n_test

    raw = grad_norms * mean_sim
    influence = raw / raw.mean()
    return influence


def build_mislabel_mask(
    noisy_labels: list,
    original_labels: list,
    train_indices: np.ndarray,
) -> np.ndarray:
    """Identify which training points have corrupted labels.

    Args:
        noisy_labels: Full 50K noisy label list from NoisyCIFAR10.labels.
        original_labels: Full 50K original labels from NoisyCIFAR10.cifar.targets.
        train_indices: (N_train,) indices into the 50K set.

    Returns:
        (N_train,) boolean array, True = mislabeled.
    """
    mask = np.array([
        noisy_labels[int(i)] != original_labels[int(i)]
        for i in train_indices
    ])
    return mask


def save_influence_jsonl(
    influence: torch.Tensor,
    mislabel_mask: np.ndarray,
    train_indices: np.ndarray,
    noisy_labels: list,
    original_labels: list,
    output_dir: Path,
    k: int,
) -> Path:
    """Write per-training-point influence data to JSONL.

    Each line: {index, influence, mislabeled, original_label, noisy_label}
    """
    path = output_dir / f"influence_k{k}.jsonl"
    influence_cpu = influence.cpu().numpy()

    with open(path, "w") as f:
        for i in range(len(influence_cpu)):
            idx = int(train_indices[i])
            record = {
                "index": idx,
                "influence": float(influence_cpu[i]),
                "mislabeled": bool(mislabel_mask[i]),
                "original_label": int(original_labels[idx]),
                "noisy_label": int(noisy_labels[idx]),
            }
            f.write(json.dumps(record) + "\n")

    logger.info(f"Saved {len(influence_cpu)} influence scores to {path}")
    return path


def compute_summary_stats(
    influence: torch.Tensor,
    mislabel_mask: np.ndarray,
) -> Dict[str, float]:
    """Aggregate influence statistics for one k value.

    Returns dict with:
        mean_influence_mislabeled: mean influence over mislabeled points
        mean_influence_clean: mean influence over clean points
        influence_ratio: mislabeled / clean ratio
        frac_mislabeled: fraction of training set that is mislabeled
        top1pct_frac_mislabeled: fraction of top 1% influential points that are mislabeled
        top5pct_frac_mislabeled: fraction of top 5% influential points that are mislabeled
    """
    inf_np = influence.cpu().numpy()
    mask = mislabel_mask.astype(bool)
    n = len(inf_np)

    mean_mis = float(inf_np[mask].mean()) if mask.any() else 0.0
    mean_clean = float(inf_np[~mask].mean()) if (~mask).any() else 0.0

    # Top-k analysis
    sorted_idx = np.argsort(inf_np)[::-1]
    top1pct = sorted_idx[:max(1, n // 100)]
    top5pct = sorted_idx[:max(1, n // 20)]

    return {
        "mean_influence_mislabeled": mean_mis,
        "mean_influence_clean": mean_clean,
        "influence_ratio": mean_mis / mean_clean if mean_clean > 0 else float("inf"),
        "frac_mislabeled": float(mask.sum()) / n,
        "top1pct_frac_mislabeled": float(mask[top1pct].sum()) / len(top1pct),
        "top5pct_frac_mislabeled": float(mask[top5pct].sum()) / len(top5pct),
    }
