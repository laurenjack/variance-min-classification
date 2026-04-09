"""Shared evaluation metrics for calibration experiments."""

from typing import Dict

import torch
import torch.nn.functional as F


def compute_ece(
    confidences: torch.Tensor, correct: torch.Tensor, num_bins: int = 20
) -> float:
    """Expected Calibration Error with equal-width bins."""
    bin_boundaries = torch.linspace(0, 1, num_bins + 1)
    n = len(confidences)
    ece = 0.0
    for i in range(num_bins):
        lo, hi = bin_boundaries[i].item(), bin_boundaries[i + 1].item()
        mask = (confidences > lo) & (confidences <= hi)
        if i == 0:
            mask = mask | (confidences == lo)
        n_bin = mask.sum().item()
        if n_bin > 0:
            avg_confidence = confidences[mask].mean().item()
            avg_accuracy = correct[mask].float().mean().item()
            ece += (n_bin / n) * abs(avg_accuracy - avg_confidence)
    return ece


def compute_brier_score(probs: torch.Tensor, labels: torch.Tensor) -> float:
    """Multi-class Brier score."""
    one_hot = F.one_hot(labels, num_classes=probs.size(1)).float()
    return ((probs - one_hot) ** 2).sum(dim=1).mean().item()


def evaluate_logits(
    logits: torch.Tensor, labels: torch.Tensor
) -> Dict[str, float]:
    """Compute all metrics from logits and labels.

    Returns:
        Dict with nll, accuracy, ece, brier, auroc, aupr.
    """
    from sklearn.metrics import roc_auc_score, average_precision_score

    probs = F.softmax(logits, dim=-1)
    max_probs, predictions = probs.max(dim=1)
    correct = (predictions == labels)

    nll = F.cross_entropy(logits, labels).item()
    accuracy = correct.float().mean().item()
    ece = compute_ece(max_probs, correct)
    brier = compute_brier_score(probs, labels)

    probs_np = probs.numpy()
    labels_np = labels.numpy()

    try:
        auroc = roc_auc_score(labels_np, probs_np, multi_class="ovr", average="macro")
    except ValueError:
        auroc = 0.0

    try:
        labels_one_hot = F.one_hot(labels, num_classes=probs.size(1)).numpy()
        aupr = average_precision_score(labels_one_hot, probs_np, average="macro")
    except ValueError:
        aupr = 0.0

    return {
        "nll": round(nll, 6),
        "accuracy": round(accuracy, 6),
        "ece": round(ece, 6),
        "brier": round(brier, 6),
        "auroc": round(auroc, 6),
        "aupr": round(aupr, 6),
    }


def evaluate_probs(
    probs: torch.Tensor, labels: torch.Tensor
) -> Dict[str, float]:
    """Compute all metrics from probabilities and labels.

    Like evaluate_logits but takes pre-computed probabilities (e.g. from
    histogram binning). NLL computed via -log(prob) of true class.
    """
    from sklearn.metrics import roc_auc_score, average_precision_score

    probs = probs.clamp(min=1e-8)
    max_probs, predictions = probs.max(dim=1)
    correct = (predictions == labels)

    nll = -torch.log(probs[torch.arange(len(labels)), labels]).mean().item()
    accuracy = correct.float().mean().item()
    ece = compute_ece(max_probs, correct)
    brier = compute_brier_score(probs, labels)

    probs_np = probs.numpy()
    labels_np = labels.numpy()

    try:
        auroc = roc_auc_score(labels_np, probs_np, multi_class="ovr", average="macro")
    except ValueError:
        auroc = 0.0

    try:
        labels_one_hot = F.one_hot(labels, num_classes=probs.size(1)).numpy()
        aupr = average_precision_score(labels_one_hot, probs_np, average="macro")
    except ValueError:
        aupr = 0.0

    return {
        "nll": round(nll, 6),
        "accuracy": round(accuracy, 6),
        "ece": round(ece, 6),
        "brier": round(brier, 6),
        "auroc": round(auroc, 6),
        "aupr": round(aupr, 6),
    }
