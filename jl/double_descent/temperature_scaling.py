"""Temperature scaling utilities.

Fits a scalar temperature T on held-out validation logits via L-BFGS,
then evaluates logits/T on a test set. Used by both ResNet18 and
Transformer evaluation pipelines.
"""

import logging
from typing import Callable, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def fit_temperature(
    logits: torch.Tensor,
    labels: torch.Tensor,
    chunk_size: int = 0,
) -> float:
    """Fit scalar temperature T via L-BFGS on validation logits.

    Uses lr=1.0, max_iter=200. The lr=0.01 default fails when the
    optimum is far from T=1 (e.g. T~2.5 for overconfident models).

    Args:
        logits: (N, C) float tensor of pre-softmax logits.
        labels: (N,) long tensor of class labels.
        chunk_size: If > 0, compute cross-entropy in chunks to avoid
            OOM on large vocab (e.g. transformer with V~10K).

    Returns:
        Fitted temperature as a float.
    """
    temperature = nn.Parameter(torch.ones(1, device=logits.device))
    optimizer = torch.optim.LBFGS([temperature], lr=1.0, max_iter=200)
    n = logits.size(0)

    if chunk_size > 0:
        def closure():
            optimizer.zero_grad()
            total_loss = torch.zeros((), device=logits.device)
            for i in range(0, n, chunk_size):
                chunk_logits = logits[i:i + chunk_size]
                chunk_labels = labels[i:i + chunk_size]
                loss = (
                    F.cross_entropy(chunk_logits / temperature, chunk_labels,
                                    reduction="sum") / n
                )
                loss.backward()
                total_loss = total_loss + loss.detach()
            return total_loss
    else:
        def closure():
            optimizer.zero_grad()
            loss = F.cross_entropy(logits / temperature, labels)
            loss.backward()
            return loss

    optimizer.step(closure)
    t = temperature.item()
    logger.info(f"Fitted temperature: {t:.6f}")
    return t


def metrics_with_temperature(
    logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
    compute_ece_fn: Optional[Callable] = None,
) -> Dict[str, float]:
    """Compute loss, error, and optionally ECE on temperature-scaled logits.

    Args:
        logits: (N, C) float tensor of pre-softmax logits.
        labels: (N,) long tensor of class labels.
        temperature: Scalar temperature to divide logits by.
        compute_ece_fn: Optional function(confidences, correct) -> float.

    Returns:
        Dict with ts_loss, ts_error, and optionally ts_ece.
    """
    with torch.no_grad():
        scaled_logits = logits / temperature
        loss = F.cross_entropy(scaled_logits, labels).item()
        probs = F.softmax(scaled_logits, dim=-1)
        max_probs, preds = probs.max(dim=-1)
        correct = preds == labels
        error = 1.0 - correct.float().mean().item()

        result = {"ts_loss": loss, "ts_error": error}
        if compute_ece_fn is not None:
            result["ts_ece"] = compute_ece_fn(max_probs, correct)

    return result
