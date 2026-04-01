"""Shared L-BFGS + L2 optimization for final-layer fine-tuning.

Fine-tunes a single nn.Linear layer using L-BFGS with L2 regularization
to reach a stationary point of loss + lambda * ||W||^2. This enables
training-point decomposition per Yeh & Kim et al. (2018).
"""

import logging
from typing import Callable, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def fine_tune_final_layer(
    features: torch.Tensor,
    targets: torch.Tensor,
    linear_layer: nn.Linear,
    l2_lambda: float = 1e-5,
    max_steps: int = 100,
    chunk_size: int = 20000,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """Fine-tune a linear layer with L-BFGS and L2 regularization.

    Args:
        features: [N, d_in] pre-extracted features from frozen backbone.
        targets: [N] class/token indices.
        linear_layer: The layer to optimize, initialized with trained weights.
        l2_lambda: L2 regularization strength.
        max_steps: Number of L-BFGS steps.
        chunk_size: Chunk size for logit computation to avoid OOM.
        device: Device to run on. If None, uses features' device.

    Returns:
        Metadata dict with final_loss, final_grad_norm, steps, l2_lambda.
    """
    if device is not None:
        features = features.to(device)
        targets = targets.to(device)
        linear_layer = linear_layer.to(device)

    linear_layer.train()
    for p in linear_layer.parameters():
        p.requires_grad_(True)

    N = features.shape[0]

    optimizer = torch.optim.LBFGS(
        linear_layer.parameters(),
        lr=1.0,
        max_iter=20,
        history_size=100,
        line_search_fn="strong_wolfe",
    )

    last_loss = None

    def closure():
        nonlocal last_loss
        optimizer.zero_grad()

        # Compute cross-entropy loss in chunks to avoid OOM
        total_loss = torch.zeros(1, device=features.device, dtype=features.dtype)
        for i in range(0, N, chunk_size):
            chunk_f = features[i : i + chunk_size]
            chunk_t = targets[i : i + chunk_size]
            logits = linear_layer(chunk_f)
            chunk_loss = F.cross_entropy(logits, chunk_t, reduction="sum")
            total_loss = total_loss + chunk_loss
        avg_loss = total_loss / N

        # L2 regularization on all parameters (weights + bias)
        l2 = l2_lambda * sum(p.pow(2).sum() for p in linear_layer.parameters())
        total = avg_loss + l2
        total.backward()

        last_loss = total.item()
        return total

    for step in range(max_steps):
        optimizer.step(closure)
        if step % 10 == 0 or step == max_steps - 1:
            grad_norm = max(
                p.grad.norm().item()
                for p in linear_layer.parameters()
                if p.grad is not None
            )
            logger.info(
                f"  Step {step + 1}/{max_steps}: loss={last_loss:.6f}, "
                f"grad_norm={grad_norm:.2e}"
            )

    # Final metrics
    final_grad_norm = max(
        p.grad.norm().item()
        for p in linear_layer.parameters()
        if p.grad is not None
    )

    return {
        "final_loss": last_loss,
        "final_grad_norm": final_grad_norm,
        "steps": max_steps,
        "l2_lambda": l2_lambda,
    }
