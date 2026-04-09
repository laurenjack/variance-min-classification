"""Shared optimization for L2 calibration of a final linear layer.

Optimizes a single nn.Linear layer using L-BFGS or SGD with L2 regularization,
reaching a stationary point of loss + lambda * ||W||^2.
"""

import logging
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def lambda_dir_name(l2_lambda: float, sgd: bool = False) -> str:
    """Format lambda value as a directory name.

    Examples:
        lambda_dir_name(1e-5) -> 'lambda_1e-05'
        lambda_dir_name(1e-5, sgd=True) -> 'sgd_lambda_1e-05'
    """
    name = f"lambda_{l2_lambda:.0e}"
    if sgd:
        name = f"sgd_{name}"
    return name


def l2_calibrate_final_layer(
    features: torch.Tensor,
    targets: torch.Tensor,
    linear_layer: nn.Linear,
    l2_lambda: float = 1e-5,
    max_steps: int = 100,
    chunk_size: int = 20000,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """L2 calibrate a linear layer with L-BFGS and L2 regularization.

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


def sgd_l2_calibrate_final_layer(
    features: torch.Tensor,
    targets: torch.Tensor,
    linear_layer: nn.Linear,
    l2_lambda: float = 1e-5,
    epochs: int = 100,
    batch_size: int = 2048,
    lr: float = 0.1,
    momentum: float = 0.9,
    warmup_epochs: int = 0,
    device: Optional[torch.device] = None,
    log_path: Optional[str] = None,
) -> Dict[str, float]:
    """L2 calibrate a linear layer with SGD and L2 regularization.

    Args:
        features: [N, d_in] pre-extracted features from frozen backbone.
        targets: [N] class/token indices.
        linear_layer: The layer to optimize, initialized with trained weights.
        l2_lambda: L2 regularization strength. Applied as weight_decay = 2 * l2_lambda
            to match L-BFGS which optimizes loss + l2_lambda * ||W||^2
            (gradient 2 * l2_lambda * W vs PyTorch's weight_decay * W).
        epochs: Number of passes over the full dataset.
        batch_size: Mini-batch size.
        lr: Peak learning rate.
        momentum: SGD momentum.
        warmup_epochs: Linear warmup epochs before cosine decay. 0 = no schedule.
        device: Device to run on. If None, uses features' device.

    Returns:
        Metadata dict with final_loss, final_grad_norm, epochs, l2_lambda, lr, momentum.
    """
    if device is not None:
        features = features.to(device)
        targets = targets.to(device)
        linear_layer = linear_layer.to(device)

    linear_layer.train()
    for p in linear_layer.parameters():
        p.requires_grad_(True)

    N = features.shape[0]

    # weight_decay in SGD adds weight_decay * param to gradient.
    # L-BFGS optimizes loss + l2_lambda * ||W||^2, gradient = 2 * l2_lambda * W.
    # So weight_decay = 2 * l2_lambda to match.
    optimizer = torch.optim.SGD(
        linear_layer.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=2 * l2_lambda,
    )

    # Learning rate schedule: linear warmup then cosine decay
    scheduler = None
    if warmup_epochs > 0:
        from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
        warmup = LinearLR(optimizer, start_factor=1e-3, total_iters=warmup_epochs)
        cosine = CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs)
        scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[warmup_epochs])

    last_loss = None
    history: List[Dict[str, float]] = []

    for epoch in range(epochs):
        # Shuffle indices each epoch
        perm = torch.randperm(N, device=features.device)
        epoch_loss = 0.0
        num_batches = 0

        for i in range(0, N, batch_size):
            idx = perm[i : i + batch_size]
            batch_f = features[idx]
            batch_t = targets[idx]

            optimizer.zero_grad()
            logits = linear_layer(batch_f)
            loss = F.cross_entropy(logits, batch_t)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        if scheduler is not None:
            scheduler.step()

        last_loss = epoch_loss / num_batches
        current_lr = optimizer.param_groups[0]["lr"]
        grad_norm = max(
            p.grad.norm().item()
            for p in linear_layer.parameters()
            if p.grad is not None
        )
        entry = {"epoch": epoch + 1, "loss": last_loss, "grad_norm": grad_norm, "lr": current_lr}
        history.append(entry)

        if log_path is not None:
            import json
            with open(log_path, "a") as f:
                f.write(json.dumps(entry) + "\n")

        if epoch % 10 == 0 or epoch == epochs - 1:
            logger.info(
                f"  Epoch {epoch + 1}/{epochs}: loss={last_loss:.6f}, "
                f"grad_norm={grad_norm:.2e}"
            )

    return {
        "final_loss": last_loss,
        "final_grad_norm": history[-1]["grad_norm"],
        "epochs": epochs,
        "l2_lambda": l2_lambda,
        "lr": lr,
        "momentum": momentum,
        "history": history,
    }


def compute_brier_score(probs: torch.Tensor, labels: torch.Tensor) -> float:
    """Multi-class Brier score.

    Args:
        probs: [N, C] softmax probabilities.
        labels: [N] class indices.

    Returns:
        Brier score as a float.
    """
    one_hot = F.one_hot(labels, num_classes=probs.size(1)).float()
    return ((probs - one_hot) ** 2).sum(dim=1).mean().item()
