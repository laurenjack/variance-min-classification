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
    return_diagnostics: bool = False,
    grad_tol: float = 1e-4,
):
    """Fit scalar temperature T via L-BFGS on validation logits.

    Uses lr=1.0, max_iter=200. The lr=0.01 default fails when the
    optimum is far from T=1 (e.g. T~2.5 for overconfident models).

    Args:
        logits: (N, C) float tensor of pre-softmax logits.
        labels: (N,) long tensor of class labels.
        chunk_size: If > 0, compute cross-entropy in chunks to avoid
            OOM on large vocab (e.g. transformer with V~10K).
        return_diagnostics: If True, return (T, diag) where diag is a dict
            of convergence info.
        grad_tol: |dCE/dT| below this counts as converged.

    Returns:
        Fitted temperature as a float, or (T, diag_dict) if
        return_diagnostics=True.
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

    # Snapshot pre-fit loss (T=1) for the convergence diagnostic.
    with torch.no_grad():
        if chunk_size > 0:
            initial_loss = 0.0
            for i in range(0, n, chunk_size):
                initial_loss += float(
                    F.cross_entropy(
                        logits[i:i + chunk_size], labels[i:i + chunk_size],
                        reduction="sum",
                    ).item()
                ) / n
        else:
            initial_loss = float(F.cross_entropy(logits, labels).item())

    optimizer.step(closure)
    t = temperature.item()

    # Re-evaluate at the fitted T to capture final loss + grad.
    temperature.grad = None
    final_loss_t = closure()
    final_loss = float(final_loss_t.item())
    final_grad = float(temperature.grad.item()) if temperature.grad is not None else float("nan")
    loss_delta = initial_loss - final_loss
    converged = abs(final_grad) < grad_tol

    logger.info(
        f"Fitted T={t:.6f} | initial_loss={initial_loss:.6f} -> "
        f"final_loss={final_loss:.6f} (Δ={loss_delta:+.6f}) | "
        f"|dCE/dT|={abs(final_grad):.2e} | "
        f"{'CONVERGED' if converged else 'NOT CONVERGED'}"
    )

    if return_diagnostics:
        return t, {
            "T": t,
            "initial_loss": initial_loss,
            "final_loss": final_loss,
            "loss_delta": loss_delta,
            "final_grad": final_grad,
            "converged": converged,
        }
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
