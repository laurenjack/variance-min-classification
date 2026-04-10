"""Post-hoc calibration baselines.

Implements standard calibration methods that fit on validation logits:
1. Temperature scaling (single scalar T)
2. Vector scaling (multi-class Platt scaling)
3. Histogram binning
4. Dirichlet calibration L2 (ODIR regularization, Kull et al. 2019)

All methods fit on (val_logits, val_labels) and return parameters that
can be applied to transform logits.
"""

import logging
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# --- Temperature Scaling ---


def fit_temperature(
    val_logits: torch.Tensor,
    val_labels: torch.Tensor,
    max_iter: int = 50,
) -> float:
    """Fit scalar temperature T on validation logits via L-BFGS.

    Returns:
        Temperature T (float).
    """
    temperature = nn.Parameter(torch.ones(1))
    optimizer = torch.optim.LBFGS([temperature], lr=0.01, max_iter=max_iter)

    def closure():
        optimizer.zero_grad()
        loss = F.cross_entropy(val_logits / temperature, val_labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    T = temperature.item()
    logger.info(f"Temperature scaling: T={T:.4f}")
    return T


def apply_temperature(logits: torch.Tensor, T: float) -> torch.Tensor:
    """Apply temperature scaling to logits."""
    return logits / T


# --- Vector Scaling (multi-class Platt scaling) ---


def fit_vector_scaling(
    val_logits: torch.Tensor,
    val_labels: torch.Tensor,
    max_iter: int = 50,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fit per-class weight and bias on validation logits.

    Learns w, b such that calibrated_logits_c = w_c * logit_c + b_c.
    Initialized at w=1, b=0 (identity).

    Returns:
        (weights [C], biases [C])
    """
    C = val_logits.shape[1]
    weights = nn.Parameter(torch.ones(C))
    biases = nn.Parameter(torch.zeros(C))

    optimizer = torch.optim.LBFGS([weights, biases], lr=0.01, max_iter=max_iter)

    def closure():
        optimizer.zero_grad()
        scaled = val_logits * weights.unsqueeze(0) + biases.unsqueeze(0)
        loss = F.cross_entropy(scaled, val_labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    logger.info(
        f"Vector scaling: w=[{weights.min().item():.4f}, {weights.max().item():.4f}], "
        f"b=[{biases.min().item():.4f}, {biases.max().item():.4f}]"
    )
    return weights.detach(), biases.detach()


def apply_vector_scaling(
    logits: torch.Tensor, weights: torch.Tensor, biases: torch.Tensor
) -> torch.Tensor:
    """Apply vector scaling to logits, return calibrated logits."""
    return logits * weights.unsqueeze(0) + biases.unsqueeze(0)


# --- Histogram Binning (per-class, Zadrozny & Elkan / Guo et al.) ---


def fit_histogram_binning(
    val_logits: torch.Tensor,
    val_labels: torch.Tensor,
    num_bins: int = 15,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fit per-class histogram binning on validation set.

    Standard multiclass histogram binning (Zadrozny & Elkan 2002):
    For each class k, treat as a binary problem (y == k vs y != k).
    Bin the marginal probability p_k and compute the empirical positive
    rate per bin. At test time, replace p_k with the bin's positive rate
    and renormalize so probabilities sum to 1.

    Returns:
        (bin_boundaries [num_bins+1], bin_rates [num_classes, num_bins])
    """
    probs = F.softmax(val_logits, dim=-1)
    num_classes = probs.shape[1]

    bin_boundaries = torch.linspace(0, 1, num_bins + 1)
    bin_rates = torch.zeros(num_classes, num_bins)

    for k in range(num_classes):
        p_k = probs[:, k]
        is_k = (val_labels == k).float()

        for i in range(num_bins):
            lo, hi = bin_boundaries[i].item(), bin_boundaries[i + 1].item()
            mask = (p_k > lo) & (p_k <= hi)
            if i == 0:
                mask = mask | (p_k == lo)
            n_bin = mask.sum().item()
            if n_bin > 0:
                bin_rates[k, i] = is_k[mask].mean()
            else:
                # Empty bin: use midpoint as fallback
                bin_rates[k, i] = (lo + hi) / 2

    logger.info(f"Histogram binning: {num_bins} bins per class, {num_classes} classes")
    return bin_boundaries, bin_rates


def apply_histogram_binning(
    logits: torch.Tensor,
    bin_boundaries: torch.Tensor,
    bin_rates: torch.Tensor,
) -> torch.Tensor:
    """Apply per-class histogram binning to logits, return calibrated probabilities.

    For each class k, look up the bin index for p_k and replace with the
    fitted positive rate. Renormalize so probabilities sum to 1.
    """
    probs = F.softmax(logits, dim=-1)
    num_classes = probs.shape[1]
    num_bins = bin_rates.shape[1]

    # Vectorized bin lookup per class
    # bin_indices[n, k] = bin for sample n, class k
    bin_indices = torch.bucketize(probs, bin_boundaries[1:-1])  # [N, K]
    bin_indices = bin_indices.clamp(0, num_bins - 1)

    # Gather rates: calibrated[n, k] = bin_rates[k, bin_indices[n, k]]
    # bin_rates is [K, B], we need to index per (n, k) pair
    k_idx = torch.arange(num_classes).unsqueeze(0).expand_as(bin_indices)  # [N, K]
    calibrated = bin_rates[k_idx, bin_indices]  # [N, K]

    # Renormalize so each row sums to 1
    row_sums = calibrated.sum(dim=1, keepdim=True).clamp(min=1e-8)
    calibrated = calibrated / row_sums

    return calibrated


# --- Dirichlet Calibration L2 (ODIR) ---


def fit_dirichlet_l2(
    val_logits: torch.Tensor,
    val_labels: torch.Tensor,
    l2_off_diag: float = 1e-3,
    l2_bias: float = 1e-3,
    max_iter: int = 50,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fit Dirichlet calibration with ODIR regularization (Kull et al. 2019).

    Maps log-softmax outputs through a linear transform:
        calibrated = softmax(W @ log_softmax(z) + b)

    ODIR regularization penalizes off-diagonal elements of W and the bias b,
    leaving diagonal elements (per-class temperatures) unregularized.

    Initialized at W=I, b=0 (reduces to identity at start).

    Returns:
        (W [C, C], b [C])
    """
    C = val_logits.shape[1]
    log_probs = F.log_softmax(val_logits, dim=-1)

    W = nn.Parameter(torch.eye(C))
    b = nn.Parameter(torch.zeros(C))

    optimizer = torch.optim.LBFGS([W, b], lr=0.01, max_iter=max_iter)

    def closure():
        optimizer.zero_grad()
        calibrated_logits = log_probs @ W.t() + b.unsqueeze(0)
        nll = F.cross_entropy(calibrated_logits, val_labels)

        # ODIR: regularize off-diagonal elements of W and bias
        off_diag_mask = ~torch.eye(C, dtype=torch.bool, device=W.device)
        reg = l2_off_diag * W[off_diag_mask].pow(2).sum() + l2_bias * b.pow(2).sum()

        total = nll + reg
        total.backward()
        return total

    optimizer.step(closure)

    diag = W.detach().diag()
    off_diag = W.detach()[~torch.eye(C, dtype=torch.bool)].abs().mean()
    logger.info(
        f"Dirichlet L2: diag=[{diag.min().item():.4f}, {diag.max().item():.4f}], "
        f"off_diag_mean={off_diag.item():.4f}, "
        f"b=[{b.min().item():.4f}, {b.max().item():.4f}]"
    )
    return W.detach(), b.detach()


def apply_dirichlet_l2(
    logits: torch.Tensor, W: torch.Tensor, b: torch.Tensor
) -> torch.Tensor:
    """Apply Dirichlet calibration, return calibrated logits."""
    log_probs = F.log_softmax(logits, dim=-1)
    return log_probs @ W.t() + b.unsqueeze(0)
