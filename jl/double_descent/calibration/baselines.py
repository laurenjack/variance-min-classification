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
    max_iter: int = 200,
) -> float:
    """Fit scalar temperature T on validation logits via L-BFGS.

    NOTE: We use lr=1.0 (PyTorch default), not the lr=0.01 from Guo et al.'s
    reference implementation. Their lr=0.01 fails to converge from T=1 when
    the optimum is far away (e.g. T~2.5 for overconfident CIFAR-100 ResNet-110),
    returning T~1.2 in 50 iterations. lr=1.0 reliably finds the global optimum.

    Returns:
        Temperature T (float).
    """
    temperature = nn.Parameter(torch.ones(1))
    optimizer = torch.optim.LBFGS([temperature], lr=1.0, max_iter=max_iter)

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


# --- Histogram Binning (top-label, Guo et al. 2017) ---


def fit_histogram_binning(
    val_logits: torch.Tensor,
    val_labels: torch.Tensor,
    num_bins: int = 15,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fit top-label histogram binning on validation set (Guo et al. 2017).

    Bin the predicted-class confidence (max softmax probability) into
    equal-width bins, and for each bin compute the empirical accuracy
    (fraction of correct predictions). At test time, replace the
    confidence with the bin's accuracy.

    Returns:
        (bin_boundaries [num_bins+1], bin_accuracies [num_bins])
    """
    probs = F.softmax(val_logits, dim=-1)
    max_probs, predictions = probs.max(dim=1)
    correct = (predictions == val_labels).float()

    bin_boundaries = torch.linspace(0, 1, num_bins + 1)
    bin_accuracies = torch.zeros(num_bins)

    for i in range(num_bins):
        lo, hi = bin_boundaries[i].item(), bin_boundaries[i + 1].item()
        mask = (max_probs > lo) & (max_probs <= hi)
        if i == 0:
            mask = mask | (max_probs == lo)
        n_bin = mask.sum().item()
        if n_bin > 0:
            bin_accuracies[i] = correct[mask].mean()
        else:
            # Empty bin: use midpoint as fallback
            bin_accuracies[i] = (lo + hi) / 2

    logger.info(f"Histogram binning: {num_bins} bins (top-label)")
    return bin_boundaries, bin_accuracies


def apply_histogram_binning(
    logits: torch.Tensor,
    bin_boundaries: torch.Tensor,
    bin_accuracies: torch.Tensor,
) -> torch.Tensor:
    """Apply top-label histogram binning, return calibrated probabilities.

    Replaces the top-class confidence with the bin's empirical accuracy
    and redistributes the remaining probability mass proportionally
    among the other classes.
    """
    probs = F.softmax(logits, dim=-1)
    max_probs, top_classes = probs.max(dim=1)
    num_bins = len(bin_accuracies)

    bin_indices = torch.bucketize(max_probs, bin_boundaries[1:-1])
    bin_indices = bin_indices.clamp(0, num_bins - 1)
    calibrated_top = bin_accuracies[bin_indices]  # [N]

    # Vectorized redistribution
    n_classes = probs.shape[1]
    n = probs.shape[0]
    arange_n = torch.arange(n)

    old_top = probs[arange_n, top_classes]
    new_top = calibrated_top

    # Scale non-top classes by (1 - new_top) / (1 - old_top)
    # When old_top is ~1, fall back to uniform redistribution
    safe = old_top < 1.0 - 1e-8
    scale = torch.where(
        safe,
        (1.0 - new_top) / (1.0 - old_top).clamp(min=1e-8),
        torch.zeros_like(new_top),
    )

    calibrated = probs * scale.unsqueeze(1)
    # For non-safe rows, distribute (1 - new_top) uniformly across non-top
    uniform_share = (1.0 - new_top) / (n_classes - 1)
    calibrated[~safe] = uniform_share[~safe].unsqueeze(1).expand(-1, n_classes)

    # Set top-class probability
    calibrated[arange_n, top_classes] = new_top

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
