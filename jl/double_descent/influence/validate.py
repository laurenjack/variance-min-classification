"""Validation of the decomposition identity and Figure X plotting.

Two purposes:
1. Verify that reconstructing W from the analytic formula matches the
   actual fine-tuned weights (KL divergence + logit MSE should be ~0).
2. Plot original vs fine-tuned loss across k values (Figure X).
"""

import json
import logging
from pathlib import Path
from typing import Dict, Sequence

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def validate_decomposition(
    features: torch.Tensor,
    labels: torch.Tensor,
    linear: nn.Linear,
    lambda_l2: float,
) -> Dict[str, float]:
    """Check that the analytic decomposition reproduces the fine-tuned W.

    At stationarity: W = -(1/2*lambda*n) * R^T @ Phi
    where R_ij = softmax(f(x_i))_j - 1{j == y_i}

    Args:
        features: (N, d) training features.
        labels: (N,) integer labels.
        linear: Fine-tuned linear layer.
        lambda_l2: L2 regularization strength used in fine-tuning.

    Returns:
        Dict with kl_div, logit_mse, max_W_diff, max_b_diff.
    """
    n = features.size(0)
    num_classes = linear.out_features

    with torch.no_grad():
        logits_actual = features @ linear.weight.t() + linear.bias
        probs_actual = F.softmax(logits_actual, dim=1)
        one_hot = F.one_hot(labels, num_classes=num_classes).float()
        residuals = probs_actual - one_hot  # (N, C)

        scale = -1.0 / (2.0 * lambda_l2 * n)
        W_reconstructed = scale * (residuals.t() @ features)  # (C, d)
        b_reconstructed = scale * residuals.sum(dim=0)         # (C,)

        logits_reconstructed = features @ W_reconstructed.t() + b_reconstructed

        # KL divergence: KL(p_actual || p_reconstructed)
        log_probs_actual = F.log_softmax(logits_actual, dim=1)
        log_probs_recon = F.log_softmax(logits_reconstructed, dim=1)
        kl_div = F.kl_div(
            log_probs_recon, probs_actual, reduction='batchmean'
        ).item()

        logit_mse = F.mse_loss(logits_reconstructed, logits_actual).item()
        max_W_diff = (linear.weight - W_reconstructed).abs().max().item()
        max_b_diff = (linear.bias - b_reconstructed).abs().max().item()

    logger.info(
        f"Decomposition validation: KL={kl_div:.2e}, "
        f"logit_MSE={logit_mse:.2e}, max_W_diff={max_W_diff:.2e}, "
        f"max_b_diff={max_b_diff:.2e}"
    )
    return {
        "kl_div": kl_div,
        "logit_mse": logit_mse,
        "max_W_diff": max_W_diff,
        "max_b_diff": max_b_diff,
    }


def plot_figure_x(
    ts_eval_path: Path,
    finetuned_metrics: Dict[int, Dict[str, float]],
    output_path: Path,
) -> None:
    """Plot temperature-scaled vs L2 fine-tuned test loss across k values.

    Shows that L2 fine-tuning of the final layer preserves the double
    descent shape, validating its use for the decomposition.

    Args:
        ts_eval_path: Path to temperature_scaled_evaluation.jsonl.
        finetuned_metrics: k -> {test_loss, ...}.
        output_path: Where to save the figure.
    """
    # Load temperature-scaled evaluation
    ts_records = []
    with open(ts_eval_path) as f:
        for line in f:
            if line.strip():
                ts_records.append(json.loads(line))
    ts_records.sort(key=lambda r: r["k"])

    ts_k = [r["k"] for r in ts_records]
    ts_test_loss = [r["ts_loss"] for r in ts_records]

    ft_k = sorted(finetuned_metrics.keys())
    ft_test_loss = [finetuned_metrics[k]["test_loss"] for k in ft_k]

    # Academic style
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['DejaVu Serif', 'Times New Roman', 'Times'],
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'legend.frameon': True,
        'legend.framealpha': 0.9,
        'legend.edgecolor': '0.8',
        'axes.linewidth': 0.8,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.major.size': 4,
        'ytick.major.size': 4,
        'xtick.top': True,
        'ytick.right': False,
    })

    line_kw = dict(
        linewidth=1.8,
        solid_capstyle='round',
        solid_joinstyle='round',
        dash_capstyle='round',
        dash_joinstyle='round',
        antialiased=True,
    )
    marker_kw = dict(marker='o', markersize=5, markeredgewidth=1.1,
                     markerfacecolor='white')

    color_ts = '#1f77b4'     # blue
    color_ft = '#2ca02c'     # green

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=(7, 4), dpi=200)

    ax.plot(ts_k, ts_test_loss, color=color_ts, linestyle='-',
            label='Temperature scaled', **marker_kw, **line_kw)
    ax.plot(ft_k, ft_test_loss, color=color_ft, linestyle='-',
            label='L2 fine-tuned', **marker_kw, **line_kw)

    ax.set_xlabel('Width parameter k')
    ax.set_ylabel('Cross-entropy loss')
    ax.legend(loc='upper right')
    ax.set_title('Temperature Scaled vs L2 Fine-tuned Test Loss')

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved Figure X to {output_path}")
