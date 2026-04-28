#!/usr/bin/env python3
"""Plot bucket-mean influence vs oracle (no scatter, no error bars).

Produces 4 plots:
  - bucket_means_p_10.png     (deciles by p_oracle)
  - bucket_means_p_100.png    (centiles by p_oracle)
  - bucket_means_logp_10.png  (deciles by log p_oracle)
  - bucket_means_logp_100.png (centiles by log p_oracle)

Usage:
    python -m jl.double_descent.influence.transformer_oracle_correlation_buckets \\
        --influence-path .../influence_train.pt \\
        --oracle-path .../train_split0_log_probs.pt \\
        --output-dir .../<run-dir>
"""

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from jl.double_descent.influence.transformer_oracle_correlation import join_by_sentence

logger = logging.getLogger(__name__)


def bucket_means(x: np.ndarray, y: np.ndarray, n_bins: int):
    """Return (bucket_centers, bucket_means) for n_bins quantile buckets of x."""
    edges = np.quantile(x, np.linspace(0, 1, n_bins + 1))
    idx = np.clip(np.digitize(x, edges[1:-1]), 0, n_bins - 1)
    centers = np.empty(n_bins)
    means = np.empty(n_bins)
    for i in range(n_bins):
        sel = idx == i
        centers[i] = x[sel].mean()
        means[i] = y[sel].mean()
    return centers, means


def plot_one(centers, means, x_label, title, output_path):
    plt.rcParams.update({"font.family": "serif", "font.size": 11})
    fig, ax = plt.subplots(1, 1, figsize=(7, 5), dpi=150)
    ax.plot(centers, means, "o-", color="#d62728", markersize=4,
            markerfacecolor="white", markeredgewidth=1.2, linewidth=1.2)
    ax.axhline(1.0, color="0.5", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Influence score (mean=1)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--influence-path", required=True)
    parser.add_argument("--oracle-path", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    inf = torch.load(args.influence_path, map_location="cpu", weights_only=False)
    influence = inf["influence"].float()
    inf_targets = inf["target_ids"].long()
    inf_offsets = inf["sentence_offsets"].long()

    oracle = torch.load(args.oracle_path, map_location="cpu", weights_only=False)
    oracle_log_probs = oracle["log_probs"].float()
    oracle_targets = oracle["target_ids"].long()
    oracle_offsets = oracle["sentence_offsets"].long()

    influence_aligned, oracle_aligned, _, n_mismatch = join_by_sentence(
        influence, inf_targets, inf_offsets,
        oracle_log_probs, oracle_targets, oracle_offsets,
    )

    log_p = oracle_aligned.numpy()
    p = oracle_aligned.exp().numpy()
    inf_np = influence_aligned.numpy()
    logger.info(f"Aligned tokens: {len(inf_np)}, mismatches: {n_mismatch}")

    for n_bins, label in [(10, "10"), (100, "100")]:
        c_p, m_p = bucket_means(p, inf_np, n_bins)
        plot_one(
            c_p, m_p,
            x_label="M2M100-12B p(y_i | context_i)",
            title=f"Oracle prob vs influence (means, {n_bins} buckets)",
            output_path=output_dir / f"bucket_means_p_{label}.png",
        )

        c_lp, m_lp = bucket_means(log_p, inf_np, n_bins)
        plot_one(
            c_lp, m_lp,
            x_label="M2M100-12B log p(y_i | context_i)",
            title=f"Oracle log-prob vs influence (means, {n_bins} buckets)",
            output_path=output_dir / f"bucket_means_logp_{label}.png",
        )

    logger.info(f"Wrote 4 bucket-mean plots to {output_dir}")


if __name__ == "__main__":
    main()
