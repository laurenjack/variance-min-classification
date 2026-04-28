#!/usr/bin/env python3
"""Variant of transformer_oracle_correlation that plots p (not log p) on the
x-axis. Useful when the per-train influence factor is in probability space
(e.g., the correct-token-only formula uses 1 - p_model(y_i)).

Reuses join_by_sentence from transformer_oracle_correlation.

Usage:
    python -m jl.double_descent.influence.transformer_oracle_correlation_p \\
        --influence-path .../influence_train.pt \\
        --oracle-path .../train_split0_log_probs.pt \\
        --output-dir .../<run-dir>
"""

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats

from jl.double_descent.influence.transformer_oracle_correlation import join_by_sentence

logger = logging.getLogger(__name__)


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

    n_total_sent = len(inf_offsets) - 1
    logger.info(
        f"Aligned {n_total_sent - n_mismatch}/{n_total_sent} sentences, "
        f"{influence_aligned.numel()} tokens."
    )

    # Convert to probability space
    p_oracle = oracle_aligned.exp().numpy()
    inf_np = influence_aligned.numpy()

    pearson_r, pearson_p = stats.pearsonr(p_oracle, inf_np)
    spearman_r, spearman_p = stats.spearmanr(p_oracle, inf_np)
    logger.info(
        f"Pearson r = {pearson_r:.4f} (p={pearson_p:.2e}); "
        f"Spearman rho = {spearman_r:.4f} (p={spearman_p:.2e})"
    )

    # Bin by deciles of p_oracle
    n_bins = 10
    bin_edges = np.quantile(p_oracle, np.linspace(0, 1, n_bins + 1))
    bin_idx = np.clip(np.digitize(p_oracle, bin_edges[1:-1]), 0, n_bins - 1)
    bin_centers, bin_means, bin_stds, bin_counts = [], [], [], []
    for i in range(n_bins):
        sel = bin_idx == i
        bin_centers.append(float(p_oracle[sel].mean()))
        bin_means.append(float(inf_np[sel].mean()))
        bin_stds.append(float(inf_np[sel].std()))
        bin_counts.append(int(sel.sum()))

    summary = {
        "n_aligned_tokens": int(influence_aligned.numel()),
        "n_total_sentences": int(n_total_sent),
        "n_mismatch_sentences": int(n_mismatch),
        "x_quantity": "p_oracle (exp of stored log_p)",
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "spearman_r": float(spearman_r),
        "spearman_p": float(spearman_p),
        "decile_p_centers": bin_centers,
        "decile_influence_means": bin_means,
        "decile_influence_stds": bin_stds,
        "decile_counts": bin_counts,
    }
    with open(output_dir / "oracle_correlation_p.json", "w") as f:
        json.dump(summary, f, indent=2)

    plt.rcParams.update({"font.family": "serif", "font.size": 11})
    fig, ax = plt.subplots(1, 1, figsize=(7, 5), dpi=150)

    n_plot = min(50000, len(inf_np))
    rng = np.random.default_rng(0)
    sample_idx = rng.choice(len(inf_np), size=n_plot, replace=False)
    ax.scatter(
        p_oracle[sample_idx], inf_np[sample_idx],
        s=2, alpha=0.15, color="#1f77b4", rasterized=True,
        label=f"per-token (n={n_plot:,})",
    )
    ax.errorbar(
        bin_centers, bin_means, yerr=bin_stds,
        fmt="o-", color="#d62728", markersize=6, linewidth=1.5,
        capsize=3, label="decile mean ± std",
    )
    ax.axhline(1.0, color="0.5", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.set_xlabel("M2M100-12B p(y_i | context_i)")
    ax.set_ylabel("Influence score (mean=1)")
    ax.set_title(
        f"Oracle prob vs influence  "
        f"(Pearson r={pearson_r:.3f}, Spearman ρ={spearman_r:.3f})"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "oracle_vs_influence_p.png", bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Wrote outputs to {output_dir}")


if __name__ == "__main__":
    main()
