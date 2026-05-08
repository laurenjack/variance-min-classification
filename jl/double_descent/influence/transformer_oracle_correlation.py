#!/usr/bin/env python3
"""Correlate per-training-token influence with the oracle's log p(y_i | context_i).

Reads:
  - influence_train.pt produced by transformer_main: per-token influence, target_ids,
    sentence_offsets.
  - {train_split}_log_probs.pt produced by extract_m2m100_reference: per-token
    M2M100-12B log-probability of y_i, target_ids, sentence_offsets.

Joins them per-token (sentence-by-sentence; padding/length disagreements are
flagged), computes Pearson and Spearman correlation, writes a JSON summary
and a scatter plot.

Usage:
    python -m jl.double_descent.influence.transformer_oracle_correlation \\
        --influence-path .../influence/d96_split0/influence_train.pt \\
        --oracle-path .../iwslt14.m2m100.de-en/train_split0_log_probs.pt \\
        --output-dir .../influence/d96_split0
"""

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats

logger = logging.getLogger(__name__)


def join_by_sentence(
    influence: torch.Tensor,
    inf_targets: torch.Tensor,
    inf_offsets: torch.Tensor,
    oracle_log_probs: torch.Tensor,
    oracle_targets: torch.Tensor,
    oracle_offsets: torch.Tensor,
):
    """Align per-token influence and oracle log-prob, sentence by sentence.

    The two pipelines tokenize differently:
      - influence_train.pt counts non-pad target positions in our compact-vocab
        small Transformer (decoder feeds [BOS, t1, ..., tN, EOS], target shift
        gives [t1, ..., tN, EOS]).
      - oracle_log_probs.pt counts non-pad target positions under M2M100's own
        tokenization (target [__en__, t1, ..., tN, </s>] shifted by 1 gives
        [t1, ..., tN, </s>]).

    M2M100's BPE is the same as our compact vocab (the compact vocab is exactly
    the subset of M2M100 IDs that appears in train), so per-sentence token
    counts should match. We assert they do; if not, that sentence is dropped
    and counted as a mismatch.
    """
    n_sent_inf = len(inf_offsets) - 1
    n_sent_oracle = len(oracle_offsets) - 1
    assert n_sent_inf == n_sent_oracle, (
        f"Sentence count mismatch: influence has {n_sent_inf}, oracle has {n_sent_oracle}"
    )

    matched_influence = []
    matched_oracle = []
    matched_targets = []

    n_mismatch = 0
    for s in range(n_sent_inf):
        ia, ib = int(inf_offsets[s].item()), int(inf_offsets[s + 1].item())
        oa, ob = int(oracle_offsets[s].item()), int(oracle_offsets[s + 1].item())
        n_inf = ib - ia
        n_oracle = ob - oa

        if n_inf != n_oracle:
            n_mismatch += 1
            continue

        # Sanity: target IDs should agree position by position
        inf_t = inf_targets[ia:ib]
        oracle_t = oracle_targets[oa:ob]
        if not torch.equal(inf_t, oracle_t):
            # Token mismatch despite same length — drop sentence
            n_mismatch += 1
            continue

        matched_influence.append(influence[ia:ib])
        matched_oracle.append(oracle_log_probs[oa:ob])
        matched_targets.append(inf_t)

    if not matched_influence:
        raise RuntimeError("No sentences matched between influence and oracle outputs.")

    return (
        torch.cat(matched_influence),
        torch.cat(matched_oracle),
        torch.cat(matched_targets),
        n_mismatch,
    )


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

    logger.info(f"Loading influence from {args.influence_path}")
    inf = torch.load(args.influence_path, map_location="cpu", weights_only=False)
    influence = inf["influence"].float()
    inf_targets = inf["target_ids"].long()
    inf_offsets = inf["sentence_offsets"].long()

    logger.info(f"Loading oracle log-probs from {args.oracle_path}")
    oracle = torch.load(args.oracle_path, map_location="cpu", weights_only=False)
    oracle_log_probs = oracle["log_probs"].float()
    oracle_targets = oracle["target_ids"].long()
    oracle_offsets = oracle["sentence_offsets"].long()

    influence_aligned, oracle_aligned, targets_aligned, n_mismatch = join_by_sentence(
        influence, inf_targets, inf_offsets,
        oracle_log_probs, oracle_targets, oracle_offsets,
    )

    n_total_sent = len(inf_offsets) - 1
    logger.info(
        f"Aligned {n_total_sent - n_mismatch}/{n_total_sent} sentences "
        f"({n_mismatch} mismatches dropped), {influence_aligned.numel()} tokens."
    )

    inf_np = influence_aligned.numpy()
    log_p_np = oracle_aligned.numpy()

    pearson_r, pearson_p = stats.pearsonr(log_p_np, inf_np)
    spearman_r, spearman_p = stats.spearmanr(log_p_np, inf_np)
    logger.info(
        f"Pearson r = {pearson_r:.4f} (p={pearson_p:.2e}); "
        f"Spearman rho = {spearman_r:.4f} (p={spearman_p:.2e})"
    )

    # Binned means: average influence in deciles of log p(y)
    n_bins = 10
    bin_edges = np.quantile(log_p_np, np.linspace(0, 1, n_bins + 1))
    bin_idx = np.clip(np.digitize(log_p_np, bin_edges[1:-1]), 0, n_bins - 1)
    bin_centers = []
    bin_means = []
    bin_stds = []
    bin_counts = []
    for i in range(n_bins):
        sel = bin_idx == i
        bin_centers.append(float(log_p_np[sel].mean()))
        bin_means.append(float(inf_np[sel].mean()))
        bin_stds.append(float(inf_np[sel].std()))
        bin_counts.append(int(sel.sum()))

    summary = {
        "n_aligned_tokens": int(influence_aligned.numel()),
        "n_total_sentences": int(n_total_sent),
        "n_mismatch_sentences": int(n_mismatch),
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "spearman_r": float(spearman_r),
        "spearman_p": float(spearman_p),
        "decile_log_p_centers": bin_centers,
        "decile_influence_means": bin_means,
        "decile_influence_stds": bin_stds,
        "decile_counts": bin_counts,
    }
    with open(output_dir / "oracle_correlation.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Plot: scatter (subsampled) + decile means overlay
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
    })
    fig, ax = plt.subplots(1, 1, figsize=(7, 5), dpi=150)

    n_plot = min(50000, len(inf_np))
    rng = np.random.default_rng(0)
    sample_idx = rng.choice(len(inf_np), size=n_plot, replace=False)
    ax.scatter(
        log_p_np[sample_idx], inf_np[sample_idx],
        s=2, alpha=0.15, color="#1f77b4", rasterized=True,
        label=f"per-token (n={n_plot:,})",
    )
    ax.errorbar(
        bin_centers, bin_means, yerr=bin_stds,
        fmt="o-", color="#d62728", markersize=6, linewidth=1.5,
        capsize=3, label="decile mean ± std",
    )
    ax.axhline(1.0, color="0.5", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.set_xlabel("M2M100-12B log p(y_i | context_i)")
    ax.set_ylabel("Influence score (mean=1)")
    ax.set_title(
        f"Oracle log-prob vs influence  "
        f"(Pearson r={pearson_r:.3f}, Spearman ρ={spearman_r:.3f})"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "oracle_vs_influence.png", bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Wrote outputs to {output_dir}")


if __name__ == "__main__":
    main()
