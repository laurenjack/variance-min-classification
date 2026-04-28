#!/usr/bin/env python3
"""Per-test-point influence-share analysis for the transformer, bucketed by
oracle entropy of the training token.

For each (train i, test t) define
    c_{i,t} = (1 - p_model(y_i)) * |phi_i . phi_t|

(correct-token residual magnitude * absolute feature dot product, the
per-(i,t) factor inside our correct-token-only RMS influence).

Bucket train tokens into n quantile buckets by entropy = -log p_oracle(y_i).
For each test token t and bucket b:
    share_b(t) = sum over i in B_b of c_{i,t}  /  sum over all i of c_{i,t}

Aggregate over test points (mean) -> n numbers per run. Baseline = 1/n
(each bucket holds 1/n of train tokens by construction). The
support-vector theory predicts share_b > baseline at high entropy
(surprising tokens), share_b < baseline at low entropy.

Reuses cached features+untied_output_proj from prior transformer_main runs.
Multiple runs are combined into a single comparison plot.

Usage:
    python -m jl.double_descent.influence.transformer_influence_share_buckets \\
        --run d=96:./data/transformer_m2m100_variance/04-02-1520/influence/d96_split0_rms_lam1e-3 \\
        --run d=32:./data/transformer_m2m100_variance/04-02-1520/influence/d32_split0_rms_lam1e-3 \\
        --oracle-path ./data/iwslt14.m2m100.de-en/train_split0_log_probs.pt \\
        --output-dir ./data/transformer_m2m100_variance/04-02-1520/influence/share_buckets
"""

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def parse_run(s: str):
    if "=" not in s:
        raise argparse.ArgumentTypeError(f"--run needs LABEL=PATH, got {s!r}")
    label, path = s.split("=", 1)
    return label, Path(path)


def load_run_assets(source_dir: Path, device: torch.device, factor: str):
    """Load features_train, features_test, output_proj and compute the per-train factor.

    factor='magnitude': train_factor_i = ||r_i||_2 = ||softmax(W phi_i + b) - one_hot(y_i)||_2
    factor='correct':   train_factor_i = 1 - p_model(y_i)
    """
    train_blob = torch.load(source_dir / "features_train.pt", map_location="cpu",
                            weights_only=False)
    test_blob = torch.load(source_dir / "features_test.pt", map_location="cpu",
                           weights_only=False)
    proj_state = torch.load(source_dir / "untied_output_proj.pt", map_location="cpu",
                            weights_only=True)

    phi_train = train_blob["features"].float().to(device)
    y_train = train_blob["target_ids"].long().to(device)
    train_offsets = train_blob["sentence_offsets"]
    phi_test = test_blob["features"].float().to(device)

    d_model = phi_train.size(1)
    vocab_size = proj_state["weight"].size(0)
    output_proj = nn.Linear(d_model, vocab_size, bias=True).to(device)
    output_proj.load_state_dict({k: v.to(device) for k, v in proj_state.items()})
    output_proj.eval()

    n = phi_train.size(0)
    train_factor = torch.empty(n, device=device, dtype=torch.float32)
    chunk = 4096
    with torch.no_grad():
        for s in range(0, n, chunk):
            e = min(s + chunk, n)
            logits = output_proj(phi_train[s:e])
            probs = F.softmax(logits, dim=-1)
            if factor == "magnitude":
                # ||softmax - one_hot||_2 per token
                probs_at_y = probs.gather(-1, y_train[s:e].unsqueeze(-1)).squeeze(-1)
                # ||r||_2^2 = sum_j p_j^2  - 2 p_y + 1
                sum_sq = probs.pow(2).sum(dim=-1)
                train_factor[s:e] = (sum_sq - 2 * probs_at_y + 1.0).clamp_min(0.0).sqrt()
            elif factor == "correct":
                probs_at_y = probs.gather(-1, y_train[s:e].unsqueeze(-1)).squeeze(-1)
                train_factor[s:e] = 1.0 - probs_at_y
            else:
                raise ValueError(f"Unknown factor: {factor!r}")

    return {
        "phi_train": phi_train,
        "phi_test": phi_test,
        "train_factor": train_factor,
        "train_target_ids": train_blob["target_ids"].long(),
        "train_offsets": train_offsets,
    }


def align_oracle(train_target_ids, train_offsets, oracle):
    """Sentence-by-sentence assert alignment, return per-train-token log p."""
    o_log_p = oracle["log_probs"].float()
    o_targets = oracle["target_ids"].long()
    o_offsets = oracle["sentence_offsets"].long()

    n_sent_train = len(train_offsets) - 1
    n_sent_oracle = len(o_offsets) - 1
    assert n_sent_train == n_sent_oracle, (
        f"Sentence count mismatch: train {n_sent_train}, oracle {n_sent_oracle}"
    )

    aligned_log_p = torch.empty(train_target_ids.shape[0], dtype=torch.float32)
    n_mismatch = 0
    for s in range(n_sent_train):
        ia, ib = int(train_offsets[s]), int(train_offsets[s + 1])
        oa, ob = int(o_offsets[s]), int(o_offsets[s + 1])
        n_inf = ib - ia
        n_oracle = ob - oa
        if n_inf != n_oracle or not torch.equal(train_target_ids[ia:ib], o_targets[oa:ob]):
            n_mismatch += 1
            # Drop sentence by setting NaN; we'll filter later
            aligned_log_p[ia:ib] = float("nan")
            continue
        aligned_log_p[ia:ib] = o_log_p[oa:ob]
    return aligned_log_p, n_mismatch


def bucket_share(
    phi_train: torch.Tensor,
    train_factor: torch.Tensor,
    bucket_id: torch.Tensor,
    phi_test: torch.Tensor,
    n_bins: int,
    train_chunk: int = 4096,
    test_chunk: int = 4096,
):
    """Compute mean over test points of bucket-shares.

    Returns: array of shape (n_bins,) summing to ~1.
    """
    n_train = phi_train.size(0)
    n_test = phi_test.size(0)
    device = phi_train.device

    # Per-test-point accumulators
    bucket_contrib = torch.zeros(n_bins, n_test, device=device, dtype=torch.float32)
    total_contrib = torch.zeros(n_test, device=device, dtype=torch.float32)

    with torch.no_grad():
        for ts in range(0, n_test, test_chunk):
            te = min(ts + test_chunk, n_test)
            phi_test_block = phi_test[ts:te]                      # [tc, d]
            for trs in range(0, n_train, train_chunk):
                tre = min(trs + train_chunk, n_train)
                dots = phi_train[trs:tre] @ phi_test_block.t()    # [trc, tc]
                contrib = train_factor[trs:tre].unsqueeze(1) * dots.abs()
                # Per-bucket sum
                local_buckets = bucket_id[trs:tre]
                for b in range(n_bins):
                    mask = local_buckets == b
                    if mask.any():
                        bucket_contrib[b, ts:te] += contrib[mask].sum(dim=0)
                total_contrib[ts:te] += contrib.sum(dim=0)

    # Per-test share, then mean
    safe_total = total_contrib.clamp_min(1e-30)
    share_per_test = bucket_contrib / safe_total.unsqueeze(0)     # [bins, n_test]
    mean_share = share_per_test.mean(dim=1).cpu().numpy()
    return mean_share


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", type=parse_run, required=True,
                        help="label=source_dir, repeatable")
    parser.add_argument("--oracle-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--n-bins", type=int, default=10)
    parser.add_argument("--factor", choices=["magnitude", "correct"], default="magnitude",
                        help="Per-train factor: 'magnitude' = ||r_i||_2 (matches CIFAR), "
                             "'correct' = 1 - p_model(y_i)")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logger.info(f"Using device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    oracle = torch.load(args.oracle_path, map_location="cpu", weights_only=False)
    logger.info(f"Loaded oracle log-probs from {args.oracle_path}")

    # Define common quantile bin edges using the FIRST run's training entropy.
    # Using the oracle directly (independent of run) gives identical buckets
    # for all runs since they're all on the same train split.
    log_p_oracle_full = oracle["log_probs"].float().numpy()
    entropy_full = -log_p_oracle_full
    edges = np.quantile(entropy_full, np.linspace(0, 1, args.n_bins + 1))
    bucket_centers = np.array([
        entropy_full[(entropy_full >= edges[i]) & (entropy_full <= edges[i + 1])].mean()
        for i in range(args.n_bins)
    ])
    logger.info(
        f"Bucket entropy edges (nats): {edges.tolist()}\n"
        f"Bucket entropy centers (nats): {bucket_centers.tolist()}"
    )

    all_results = {}
    for label, source_dir in args.run:
        logger.info(f"=== run {label}: {source_dir} (factor={args.factor}) ===")
        a = load_run_assets(source_dir, device, factor=args.factor)

        aligned_log_p, n_mismatch = align_oracle(
            a["train_target_ids"], a["train_offsets"], oracle,
        )
        n_total = aligned_log_p.shape[0]
        valid = ~torch.isnan(aligned_log_p)
        n_valid = int(valid.sum().item())
        logger.info(f"  aligned: {n_valid}/{n_total} train tokens (mismatched sentences: {n_mismatch})")

        if n_valid < n_total:
            phi_train = a["phi_train"][valid]
            train_factor = a["train_factor"][valid]
            entropy_train = (-aligned_log_p[valid]).to(device)
        else:
            phi_train = a["phi_train"]
            train_factor = a["train_factor"]
            entropy_train = (-aligned_log_p).to(device)

        # Assign to global buckets defined by oracle quantiles
        bucket_id = torch.from_numpy(
            np.clip(np.digitize(entropy_train.cpu().numpy(), edges[1:-1]),
                    0, args.n_bins - 1)
        ).long().to(device)

        # Sanity: bucket population
        for b in range(args.n_bins):
            count = int((bucket_id == b).sum().item())
            logger.info(f"  bucket {b}: {count} train tokens ({100*count/n_valid:.2f}%)")

        share = bucket_share(
            phi_train, train_factor, bucket_id, a["phi_test"],
            n_bins=args.n_bins,
        )
        logger.info(f"  shares: {share.tolist()}")
        all_results[label] = {
            "n_valid_train": n_valid,
            "n_test": int(a["phi_test"].size(0)),
            "share_per_bucket": share.tolist(),
        }

        # Free GPU memory between runs
        del a, phi_train, train_factor, entropy_train, bucket_id
        torch.cuda.empty_cache()

    summary = {
        "n_bins": args.n_bins,
        "baseline_share": 1.0 / args.n_bins,
        "factor": args.factor,
        "bucket_entropy_edges": edges.tolist(),
        "bucket_entropy_centers": bucket_centers.tolist(),
        "runs": all_results,
    }
    with open(output_dir / "share_buckets.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Wrote {output_dir / 'share_buckets.json'}")

    # Plot
    plt.rcParams.update({"font.family": "serif", "font.size": 11})
    fig, ax = plt.subplots(1, 1, figsize=(8, 5), dpi=150)
    colors = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e"]
    for i, (label, res) in enumerate(all_results.items()):
        ax.plot(
            bucket_centers, res["share_per_bucket"],
            "o-", color=colors[i % len(colors)],
            markersize=5, markerfacecolor="white", markeredgewidth=1.2,
            linewidth=1.5, label=label,
        )
    ax.axhline(1.0 / args.n_bins, color="0.5", linestyle="--", linewidth=0.8,
               alpha=0.8, label=f"Baseline (1/{args.n_bins})")
    ax.set_xlabel("Bucket entropy center (nats):  -log p_oracle(y_i)")
    ax.set_ylabel(f"Mean share over test points  (sums to 1 across {args.n_bins} buckets)")
    if args.factor == "magnitude":
        formula_tex = r"$c_{i,t} = \|r_i\|_2 \cdot |\varphi_i \cdot \varphi_t|$"
    else:
        formula_tex = r"$c_{i,t} = (1 - p_{model}(y_i)) \cdot |\varphi_i \cdot \varphi_t|$"
    ax.set_title(
        "Influence-share by entropy bucket of training tokens\n" + formula_tex
    )
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    plot_path = output_dir / "share_buckets.png"
    fig.savefig(plot_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Wrote {plot_path}")


if __name__ == "__main__":
    main()
