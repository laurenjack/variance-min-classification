#!/usr/bin/env python3
"""Extract per-bucket shadow shares at the early-stop point from a
completed bucket-shadow run.

The trainer already logs `shadow_shares` (cumulative, normalized) to the
metrics JSONL every `log_interval` steps and writes `best_valid_step` to
the final `bucket_shares_d{d}_{Nk}.json`.  This script joins the two:
for each d_model, find the JSONL record at (or nearest to)
best_valid_step and emit a sibling `bucket_shares_es_d{d}_{Nk}.json`
plus a combined plot.

Usage:
    python -m jl.double_descent.transformer.extract_early_stop_shares \\
        ./output/shadow_adamw/05-09-0627
"""

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source_dir", help="Directory with bucket_shares_d*.json + metrics_d*.jsonl")
    p.add_argument("--output-dir", default=None,
                   help="Where to put plots/json (default: --source-dir)")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(name)s] %(message)s",
                        datefmt="%H:%M:%S")
    src = Path(args.source_dir)
    out = Path(args.output_dir) if args.output_dir else src
    out.mkdir(parents=True, exist_ok=True)

    summary_files = sorted(src.glob("bucket_shares_d*_*k.json"))
    if not summary_files:
        raise FileNotFoundError(f"No bucket_shares_d*.json under {src}")
    logger.info(f"Found {len(summary_files)} share summaries")

    final_summaries = []
    es_summaries = []
    for sf in summary_files:
        s = json.loads(sf.read_text())
        d = s["d_model"]
        samples_k = s["train_samples"] // 1000
        best_step = s["best_valid_step"]
        best_loss = s["best_valid_loss"]

        metrics_file = src / f"metrics_d{d}_{samples_k}k.jsonl"
        if not metrics_file.exists():
            logger.warning(f"  d={d}: no {metrics_file.name}, skipping early-stop")
            continue

        records = [json.loads(l) for l in metrics_file.read_text().splitlines()
                   if "shadow_shares" in l]
        if not records:
            logger.warning(f"  d={d}: no shadow_shares records in metrics, skipping")
            continue

        # Find record at (or nearest to) best_valid_step
        nearest = min(records, key=lambda r: abs(r["step"] - best_step))
        delta = nearest["step"] - best_step
        es_shares = nearest["shadow_shares"]
        es_norms = nearest["shadow_norms"]

        es_summary = {
            **{k: v for k, v in s.items()
               if k in ("d_model", "train_samples", "n_bins",
                        "bucket_entropy_edges", "bucket_entropy_centers")},
            "best_valid_step": best_step,
            "best_valid_loss": best_loss,
            "snapshot_step": nearest["step"],
            "snapshot_step_delta": delta,
            "shadow_norms": es_norms,
            "shadow_shares": es_shares,
            "valid_loss_at_snapshot": nearest.get("valid_loss"),
            "train_loss_at_snapshot": nearest.get("train_loss"),
        }
        es_path = src / f"bucket_shares_es_d{d}_{samples_k}k.json"
        es_path.write_text(json.dumps(es_summary, indent=2))
        logger.info(
            f"  d={d}: best_valid_step={best_step} -> snapshot at step {nearest['step']} "
            f"(Δ={delta:+d}), valid_loss={best_loss:.3f}; wrote {es_path.name}"
        )
        final_summaries.append(s)
        es_summaries.append(es_summary)

    if not es_summaries:
        logger.error("No early-stop summaries built; exiting.")
        return

    # Combined plot: final shares (solid) vs early-stop shares (dashed)
    final_summaries.sort(key=lambda s: s["d_model"])
    es_summaries.sort(key=lambda s: s["d_model"])

    centers = np.array(final_summaries[0]["bucket_entropy_centers"])
    n_bins = final_summaries[0]["n_bins"]
    baseline = 1.0 / n_bins

    plt.rcParams.update({"font.family": "serif", "font.size": 11})
    fig, ax = plt.subplots(1, 1, figsize=(9, 5.5), dpi=150)
    colors = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e"]
    for i, (s_final, s_es) in enumerate(zip(final_summaries, es_summaries)):
        c = colors[i % len(colors)]
        d = s_final["d_model"]
        ax.plot(
            centers, s_final["shadow_shares"], "o-", color=c,
            markersize=5, markerfacecolor="white", markeredgewidth=1.2,
            linewidth=1.5, label=f"d={d} final",
        )
        ax.plot(
            centers, s_es["shadow_shares"], "s--", color=c,
            markersize=5, markerfacecolor=c, markeredgewidth=0,
            linewidth=1.2, alpha=0.85,
            label=f"d={d} @ best valid (step {s_es['best_valid_step']})",
        )
    ax.axhline(baseline, color="0.5", linestyle=":", linewidth=0.8, alpha=0.8,
               label=f"Baseline (1/{n_bins})")
    ax.set_xlabel("Bucket entropy center (nats):  -log p_oracle(y_i)")
    ax.set_ylabel("Bucket share of cumulative shadow ||W_b||₂  (sums to 1)")
    ax.set_title(
        "Per-bucket shadow shares: final vs early-stop snapshot\n"
        "(circles = end of training; squares = best valid_loss step)"
    )
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    plot_path = out / "bucket_shares_final_vs_early_stop.png"
    fig.savefig(plot_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Wrote {plot_path}")


if __name__ == "__main__":
    main()
