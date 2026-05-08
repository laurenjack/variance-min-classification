#!/usr/bin/env python3
"""Plot per-bucket shadow shares from a bucket_shadow training run.

Reads bucket_shares_d*.json files and renders a single-panel plot with
one line per d_model, x-axis = bucket entropy center, y-axis = global
L2 share.  Baseline = 1/n_bins is drawn as a dashed line.

Usage:
    python -m jl.double_descent.transformer.plot_bucket_shadow \\
        ./output/shadow/06-15-1234 --output-dir ./data/shadow/06-15-1234
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
    p.add_argument("source_dir", help="Directory containing bucket_shares_d*.json")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--prefix", default="bucket_shares_d",
                   help="Filename prefix to glob (default: bucket_shares_d)")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(name)s] %(message)s",
                        datefmt="%H:%M:%S")
    src = Path(args.source_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    files = sorted(src.glob(f"{args.prefix}*.json"))
    if not files:
        raise FileNotFoundError(f"No {args.prefix}*.json under {src}")
    logger.info(f"Found {len(files)} share files")

    summaries = []
    for f in files:
        s = json.loads(f.read_text())
        summaries.append(s)

    summaries.sort(key=lambda s: s["d_model"])

    # All runs should share entropy edges (same oracle).  Use the first.
    centers = np.array(summaries[0]["bucket_entropy_centers"])
    n_bins = summaries[0]["n_bins"]
    baseline = 1.0 / n_bins

    plt.rcParams.update({"font.family": "serif", "font.size": 11})
    fig, ax = plt.subplots(1, 1, figsize=(8, 5), dpi=150)
    colors = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e"]
    for i, s in enumerate(summaries):
        shares = s["shadow_shares"]
        ax.plot(
            centers, shares, "o-",
            color=colors[i % len(colors)],
            markersize=5, markerfacecolor="white", markeredgewidth=1.2,
            linewidth=1.5, label=f"d={s['d_model']}",
        )
    ax.axhline(baseline, color="0.5", linestyle="--", linewidth=0.8, alpha=0.8,
               label=f"Baseline (1/{n_bins})")
    ax.set_xlabel("Bucket entropy center (nats):  -log p_oracle(y_i)")
    ax.set_ylabel("Bucket share of cumulative shadow ||W_b||_2  (sums to 1)")
    ax.set_title(
        "Per-bucket SGD-shadow contribution to model parameters\n"
        "(global L2 across all layers, init=0, vanilla-SGD shadows under AdamW main)"
    )
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    plot_path = out / "bucket_shadow_shares.png"
    fig.savefig(plot_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Wrote {plot_path}")

    # also dump a flat csv-ish json
    summary = {
        "n_bins": n_bins,
        "bucket_entropy_centers": centers.tolist(),
        "runs": {
            f"d={s['d_model']}": {
                "shadow_shares": s["shadow_shares"],
                "shadow_norms": s["shadow_norms"],
            }
            for s in summaries
        },
    }
    json_path = out / "bucket_shadow_shares.json"
    json_path.write_text(json.dumps(summary, indent=2))
    logger.info(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
