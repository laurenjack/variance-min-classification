#!/usr/bin/env python3
"""DEPRECATED — superseded by plot_influence_share.py (which uses share_M,
the canonical metric). Will be removed on the next refactor pass.

Plot influence-ratio + top-1%/top-5% fraction-mislabeled vs k.

Reads:
  - summary.jsonl: per-k influence_ratio (mislabeled / clean)
  - influence_k*.jsonl: per-training-point influence + mislabel flag

Usage:
    python -m jl.double_descent.influence.plot_influence_vs_k \\
        ./data/resnet18/04-11-1602/influence \\
        --output ./data/resnet18/04-11-1602/influence/influence_vs_k.png
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_summary(influence_dir: Path):
    rows = []
    with open(influence_dir / "summary.jsonl") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    rows.sort(key=lambda r: r["k"])
    return rows


def load_per_point(influence_dir: Path, k: int):
    influences = []
    mislabels = []
    with open(influence_dir / f"influence_k{k}.jsonl") as f:
        for line in f:
            r = json.loads(line)
            influences.append(r["influence"])
            mislabels.append(r["mislabeled"])
    return np.asarray(influences), np.asarray(mislabels)


def fraction_mislabeled_in_top(
    influences: np.ndarray, mislabels: np.ndarray, frac: float
) -> float:
    n = int(len(influences) * frac)
    if n == 0:
        return 0.0
    top_idx = np.argpartition(influences, -n)[-n:]
    return float(mislabels[top_idx].mean())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("influence_dir", type=str)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--top1-pct", type=float, default=0.01)
    parser.add_argument("--top5-pct", type=float, default=0.05)
    args = parser.parse_args()

    influence_dir = Path(args.influence_dir)
    output_path = Path(args.output) if args.output else (influence_dir / "influence_vs_k.png")

    summary = load_summary(influence_dir)
    ks = [r["k"] for r in summary]
    ratios = [r["influence_ratio"] for r in summary]

    top1 = []
    top5 = []
    base_rate = None
    for r in summary:
        k = r["k"]
        infl, mis = load_per_point(influence_dir, k)
        if base_rate is None:
            base_rate = float(mis.mean())
        top1.append(fraction_mislabeled_in_top(infl, mis, args.top1_pct))
        top5.append(fraction_mislabeled_in_top(infl, mis, args.top5_pct))

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
    })
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), dpi=150, sharex=True)

    ax = axes[0]
    ax.plot(ks, ratios, "o-", color="#d62728", markersize=5, markerfacecolor="white",
            markeredgewidth=1.2, linewidth=1.5)
    ax.axhline(1.0, color="0.5", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.set_ylabel("Influence ratio (mislabeled / clean)")
    ax.set_title("Training Point Influence Decomposition")
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(ks, top1, "o-", color="#d62728", markersize=5, markerfacecolor="white",
            markeredgewidth=1.2, linewidth=1.5,
            label=f"Top {args.top1_pct*100:.0f}%")
    ax.plot(ks, top5, "o-", color="#1f77b4", markersize=4, markerfacecolor="#1f77b4",
            linewidth=1.5, label=f"Top {args.top5_pct*100:.0f}%")
    ax.axhline(base_rate, color="0.5", linestyle="--", linewidth=0.8, alpha=0.7,
               label=f"Base rate ({base_rate*100:.1f}%)")
    ax.set_ylabel("Fraction mislabeled")
    ax.set_xlabel("Width parameter k")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
