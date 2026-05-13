#!/usr/bin/env python3
"""DEPRECATED — superseded by plot_influence_share.py (which uses share_M,
the canonical metric). Will be removed on the next refactor pass.

Single-panel plot: mean |influence| ratio (mislabeled / clean) vs k.

Reads per-training-point influence files (influence_k{k}{suffix}.jsonl) and
computes the mis/clean influence ratio for each k, then plots a single curve.

Usage:
    python -m jl.double_descent.influence.plot_mislabel_ratio \\
        data/resnet18/figure_2_es_final_run/influence \\
        --suffix _lam1e-05_fp64_tg1e-10 \\
        --output cifar_mislabel_ratio_lam1e-5.png
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def discover_ks(influence_dir: Path, suffix: str):
    pat = re.compile(rf"^influence_k(\d+){re.escape(suffix)}\.jsonl$")
    ks = []
    for p in influence_dir.iterdir():
        m = pat.match(p.name)
        if m:
            ks.append(int(m.group(1)))
    return sorted(ks)


def mis_clean_ratio(influence_dir: Path, k: int, suffix: str) -> float:
    mis, clean = [], []
    with open(influence_dir / f"influence_k{k}{suffix}.jsonl") as f:
        for line in f:
            r = json.loads(line)
            v = abs(r["influence"])
            (mis if r["mislabeled"] else clean).append(v)
    return float(np.mean(mis) / np.mean(clean))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("influence_dir", type=str)
    parser.add_argument(
        "--suffix", default="",
        help="File suffix used in the influence sweep, e.g. '_lam1e-05_fp64_tg1e-10'",
    )
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument(
        "--title", default=None,
        help="Plot title (default: derived from suffix)",
    )
    args = parser.parse_args()

    influence_dir = Path(args.influence_dir)
    ks = discover_ks(influence_dir, args.suffix)
    if not ks:
        raise FileNotFoundError(
            f"No influence_k*{args.suffix}.jsonl in {influence_dir}"
        )

    ratios = [mis_clean_ratio(influence_dir, k, args.suffix) for k in ks]

    out = Path(args.output) if args.output else (
        influence_dir / f"mislabel_ratio{args.suffix}.png"
    )

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.direction": "in",
        "ytick.direction": "in",
    })
    fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=150)
    ax.plot(
        ks, ratios, "o-",
        color="#d62728",
        markersize=6,
        markerfacecolor="white",
        markeredgewidth=1.4,
        linewidth=1.8,
    )
    ax.axhline(
        1.0, color="0.5", linestyle="--", linewidth=0.8, alpha=0.7,
        label="ratio = 1 (no signal)",
    )
    title = args.title or f"Mean |influence| ratio (mislabeled / clean)  |  suffix='{args.suffix}'"
    ax.set_title(title)
    ax.set_xlabel("Width parameter k")
    ax.set_ylabel("Influence ratio (mislabeled / clean)")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
