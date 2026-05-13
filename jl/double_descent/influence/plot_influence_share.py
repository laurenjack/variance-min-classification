#!/usr/bin/env python3
"""Canonical influence-share plot: mean share_M(t) vs k.

For each training point i and test point t, define the contribution magnitude
    c_{i,t} = ||r_i||_2 * |phi_i . phi_t|
where r_i is the FT model's residual on training point i and phi_i, phi_t are
its feature embeddings. Then the per-test share of total influence going to
the mislabeled bucket M is
    share_M(t) = sum_{i in M} c_{i,t} / sum_i c_{i,t} .
The plot reports the mean of share_M(t) over test points t, per k.

Baseline reference = the mislabel rate p (typically ~0.151). When share_M
sits above p, mislabels are over-represented in the influence decomposition.

Inputs (all produced by `influence_main.py`):
  - share_per_test_k{k}{suffix}.npy   (one [N_test] array per k)

Usage:
    python -m jl.double_descent.influence.plot_influence_share \\
        data/resnet18/figure_2_es_final_run/influence \\
        --suffix _lam1e-05_fp64_tg1e-10 \\
        --output cifar_influence_share_lam1e-5.png
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def discover_ks(influence_dir: Path, suffix: str):
    pat = re.compile(rf"^share_per_test_k(\d+){re.escape(suffix)}\.npy$")
    ks = []
    for p in influence_dir.iterdir():
        m = pat.match(p.name)
        if m:
            ks.append(int(m.group(1)))
    return sorted(ks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("influence_dir", type=str)
    parser.add_argument(
        "--suffix", default="",
        help="File suffix used in the influence sweep, e.g. "
             "'_lam1e-05_fp64_tg1e-10'",
    )
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument(
        "--title", default=None, help="Plot title (default: derived from suffix)"
    )
    parser.add_argument(
        "--mislabel-rate", type=float, default=0.151,
        help="Baseline mislabel rate (default 0.151 = 15.1%% CIFAR-10 noise). "
             "Drawn as a horizontal reference line.",
    )
    args = parser.parse_args()

    influence_dir = Path(args.influence_dir)
    ks = discover_ks(influence_dir, args.suffix)
    if not ks:
        raise FileNotFoundError(
            f"No share_per_test_k*{args.suffix}.npy in {influence_dir}"
        )

    mean_shares = []
    for k in ks:
        share = np.load(
            influence_dir / f"share_per_test_k{k}{args.suffix}.npy"
        )
        mean_shares.append(float(share.mean()))

    out = (
        Path(args.output) if args.output
        else influence_dir / f"influence_share{args.suffix}.png"
    )

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.direction": "in",
        "ytick.direction": "in",
    })
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    ax.plot(
        ks, mean_shares, "o-",
        color="#d62728",
        markersize=5,
        markerfacecolor="white",
        markeredgewidth=1.2,
        linewidth=1.5,
        label="Mean share_M(t) over test points",
    )
    ax.axhline(
        args.mislabel_rate,
        color="0.5", linestyle="--", linewidth=0.8, alpha=0.8,
        label=f"Mislabel rate ({args.mislabel_rate*100:.1f}%)",
    )
    title = args.title or (
        f"Influence share by mislabeled training points vs k  "
        f"(suffix='{args.suffix}')"
    )
    ax.set_title(title)
    ax.set_xlabel("Width parameter k")
    ax.set_ylabel(
        r"$\mathrm{share}_M(t) = \frac{\sum_{i \in M} c_{i,t}}{\sum_i c_{i,t}}$, "
        r"mean over $t$"
    )
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
