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

Inputs (produced by `influence_main.py`):
  - share_per_test_k{k}{suffix}.npy   (one [N_test] array per k)

Usage:
    python -m jl.double_descent.influence.plot_influence_share \\
        data/resnet18/figure_2_es_final_run/influence \\
        --suffix _lam1e-05_fp64_tg1e-10 \\
        --output cifar_influence_share_lam1e-5.png
"""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def discover_ks(influence_dir: Path, suffix: str):
    pat = re.compile(rf"^projections_k(\d+){re.escape(suffix)}\.npz$")
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
        "--mislabel-rate", type=float, default=0.151,
        help="Baseline mislabel rate. Drawn as a horizontal reference line.",
    )
    args = parser.parse_args()

    influence_dir = Path(args.influence_dir)
    ks = discover_ks(influence_dir, args.suffix)
    if not ks:
        raise FileNotFoundError(
            f"No share_per_test_k*{args.suffix}.npy in {influence_dir}"
        )

    mean_shares = []
    correct_shares = []
    for k in ks:
        d = np.load(influence_dir / f"projections_k{k}{args.suffix}.npz")
        pm, pc = d["proj_M"], d["proj_C"]
        denom = np.abs(pm) + np.abs(pc)
        denom = np.maximum(denom, 1e-30)
        abs_M = np.abs(pm) / denom
        abs_C = np.abs(pc) / denom
        mean_shares.append(float(abs_M.mean()))
        correct_shares.append(float(abs_C.mean()))

    out = (
        Path(args.output) if args.output
        else influence_dir / f"influence_share{args.suffix}.png"
    )

    # Academic-paper styling
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "DejaVu Serif", "Times New Roman", "Times"],
        "font.size": 13,
        "axes.labelsize": 14,
        "axes.titlesize": 14,
        "axes.linewidth": 0.9,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.minor.size": 2.5,
        "ytick.minor.size": 2.5,
        "xtick.top": True,
        "ytick.right": True,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "legend.fontsize": 12,
        "legend.frameon": False,
    })

    fig, ax = plt.subplots(figsize=(7.0, 4.4), dpi=200)

    ax.plot(
        ks, mean_shares,
        "o-",
        color="#0b3d91",
        markersize=4.5,
        markerfacecolor="white",
        markeredgewidth=1.2,
        linewidth=1.6,
        label="Mislabeled influence share",
        zorder=3,
    )
    ax.plot(
        ks, correct_shares,
        "s-",
        color="#a4262c",
        markersize=4.5,
        markerfacecolor="white",
        markeredgewidth=1.2,
        linewidth=1.6,
        label="Correctly-labeled influence share",
        zorder=3,
    )
    ax.axhline(
        args.mislabel_rate,
        color="0.4",
        linestyle=(0, (4, 2)),
        linewidth=1.0,
        label="Mislabel rate",
        zorder=2,
    )

    ax.set_xlabel("Width parameter $k$")
    ax.set_ylabel("Influence share")
    ax.set_xlim(0, max(ks) + 2)
    ax.legend(loc="center right", handlelength=2.2, borderpad=0.6)

    # Subtle grid
    ax.grid(True, which="major", linewidth=0.4, alpha=0.5)
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
