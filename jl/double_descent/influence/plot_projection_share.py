#!/usr/bin/env python3
"""DEPRECATED -- superseded by plot_influence_share.py (which now uses the
canonical projection-based mislabeled influence share). The
projection_per_test_k*.npz input is no longer produced; this script will
fail at runtime. Will be removed in the next refactor pass.

Plot signed and absolute projection-based influence shares vs k.

For each training point i and test point t, define the projection scalar
    alpha_{i,t} = <contribution_{i,t}, u_t>
where contribution_{i,t} = -1/(2*lambda*n) * r_i * (phi_i . phi_t) is the
representer-decomposition contribution to test point t's class logits, and
u_t = logits_t / ||logits_t|| is the FT logits direction. By construction
    sum_i alpha_{i,t} = ||logits_t||.

We compare two shares of the mislabeled bucket M:
    share_signed(t) = sum_{i in M} alpha_{i,t} / sum_i alpha_{i,t}
    share_abs(t)    = sum_{i in M} |alpha_{i,t}| / sum_i |alpha_{i,t}|

Their gap reveals cancellation: if mislabeled training points contribute
vectors that cancel each other, share_signed << share_abs. The
cancellation index sum_i alpha_{i,t} / sum_i |alpha_{i,t}| is in [0,1]:
1 = no cancellation, 0 = perfect cancellation.

Inputs (produced by influence_main.py):
  - projection_per_test_k{k}{suffix}.npz  with arrays signed_M, signed_T,
    abs_M, abs_T (each [N_test]).

Usage:
    python -m jl.double_descent.influence.plot_projection_share \\
        data/resnet18/figure_2_es_final_run/influence \\
        --suffix _lam1e-05_fp64_distill_tg1e-10 \\
        --output projection_share_lam1e-5.png
"""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def discover_ks(influence_dir: Path, suffix: str):
    pat = re.compile(rf"^projection_per_test_k(\d+){re.escape(suffix)}\.npz$")
    ks = []
    for p in influence_dir.iterdir():
        m = pat.match(p.name)
        if m:
            ks.append(int(m.group(1)))
    return sorted(ks)


def collect(influence_dir: Path, suffix: str):
    ks = discover_ks(influence_dir, suffix)
    if not ks:
        raise FileNotFoundError(
            f"No projection_per_test_k*{suffix}.npz in {influence_dir}"
        )
    signed_shares, abs_shares, cancellation = [], [], []
    for k in ks:
        d = np.load(influence_dir / f"projection_per_test_k{k}{suffix}.npz")
        signed_share_t = d["signed_M"] / np.maximum(np.abs(d["signed_T"]), 1e-30)
        abs_share_t = d["abs_M"] / np.maximum(d["abs_T"], 1e-30)
        cancel_t = d["signed_T"] / np.maximum(d["abs_T"], 1e-30)
        signed_shares.append(float(signed_share_t.mean()))
        abs_shares.append(float(abs_share_t.mean()))
        cancellation.append(float(cancel_t.mean()))
    return ks, signed_shares, abs_shares, cancellation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("influence_dir", type=str)
    parser.add_argument("--suffix", default="")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument(
        "--mislabel-rate", type=float, default=0.151,
        help="Baseline mislabel rate for share panels.",
    )
    args = parser.parse_args()

    influence_dir = Path(args.influence_dir)
    ks, signed_shares, abs_shares, cancellation = collect(influence_dir, args.suffix)

    out = (
        Path(args.output) if args.output
        else influence_dir / f"projection_share{args.suffix}.png"
    )

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "DejaVu Serif", "Times New Roman", "Times"],
        "font.size": 12,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "axes.linewidth": 0.9,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
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
        "legend.fontsize": 11,
        "legend.frameon": False,
    })

    fig, axes = plt.subplots(3, 1, figsize=(7.5, 9.2), sharex=True, dpi=200)

    # Panel 1: signed share
    ax = axes[0]
    ax.plot(ks, signed_shares, "o-", color="#0b3d91", markersize=4.5,
            markerfacecolor="white", markeredgewidth=1.2, linewidth=1.6,
            label="Mislabeled signed share", zorder=3)
    ax.axhline(args.mislabel_rate, color="0.4", linestyle=(0, (4, 2)),
               linewidth=1.0, label="Mislabel rate", zorder=2)
    ax.set_ylabel("Signed share")
    ax.legend(loc="upper right", handlelength=2.2, borderpad=0.6)
    ax.grid(True, which="major", linewidth=0.4, alpha=0.5)
    ax.set_axisbelow(True)

    # Panel 2: abs share
    ax = axes[1]
    ax.plot(ks, abs_shares, "s-", color="#a4262c", markersize=4.5,
            markerfacecolor="white", markeredgewidth=1.2, linewidth=1.6,
            label="Mislabeled absolute share", zorder=3)
    ax.axhline(args.mislabel_rate, color="0.4", linestyle=(0, (4, 2)),
               linewidth=1.0, label="Mislabel rate", zorder=2)
    ax.set_ylabel("Absolute share")
    ax.legend(loc="upper right", handlelength=2.2, borderpad=0.6)
    ax.grid(True, which="major", linewidth=0.4, alpha=0.5)
    ax.set_axisbelow(True)

    # Panel 3: cancellation index
    ax = axes[2]
    ax.plot(ks, cancellation, "^-", color="#3f7d20", markersize=5,
            markerfacecolor="white", markeredgewidth=1.2, linewidth=1.6,
            label=r"Cancellation index $\sum_i \alpha_{i,t} / \sum_i |\alpha_{i,t}|$",
            zorder=3)
    ax.axhline(1.0, color="0.4", linestyle=(0, (4, 2)), linewidth=1.0,
               label="No cancellation", zorder=2)
    ax.set_xlabel(r"Width parameter $k$")
    ax.set_ylabel("Cancellation index")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right", handlelength=2.2, borderpad=0.6)
    ax.grid(True, which="major", linewidth=0.4, alpha=0.5)
    ax.set_axisbelow(True)

    axes[-1].set_xlim(0, max(ks) + 2)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
