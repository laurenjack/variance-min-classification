#!/usr/bin/env python3
"""Plot clean-trained vs baseline (noisy) ResNet18 test error vs k.

Reads two run directories (each with metrics_k{K}.jsonl) and plots their
final-epoch test_error per k on a single panel. Same CIFAR-10 test set
for both, so the comparison is direct.

Usage:
    python -m jl.double_descent.resnet18.plot_clean_vs_baseline \
        --clean-dir   ./data/resnet18_clean/05-16-1523 \
        --baseline-dir ./data/resnet18/04-11-1602 \
        --output-dir   ./data/resnet18_clean/05-16-1523
"""

import argparse
import json
import logging
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

METRICS_RE = re.compile(r"metrics_k(\d+)\.jsonl$")


def _parse_rows(f: Path) -> list:
    rows = []
    for line in f.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def load_final_per_k(run_dir: Path) -> dict:
    """Return {k: final-epoch metrics dict} for every metrics_k*.jsonl in run_dir."""
    out = {}
    for f in sorted(run_dir.glob("metrics_k*.jsonl")):
        m = METRICS_RE.search(f.name)
        if not m:
            continue
        k = int(m.group(1))
        rows = _parse_rows(f)
        if not rows:
            logger.warning(f"  {f.name}: no parseable rows, skipping")
            continue
        out[k] = rows[-1]
    return out


def load_es_per_k(run_dir: Path) -> dict:
    """Return {k: best-val-loss epoch metrics dict} for every metrics_k*.jsonl."""
    out = {}
    for f in sorted(run_dir.glob("metrics_k*.jsonl")):
        m = METRICS_RE.search(f.name)
        if not m:
            continue
        k = int(m.group(1))
        rows = _parse_rows(f)
        has_val = [r for r in rows if r.get("val_loss") is not None]
        if not has_val:
            logger.warning(f"  {f.name}: no val_loss column, can't derive ES")
            continue
        best = min(has_val, key=lambda r: r["val_loss"])
        out[k] = best
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--clean-dir", required=True)
    p.add_argument("--baseline-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument(
        "--metric", default="test_error",
        choices=["test_error", "test_loss"],
        help="Which final-epoch metric to plot (default: test_error).",
    )
    p.add_argument(
        "--log-x", action="store_true",
        help="Use log scale on the k axis.",
    )
    p.add_argument(
        "--no-es", action="store_true",
        help="Omit the baseline early-stop line (just final-epoch curves).",
    )
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    clean_dir = Path(args.clean_dir)
    base_dir = Path(args.baseline_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    clean_final = load_final_per_k(clean_dir)
    base_final = load_final_per_k(base_dir)
    clean_es = load_es_per_k(clean_dir)
    base_es = load_es_per_k(base_dir)
    common_ks = sorted(set(clean_final.keys()) & set(base_final.keys()))
    if not common_ks:
        raise RuntimeError("No common k values between clean and baseline dirs.")
    logger.info(f"Common k values: {common_ks}")

    clean_y = np.array([clean_final[k][args.metric] for k in common_ks])
    base_y = np.array([base_final[k][args.metric] for k in common_ks])
    clean_es_y = np.array([clean_es[k][args.metric] for k in common_ks])
    base_es_y = np.array([base_es[k][args.metric] for k in common_ks])
    x = np.array(common_ks)

    plt.rcParams.update({
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 15,
        "legend.fontsize": 11,
        "mathtext.fontset": "cm",
    })
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    ax.plot(x, base_y, "-o", color="#d62728", lw=2, markersize=4, label="Mislabeled")
    if not args.no_es:
        ax.plot(
            x, base_es_y, "--o", color="#d62728", alpha=0.65,
            lw=2, markersize=4, label="Mislabeled - early-stop",
        )
    ax.plot(x, clean_y, "-s", color="#1f77b4", lw=2, markersize=4, label="Clean")
    ax.set_xlabel(r"Model Width ($k$)")
    ax.set_ylabel({"test_error": "Test Error", "test_loss": "Test Loss"}[args.metric])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if args.log_x:
        ax.set_xscale("log")
    ax.legend(frameon=False)
    fig.tight_layout()

    suffix = f"_{args.metric}" + ("_logx" if args.log_x else "")
    plot_path = out_dir / f"resnet_clean_vs_baseline{suffix}.png"
    fig.savefig(plot_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Wrote {plot_path}")

    # Print the comparison table to stdout for quick inspection.
    print(f"\n{'k':>4}  {'base.fin':>9}  {'base.es':>9}  {'clean.fin':>10}")
    for k, bf, be, cf in zip(common_ks, base_y, base_es_y, clean_y):
        print(f"{k:>4}  {bf:>9.4f}  {be:>9.4f}  {cf:>10.4f}")


if __name__ == "__main__":
    main()
