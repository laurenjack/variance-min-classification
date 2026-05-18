#!/usr/bin/env python3
"""Plot variance evaluation results: bias-variance decomposition vs k.

Produces a 2-panel figure: top = FINAL checkpoints, bottom = early-stop
checkpoints. Both panels share the x-axis (model width k).

Usage:
    # Point at the run directory:
    python -m jl.double_descent.resnet18.plot_variance_evaluation \
        ./data/resnet18_variance/05-17-1050

    # Or at the final evaluation.jsonl (parent's early_stop/ used for the
    # second panel):
    python -m jl.double_descent.resnet18.plot_variance_evaluation \
        ./data/resnet18_variance/05-17-1050/evaluation.jsonl
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt


def load_evaluation(eval_path: Path) -> List[Dict]:
    """Load evaluation results from JSONL file, sorted by k."""
    results = []
    with open(eval_path, "r") as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    if not results:
        raise FileNotFoundError(f"No evaluation results found in {eval_path}")
    return sorted(results, key=lambda r: r["k"])


def _plot_panel(ax, results: List[Dict], panel_title: str, show_xlabel: bool) -> None:
    k_values = [r["k"] for r in results]
    test_losses = [r["mean_test_loss"] for r in results]
    jensen_gaps = [r["mean_jensen_gap"] for r in results]
    entropy_bias = [tl - jg for tl, jg in zip(test_losses, jensen_gaps)]

    ax.plot(k_values, test_losses, "-o", color="#1f77b4", lw=2, markersize=4, label="Test Loss")
    ax.plot(k_values, jensen_gaps, "-s", color="#d62728", lw=2, markersize=4, label="Jensen Gap (variance)")
    ax.plot(k_values, entropy_bias, "-^", color="#2ca02c", lw=2, markersize=4, label="Entropy + Bias")

    ax.set_ylabel("Cross-Entropy Loss")
    ax.set_title(panel_title, fontsize=13)
    ax.set_ylim(bottom=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)
    if show_xlabel:
        ax.set_xlabel(r"Model Width ($k$)")


def plot_evaluation(
    final_path: Path,
    es_path: Optional[Path],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 15,
        'legend.fontsize': 11,
        'mathtext.fontset': 'cm',
    })

    final_results = load_evaluation(final_path)

    if es_path is not None and es_path.exists():
        es_results = load_evaluation(es_path)
        fig, (ax_top, ax_bot) = plt.subplots(
            2, 1, figsize=(8, 9), dpi=150, sharex=True,
        )
        _plot_panel(ax_top, final_results, "Final checkpoint", show_xlabel=False)
        _plot_panel(ax_bot, es_results, "Early-stop checkpoint", show_xlabel=True)
        fig.suptitle("Bias-Variance Decomposition (ResNet18 on CIFAR-10)", y=1.0)
        fig.tight_layout()
    else:
        fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
        _plot_panel(ax, final_results, "", show_xlabel=True)
        ax.set_title("Bias-Variance Decomposition (ResNet18 on CIFAR-10)")
        fig.tight_layout()

    output_path = output_dir / "bias_variance.png"
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to {output_path}")


def _resolve_paths(input_path: str) -> tuple[Path, Optional[Path], Path]:
    p = Path(input_path)
    if p.is_dir():
        run_dir = p
        final_jsonl = run_dir / "evaluation.jsonl"
    else:
        final_jsonl = p
        run_dir = p.parent
    if not final_jsonl.exists():
        raise FileNotFoundError(f"Missing final evaluation: {final_jsonl}")
    es_jsonl = run_dir / "early_stop" / "evaluation.jsonl"
    return final_jsonl, (es_jsonl if es_jsonl.exists() else None), run_dir


def main():
    parser = argparse.ArgumentParser(
        description="Plot variance evaluation: bias-variance decomposition vs k"
    )
    parser.add_argument(
        "eval_path",
        type=str,
        help="Path to a run directory or its evaluation.jsonl file. The "
             "matching early_stop/evaluation.jsonl is auto-discovered.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save plot (default: run dir).",
    )
    args = parser.parse_args()

    final_path, es_path, run_dir = _resolve_paths(args.eval_path)
    output_dir = Path(args.output_dir) if args.output_dir else run_dir
    plot_evaluation(final_path, es_path, output_dir)


if __name__ == "__main__":
    main()
