#!/usr/bin/env python3
"""Plot variance evaluation results: bias-variance decomposition vs d_model.

Produces a single figure showing test loss, Jensen Gap (variance), and implied bias.

Usage:
    python -m jl.double_descent.transformer.plot_evaluation \
        ./data/transformer_variance/03-01-1010/evaluation.jsonl \
        --output-dir ./data
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt


def load_evaluation(eval_path: str) -> List[Dict]:
    """Load evaluation results from JSONL file.

    Returns:
        List of dicts sorted by d_model.
    """
    results = []
    with open(eval_path, "r") as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))

    if not results:
        raise FileNotFoundError(f"No evaluation results found in {eval_path}")

    return sorted(results, key=lambda r: r["d_model"])


def plot_evaluation(eval_path: str, output_dir: str) -> None:
    """Plot bias-variance decomposition vs d_model.

    Supports two formats:
    - Legacy: mean_test_loss, mean_jensen_gap → plots test loss, Jensen gap, entropy+bias
    - Distributional: mean_test_loss, entropy, bias, variance → plots all four terms

    Args:
        eval_path: Path to evaluation.jsonl file.
        output_dir: Directory to save plot.
    """
    results = load_evaluation(eval_path)
    distributional = "entropy" in results[0]

    d_models = [r["d_model"] for r in results]
    test_losses = [r["mean_test_loss"] for r in results]

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    title = "Bias-Variance Decomposition"

    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 15,
        'legend.fontsize': 11,
        'mathtext.fontset': 'cm',
    })

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)

    ax.plot(d_models, test_losses, "-o", color="#1f77b4", lw=2, markersize=4, label="Test Loss")

    if distributional:
        biases = [r["bias"] for r in results]
        variances = [r["variance"] for r in results]
        entropy_val = results[0]["entropy"]  # constant across d_model

        ax.plot(d_models, biases, "-^", color="#2ca02c", lw=2, markersize=4, label="Bias")
        ax.plot(d_models, variances, "-s", color="#d62728", lw=2, markersize=4, label="Variance (Jensen Gap)")
        ax.axhline(y=entropy_val, color="#7f7f7f", ls="--", lw=1.5, label=f"Entropy ({entropy_val:.2f})")
    else:
        jensen_gaps = [r["mean_jensen_gap"] for r in results]
        entropy_bias = [tl - jg for tl, jg in zip(test_losses, jensen_gaps)]

        ax.plot(d_models, jensen_gaps, "-s", color="#d62728", lw=2, markersize=4, label="Jensen Gap (variance)")
        ax.plot(d_models, entropy_bias, "-^", color="#2ca02c", lw=2, markersize=4, label="Entropy + Bias")

    ax.set_xlabel(r"Model Width ($d_{model}$)")
    ax.set_ylabel("Cross-Entropy (nats)")
    ax.set_title(title)
    ax.set_ylim(bottom=0)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(frameon=False)

    plt.tight_layout()
    output_path = Path(output_dir) / "bias_variance.png"
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot variance evaluation: bias-variance decomposition vs d_model"
    )
    parser.add_argument(
        "eval_path",
        type=str,
        help="Path to evaluation.jsonl file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save plot (default: same directory as eval_path)",
    )
    args = parser.parse_args()

    output_dir = args.output_dir if args.output_dir else str(Path(args.eval_path).parent)
    plot_evaluation(args.eval_path, output_dir)


if __name__ == "__main__":
    main()
