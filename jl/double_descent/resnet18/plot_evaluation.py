#!/usr/bin/env python3
"""Plot variance evaluation results: bias-variance decomposition vs k.

Produces a single figure showing test loss, Jensen Gap (variance), and implied bias.

Usage:
    python -m jl.double_descent.resnet18.plot_evaluation \
        ./data/resnet18_variance/03-01-1010/evaluation.jsonl \
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
        List of dicts sorted by k.
    """
    results = []
    with open(eval_path, "r") as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))

    if not results:
        raise FileNotFoundError(f"No evaluation results found in {eval_path}")

    return sorted(results, key=lambda r: r["k"])


def plot_evaluation(eval_path: str, output_dir: str, temperature_scaled: bool = False) -> None:
    """Plot bias-variance decomposition vs k.

    Args:
        eval_path: Path to evaluation.jsonl file.
        output_dir: Directory to save plot.
        temperature_scaled: If True, append "(Temperature Scaled)" to plot title.
    """
    results = load_evaluation(eval_path)

    k_values = [r["k"] for r in results]
    test_losses = [r["mean_test_loss"] for r in results]
    jensen_gaps = [r["mean_jensen_gap"] for r in results]
    implied_bias = [tl - jg for tl, jg in zip(test_losses, jensen_gaps)]

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    title = "Bias-Variance Decomposition vs k (ResNet18 on CIFAR-10)"
    if temperature_scaled:
        title += " (Temperature Scaled)"

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    ax.plot(k_values, test_losses, "-o", color="tab:blue", lw=2, markersize=4, label="Mean Test Loss")
    ax.plot(k_values, jensen_gaps, "-o", color="tab:orange", lw=2, markersize=4, label="Jensen Gap (variance)")
    ax.plot(k_values, implied_bias, "-o", color="tab:green", lw=2, markersize=4, label="Implied Bias")
    ax.set_xlabel("ResNet18 width parameter (k)")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.set_title(title)
    ax.set_ylim(bottom=0)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    output_path = Path(output_dir) / "bias_variance.png"
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot variance evaluation: bias-variance decomposition vs k"
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
    parser.add_argument(
        "--temperature-scaled",
        action="store_true",
        help="Label plot as temperature-scaled results",
    )
    args = parser.parse_args()

    output_dir = args.output_dir if args.output_dir else str(Path(args.eval_path).parent)
    plot_evaluation(args.eval_path, output_dir, temperature_scaled=args.temperature_scaled)


if __name__ == "__main__":
    main()
