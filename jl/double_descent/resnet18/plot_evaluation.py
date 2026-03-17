#!/usr/bin/env python3
"""Plot evaluation results for ResNet18 main (non-variance) runs.

Produces a figure with 2 subplots:
- Top: Train/Test error vs k
- Bottom: Train/Test loss vs k, with ECE on right y-axis (dual axis)

Usage:
    python -m jl.double_descent.resnet18.plot_evaluation \
        ./output/resnet18/03-01-1010/evaluation.jsonl \
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


def plot_evaluation(eval_path: str, output_dir: str, noise_level: float = 0.15) -> None:
    """Plot error and loss vs k, with ECE on secondary axis.

    Creates a single figure with 2 subplots:
    - Top: Train/Test error vs k
    - Bottom: Train/Test loss vs k (left axis), ECE vs k (right axis)

    Args:
        eval_path: Path to evaluation.jsonl file.
        output_dir: Directory to save plot.
        noise_level: Label noise fraction for title.
    """
    results = load_evaluation(eval_path)

    k_values = [r["k"] for r in results]
    train_error = [r["train_error"] for r in results]
    test_error = [r["test_error"] for r in results]
    train_loss = [r["train_loss"] for r in results]
    test_loss = [r["test_loss"] for r in results]
    ece = [r["ece"] for r in results]

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), dpi=150, sharex=True)

    # Error plot (top)
    ax1.plot(k_values, test_error, '-o', color='blue', lw=2, label='Test Error')
    ax1.plot(k_values, train_error, '--o', color='blue', lw=2, alpha=0.5, label='Train Error')
    ax1.set_ylabel('Error Rate')
    ax1.set_title(f'Double Descent: ResNet18 on CIFAR-10 ({int(noise_level*100)}% label noise)')
    ax1.set_ylim(bottom=0)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Loss plot (bottom) with ECE on right y-axis
    ax2.plot(k_values, test_loss, '-o', color='red', lw=2, label='Test Loss')
    ax2.plot(k_values, train_loss, '--o', color='red', lw=2, alpha=0.5, label='Train Loss')
    ax2.set_xlabel('ResNet18 width parameter k')
    ax2.set_ylabel('Cross-Entropy Loss')
    ax2.set_ylim(bottom=0)
    ax2.grid(True, alpha=0.3)

    # ECE on right y-axis (dual axis)
    ax2_ece = ax2.twinx()
    ax2_ece.plot(k_values, ece, '-s', color='green', lw=2, markersize=5, label='ECE')
    ax2_ece.set_ylabel('Expected Calibration Error', color='green')
    ax2_ece.tick_params(axis='y', labelcolor='green')
    ax2_ece.set_ylim(bottom=0)

    # Combined legend for loss subplot
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_ece.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    plt.tight_layout()
    output_path = Path(output_dir) / 'resnet18_evaluation.png'
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot ResNet18 evaluation: error/loss vs k with ECE"
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
        "--noise-level",
        type=float,
        default=0.15,
        help="Label noise fraction (for plot title)",
    )
    args = parser.parse_args()

    output_dir = args.output_dir if args.output_dir else str(Path(args.eval_path).parent)
    plot_evaluation(args.eval_path, output_dir, noise_level=args.noise_level)


if __name__ == "__main__":
    main()
