#!/usr/bin/env python3
"""Plot variance evaluation results: mean test loss and KL divergence vs d_model.

Produces a single figure with dual y-axes:
  - Left (blue): Mean test loss across 8 training splits
  - Right (red): KL(q_bar || q_j) averaged over models and test tokens

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
        List of dicts sorted by d_model, each with keys:
        d_model, mean_test_loss, mean_kl, num_models, total_tokens.
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
    """Plot mean test loss and KL divergence vs d_model with dual y-axes.

    Args:
        eval_path: Path to evaluation.jsonl file.
        output_dir: Directory to save plot.
    """
    results = load_evaluation(eval_path)

    d_models = [r["d_model"] for r in results]
    test_losses = [r["mean_test_loss"] for r in results]
    kl_values = [r["mean_kl"] for r in results]

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    fig, ax1 = plt.subplots(figsize=(10, 6), dpi=150)

    color_loss = "tab:blue"
    ax1.set_xlabel("Transformer embedding dimension (d_model)")
    ax1.set_ylabel("Mean Test Loss", color=color_loss)
    ax1.plot(d_models, test_losses, "-o", color=color_loss, lw=2, label="Mean Test Loss")
    ax1.tick_params(axis="y", labelcolor=color_loss)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    color_kl = "tab:red"
    ax2.set_ylabel("KL(q̄ ∥ q)", color=color_kl)
    ax2.plot(d_models, kl_values, "-s", color=color_kl, lw=2, label="KL(q̄ ∥ q)")
    ax2.tick_params(axis="y", labelcolor=color_kl)

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right")

    ax1.set_title("Variance Analysis: Bias-Variance Decomposition across d_model")

    plt.tight_layout()
    output_path = Path(output_dir) / "variance_evaluation.png"
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot variance evaluation: mean test loss and KL divergence vs d_model"
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
