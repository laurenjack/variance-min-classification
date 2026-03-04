#!/usr/bin/env python3
"""Plot variance evaluation results: bias-variance decomposition vs d_model.

Produces a single figure with shared y-axis showing four lines:
  - Mean test loss across 8 training splits
  - Jensen Gap (exact variance term using true labels)
  - Implied bias = test_loss - jensen_gap
  - KL(q_bar || q_j) (variance proxy)

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
    """Plot bias-variance decomposition vs d_model on a shared y-axis.

    Args:
        eval_path: Path to evaluation.jsonl file.
        output_dir: Directory to save plot.
    """
    results = load_evaluation(eval_path)

    d_models = [r["d_model"] for r in results]
    test_losses = [r["mean_test_loss"] for r in results]
    kl_values = [r["mean_kl"] for r in results]
    jensen_gaps = [r["mean_jensen_gap"] for r in results]
    implied_bias = [tl - jg for tl, jg in zip(test_losses, jensen_gaps)]

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

    ax.plot(d_models, test_losses, "-o", color="tab:blue", lw=2, label="Mean Test Loss")
    ax.plot(d_models, jensen_gaps, "-s", color="tab:orange", lw=2, label="Jensen Gap (variance)")
    ax.plot(d_models, implied_bias, "-^", color="tab:green", lw=2, label="Implied Bias")
    ax.plot(d_models, kl_values, "-d", color="tab:red", lw=2, label="KL(q̄ ∥ q) (variance proxy)")

    ax.set_xlabel("Transformer embedding dimension (d_model)")
    ax.set_ylabel("Cross-Entropy / KL Divergence (nats)")
    ax.set_title("Bias-Variance Decomposition across d_model")
    ax.set_ylim(bottom=0)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = Path(output_dir) / "variance_evaluation.png"
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
