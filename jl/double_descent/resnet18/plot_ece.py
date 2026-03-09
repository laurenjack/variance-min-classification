#!/usr/bin/env python3
"""Plot Expected Calibration Error vs k for ResNet18 variance models.

Usage:
    python -m jl.double_descent.resnet18.plot_ece \
        ./data/resnet18_variance/03-01-1010/ece.jsonl \
        --output-dir ./data
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt


def load_ece(eval_path: str) -> List[Dict]:
    results = []
    with open(eval_path, "r") as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))

    if not results:
        raise FileNotFoundError(f"No ECE results found in {eval_path}")

    return sorted(results, key=lambda r: r["k"])


def plot_ece(eval_path: str, output_dir: str) -> None:
    results = load_ece(eval_path)

    k_values = [r["k"] for r in results]
    ece_values = [r["ece"] for r in results]

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    ax.plot(k_values, ece_values, "-o", color="tab:red", lw=2, markersize=4)
    ax.set_xlabel("ResNet18 width parameter (k)")
    ax.set_ylabel("Expected Calibration Error")
    ax.set_title("ECE vs k (ResNet18 on CIFAR-10, split 0)")
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    output_path = Path(output_dir) / "ece.png"
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot ECE vs k for ResNet18 variance models"
    )
    parser.add_argument(
        "eval_path",
        type=str,
        help="Path to ece.jsonl file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save plot (default: same directory as eval_path)",
    )
    args = parser.parse_args()

    output_dir = args.output_dir if args.output_dir else str(Path(args.eval_path).parent)
    plot_ece(args.eval_path, output_dir)


if __name__ == "__main__":
    main()
