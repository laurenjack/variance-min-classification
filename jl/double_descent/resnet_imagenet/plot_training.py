#!/usr/bin/env python3
"""Plot epoch-wise training curves for ResNet-50 on ImageNet.

Two subplots, four lines:
  Top: train/test error vs epoch
  Bottom: train/test loss vs epoch

Usage:
    python -m jl.double_descent.resnet_imagenet.plot_training \
        ./output/resnet_imagenet/04-18-1010/metrics.jsonl
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt


def load_metrics(metrics_path: str) -> List[Dict]:
    metrics = []
    with open(metrics_path, "r") as f:
        for line in f:
            if line.strip():
                metrics.append(json.loads(line))
    if not metrics:
        raise ValueError(f"No metrics found in {metrics_path}")
    return sorted(metrics, key=lambda m: m["epoch"])


def plot_training(metrics_path: str, output_dir: str) -> None:
    metrics = load_metrics(metrics_path)

    epochs = [m["epoch"] for m in metrics]
    train_error = [m["train_error"] for m in metrics]
    test_error = [m["test_error"] for m in metrics]
    train_loss = [m["train_loss"] for m in metrics]
    test_loss = [m["test_loss"] for m in metrics]

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), dpi=150, sharex=True)

    ax1.plot(epochs, test_error, "-", color="blue", lw=2, label="Test Error")
    ax1.plot(epochs, train_error, "--", color="blue", lw=2, alpha=0.5,
             label="Train Error")
    ax1.set_ylabel("Error Rate")
    ax1.set_title("ResNet-50 ImageNet Training (He et al. 2015 recipe)")
    ax1.set_ylim(bottom=0)
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, test_loss, "-", color="red", lw=2, label="Test Loss")
    ax2.plot(epochs, train_loss, "--", color="red", lw=2, alpha=0.5,
             label="Train Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Cross-Entropy Loss")
    ax2.set_ylim(bottom=0)
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = Path(output_dir) / "resnet50_imagenet_training.png"
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot ResNet-50 ImageNet training curves"
    )
    parser.add_argument(
        "metrics_path", type=str, help="Path to metrics.jsonl",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Directory to save plot (default: metrics file's directory)",
    )
    args = parser.parse_args()

    output_dir = (
        args.output_dir if args.output_dir
        else str(Path(args.metrics_path).parent)
    )
    plot_training(args.metrics_path, output_dir)


if __name__ == "__main__":
    main()
