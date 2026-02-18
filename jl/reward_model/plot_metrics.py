#!/usr/bin/env python3
"""Plot training metrics from metrics.jsonl file.

Usage:
    python -m jl.reward_model.plot_metrics metrics.jsonl [output.png]
    python -m jl.reward_model.plot_metrics metrics.jsonl --output-dir ./data
"""

import argparse
import json
import os
import sys
from datetime import datetime

import matplotlib.pyplot as plt


def load_metrics(metrics_path: str) -> list[dict]:
    """Load metrics from a JSONL file."""
    metrics = []
    if not os.path.exists(metrics_path):
        return metrics
    with open(metrics_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                metrics.append(json.loads(line))
    return metrics


def plot_metrics(metrics: list[dict], output_path: str, val_metrics: list[dict] = None, title_suffix: str = ""):
    """Generate a 2x2 figure with training and validation metrics.

    Top row: Training loss and accuracy over steps
    Bottom row: Validation loss and accuracy over epochs
    """
    if not metrics:
        print("No metrics to plot")
        return

    steps = [m["step"] for m in metrics]
    losses = [m["loss"] for m in metrics]
    accuracies = [m["accuracy"] for m in metrics]

    # Extract learning rate for title (use max LR = target after warmup)
    lrs = [m.get("lr", 0) for m in metrics]
    lr = max(lrs) if lrs else "unknown"
    if isinstance(lr, float):
        lr_str = f"{lr:.0e}"
    else:
        lr_str = str(lr)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"Training Metrics (LR={lr_str}){title_suffix}", fontsize=14)

    # Top left: Training Loss
    axes[0, 0].plot(steps, losses, "b-", linewidth=1.5)
    axes[0, 0].set_xlabel("Step")
    axes[0, 0].set_ylabel("Training Loss")
    axes[0, 0].set_title("Training Loss")
    axes[0, 0].grid(True, alpha=0.3)

    # Top right: Training Accuracy
    axes[0, 1].plot(steps, accuracies, "g-", linewidth=1.5)
    axes[0, 1].set_xlabel("Step")
    axes[0, 1].set_ylabel("Training Accuracy (Win Rate)")
    axes[0, 1].set_title("Training Accuracy")
    axes[0, 1].grid(True, alpha=0.3)

    # Bottom row: Validation metrics (if available)
    if val_metrics:
        epochs = [m["epoch"] for m in val_metrics]
        val_losses = [m["val_loss"] for m in val_metrics]
        val_accuracies = [m["val_accuracy"] for m in val_metrics]

        # Bottom left: Validation Loss
        axes[1, 0].plot(epochs, val_losses, "b-o", linewidth=1.5, markersize=6)
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("Validation Loss")
        axes[1, 0].set_title("Validation Loss")
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_xticks(epochs)

        # Bottom right: Validation Accuracy
        axes[1, 1].plot(epochs, val_accuracies, "g-o", linewidth=1.5, markersize=6)
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("Validation Accuracy (Win Rate)")
        axes[1, 1].set_title("Validation Accuracy")
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_xticks(epochs)
    else:
        # No validation metrics - show placeholder text
        axes[1, 0].text(0.5, 0.5, "No validation metrics available",
                        ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 0].set_title("Validation Loss")
        axes[1, 1].text(0.5, 0.5, "No validation metrics available",
                        ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title("Validation Accuracy")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Plot saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot training metrics")
    parser.add_argument(
        "metrics_path",
        type=str,
        help="Path to metrics.jsonl file"
    )
    parser.add_argument(
        "output_path",
        type=str,
        nargs="?",
        default=None,
        help="Output path for PNG (default: auto-generated in same directory)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for PNG (filename auto-generated with timestamp)"
    )
    parser.add_argument(
        "--val-metrics",
        type=str,
        default=None,
        help="Path to val_metrics.jsonl file (default: auto-detected in same directory)"
    )
    args = parser.parse_args()

    if not os.path.exists(args.metrics_path):
        print(f"Error: metrics file not found: {args.metrics_path}")
        sys.exit(1)

    metrics = load_metrics(args.metrics_path)
    if not metrics:
        print("Error: no metrics found in file")
        sys.exit(1)

    # Load validation metrics (auto-detect if not specified)
    if args.val_metrics:
        val_metrics_path = args.val_metrics
    else:
        # Auto-detect: look for val_metrics.jsonl in same directory
        metrics_dir = os.path.dirname(args.metrics_path)
        val_metrics_path = os.path.join(metrics_dir, "val_metrics.jsonl")

    val_metrics = load_metrics(val_metrics_path)
    if val_metrics:
        print(f"Loaded {len(val_metrics)} validation metrics from {val_metrics_path}")
    else:
        print(f"No validation metrics found at {val_metrics_path}")

    # Determine output path
    if args.output_path:
        output_path = args.output_path
    elif args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        lrs = [m.get("lr", 0) for m in metrics]
        lr = max(lrs) if lrs else 0
        lr_str = f"{lr:.0e}".replace("-", "m").replace("+", "p") if isinstance(lr, float) else "unknown"
        output_path = os.path.join(args.output_dir, f"metrics_lr{lr_str}_{timestamp}.png")
    else:
        # Default: same directory as metrics file
        base = os.path.splitext(args.metrics_path)[0]
        output_path = f"{base}.png"

    plot_metrics(metrics, output_path, val_metrics=val_metrics)


if __name__ == "__main__":
    main()
