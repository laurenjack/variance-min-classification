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
    with open(metrics_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                metrics.append(json.loads(line))
    return metrics


def plot_metrics(metrics: list[dict], output_path: str, title_suffix: str = ""):
    """Generate a two-subplot figure with loss and accuracy over steps."""
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

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Plot loss
    ax1.plot(steps, losses, "b-", linewidth=1.5)
    ax1.set_ylabel("Training Loss")
    ax1.set_title(f"Training Metrics (LR={lr_str}){title_suffix}")
    ax1.grid(True, alpha=0.3)

    # Plot accuracy
    ax2.plot(steps, accuracies, "g-", linewidth=1.5)
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Training Accuracy (Win Rate)")
    ax2.grid(True, alpha=0.3)

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
    args = parser.parse_args()

    if not os.path.exists(args.metrics_path):
        print(f"Error: metrics file not found: {args.metrics_path}")
        sys.exit(1)

    metrics = load_metrics(args.metrics_path)
    if not metrics:
        print("Error: no metrics found in file")
        sys.exit(1)

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

    plot_metrics(metrics, output_path)


if __name__ == "__main__":
    main()
