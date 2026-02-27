#!/usr/bin/env python3
"""Plot epoch-wise training curves for a single width k.

Usage:
    python -m jl.double_descent.resnet18.plot_single_k ./output --k 18 --output-dir ./data
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt


def load_metrics_for_k(metrics_dir: str, k: int) -> List[Dict]:
    """Load metrics from metrics_k{k}.jsonl file.

    Args:
        metrics_dir: Directory containing metrics_k*.jsonl files.
        k: Width parameter value.

    Returns:
        List of metrics dictionaries for the specified k.
    """
    path = Path(metrics_dir) / f"metrics_k{k}.jsonl"

    if not path.exists():
        raise FileNotFoundError(f"Metrics file not found: {path}")

    metrics = []
    with open(path, 'r') as f:
        for line in f:
            if line.strip():
                metrics.append(json.loads(line))

    if not metrics:
        raise ValueError(f"No metrics found in {path}")

    return metrics


def get_epochwise_data(metrics: List[Dict]) -> Dict:
    """Extract epoch-wise metrics.

    Returns:
        Dict with keys: epochs, train_error, test_error, train_loss, test_loss
    """
    # Sort by epoch
    metrics = sorted(metrics, key=lambda m: m['epoch'])

    return {
        'epochs': [m['epoch'] for m in metrics],
        'train_error': [m['train_error'] for m in metrics],
        'test_error': [m['test_error'] for m in metrics],
        'train_loss': [m['train_loss'] for m in metrics],
        'test_loss': [m['test_loss'] for m in metrics],
    }


def plot_single_k(
    metrics_dir: str,
    k: int,
    output_dir: str,
    noise_level: float = 0.15,
) -> None:
    """Plot epoch-wise error and loss for a single k.

    Creates a single figure with 2 subplots:
    - Top: Train/Test error vs epoch
    - Bottom: Train/Test loss vs epoch

    Args:
        metrics_dir: Directory containing metrics_k*.jsonl files.
        k: Width parameter value.
        output_dir: Directory to save plot.
        noise_level: Label noise fraction for title.
    """
    metrics = load_metrics_for_k(metrics_dir, k)
    data = get_epochwise_data(metrics)

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), dpi=150, sharex=True)

    # Error plot
    ax1.plot(data['epochs'], data['test_error'], '-', color='blue',
             lw=2, label='Test Error')
    ax1.plot(data['epochs'], data['train_error'], '--', color='blue',
             lw=2, alpha=0.5, label='Train Error')
    ax1.set_ylabel('Error Rate')
    ax1.set_title(f'ResNet18 Training (k={k}, {int(noise_level*100)}% label noise)')
    ax1.set_ylim(bottom=0)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Loss plot
    ax2.plot(data['epochs'], data['test_loss'], '-', color='red',
             lw=2, label='Test Loss')
    ax2.plot(data['epochs'], data['train_loss'], '--', color='red',
             lw=2, alpha=0.5, label='Train Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Cross-Entropy Loss')
    ax2.set_ylim(bottom=0)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = Path(output_dir) / f'resnet18_k{k}_training.png'
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot ResNet18 training curves for a single k"
    )
    parser.add_argument(
        "metrics_dir",
        type=str,
        help="Directory containing metrics_k*.jsonl files"
    )
    parser.add_argument(
        "--k",
        type=int,
        required=True,
        help="Width parameter k to plot"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data",
        help="Directory to save plot"
    )
    parser.add_argument(
        "--noise-level",
        type=float,
        default=0.15,
        help="Label noise fraction (for plot title)"
    )
    args = parser.parse_args()

    plot_single_k(args.metrics_dir, args.k, args.output_dir, args.noise_level)


if __name__ == "__main__":
    main()
