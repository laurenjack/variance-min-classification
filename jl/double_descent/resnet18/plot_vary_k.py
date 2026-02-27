#!/usr/bin/env python3
"""Plot final-epoch metrics across varying width k values.

Usage:
    python -m jl.double_descent.resnet18.plot_vary_k ./output --min-k 18 --max-k 32 --output-dir ./data
"""

import argparse
import glob
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt


def load_metrics_for_k_range(metrics_dir: str, min_k: int, max_k: int) -> List[Dict]:
    """Load metrics from metrics_k*.jsonl files within the specified k range.

    Args:
        metrics_dir: Directory containing metrics_k*.jsonl files.
        min_k: Minimum k value (inclusive).
        max_k: Maximum k value (inclusive).

    Returns:
        List of metrics dictionaries for k values in [min_k, max_k].
    """
    path = Path(metrics_dir)
    pattern = str(path / "metrics_k*.jsonl")
    files = sorted(glob.glob(pattern))

    if not files:
        raise FileNotFoundError(f"No metrics_k*.jsonl files found in {path}")

    metrics = []
    for file_path in files:
        # Extract k value from filename (metrics_k18.jsonl -> 18)
        filename = Path(file_path).stem
        k_str = filename.replace("metrics_k", "")
        try:
            k = int(k_str)
        except ValueError:
            continue

        if min_k <= k <= max_k:
            with open(file_path, 'r') as f:
                for line in f:
                    if line.strip():
                        metrics.append(json.loads(line))

    if not metrics:
        raise ValueError(f"No metrics found for k in [{min_k}, {max_k}]")

    return metrics


def get_final_epoch_data(metrics: List[Dict]) -> Dict:
    """Extract metrics from the final epoch for each width k.

    Returns:
        Dict with keys: k_values, train_error, test_error, train_loss, test_loss
    """
    # Group by k, find max epoch per k
    by_k = {}
    for m in metrics:
        k = m['k']
        if k not in by_k:
            by_k[k] = []
        by_k[k].append(m)

    # Get final epoch for each k
    final_metrics = []
    for k, k_metrics in by_k.items():
        max_epoch = max(m['epoch'] for m in k_metrics)
        final = next(m for m in k_metrics if m['epoch'] == max_epoch)
        final_metrics.append(final)

    # Sort by k
    final_metrics.sort(key=lambda m: m['k'])

    return {
        'k_values': [m['k'] for m in final_metrics],
        'train_error': [m['train_error'] for m in final_metrics],
        'test_error': [m['test_error'] for m in final_metrics],
        'train_loss': [m['train_loss'] for m in final_metrics],
        'test_loss': [m['test_loss'] for m in final_metrics],
    }


def plot_vary_k(
    metrics_dir: str,
    min_k: int,
    max_k: int,
    output_dir: str,
    noise_level: float = 0.15,
) -> None:
    """Plot final-epoch error and loss vs k.

    Creates a single figure with 2 subplots:
    - Top: Train/Test error vs k
    - Bottom: Train/Test loss vs k

    Args:
        metrics_dir: Directory containing metrics_k*.jsonl files.
        min_k: Minimum k value (inclusive).
        max_k: Maximum k value (inclusive).
        output_dir: Directory to save plot.
        noise_level: Label noise fraction for title.
    """
    metrics = load_metrics_for_k_range(metrics_dir, min_k, max_k)
    data = get_final_epoch_data(metrics)

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), dpi=150, sharex=True)

    # Error plot
    ax1.plot(data['k_values'], data['test_error'], '-o', color='blue',
             lw=2, label='Test Error')
    ax1.plot(data['k_values'], data['train_error'], '--o', color='blue',
             lw=2, alpha=0.5, label='Train Error')
    ax1.set_ylabel('Error Rate')
    ax1.set_title(f'Double Descent: ResNet18 on CIFAR-10 ({int(noise_level*100)}% label noise)')
    ax1.set_ylim(bottom=0)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Loss plot
    ax2.plot(data['k_values'], data['test_loss'], '-o', color='red',
             lw=2, label='Test Loss')
    ax2.plot(data['k_values'], data['train_loss'], '--o', color='red',
             lw=2, alpha=0.5, label='Train Loss')
    ax2.set_xlabel('ResNet18 width parameter k')
    ax2.set_ylabel('Cross-Entropy Loss')
    ax2.set_ylim(bottom=0)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = Path(output_dir) / f'resnet18_vary_k_{min_k}_to_{max_k}.png'
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot ResNet18 double descent: final metrics vs k"
    )
    parser.add_argument(
        "metrics_dir",
        type=str,
        help="Directory containing metrics_k*.jsonl files"
    )
    parser.add_argument(
        "--min-k",
        type=int,
        required=True,
        help="Minimum k value (inclusive)"
    )
    parser.add_argument(
        "--max-k",
        type=int,
        required=True,
        help="Maximum k value (inclusive)"
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

    plot_vary_k(args.metrics_dir, args.min_k, args.max_k, args.output_dir, args.noise_level)


if __name__ == "__main__":
    main()
