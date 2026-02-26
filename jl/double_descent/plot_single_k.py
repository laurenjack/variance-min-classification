#!/usr/bin/env python3
"""Plot training curves for a single k value over epochs.

Usage:
    python -m jl.double_descent.plot_single_k ./data/lambda_output --k 12 --output-dir ./data
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt


def load_metrics_for_k(metrics_path: str, k: int) -> List[Dict]:
    """Load metrics for a specific k value.

    Args:
        metrics_path: Path to directory containing metrics_k*.jsonl files,
                     or path to a specific metrics_k{k}.jsonl file.
        k: The width parameter to load.

    Returns:
        List of metrics dictionaries sorted by epoch.
    """
    path = Path(metrics_path)

    if path.is_dir():
        file_path = path / f"metrics_k{k}.jsonl"
    else:
        file_path = path

    if not file_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {file_path}")

    metrics = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                m = json.loads(line)
                if m.get('k') == k:
                    metrics.append(m)

    metrics.sort(key=lambda m: m['epoch'])
    return metrics


def plot_single_k(
    metrics_path: str,
    k: int,
    output_dir: str,
) -> None:
    """Plot training curves for a single k value.

    Args:
        metrics_path: Path to metrics directory or file.
        k: Width parameter to plot.
        output_dir: Directory to save plots.
    """
    metrics = load_metrics_for_k(metrics_path, k)

    if not metrics:
        print(f"No metrics found for k={k}")
        return

    epochs = [m['epoch'] for m in metrics]
    train_error = [m['train_error'] for m in metrics]
    test_error = [m['test_error'] for m in metrics]
    train_loss = [m['train_loss'] for m in metrics]
    test_loss = [m['test_loss'] for m in metrics]

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Plot 1: Error over epochs
    fig, ax = plt.subplots(1, 1, figsize=(10, 6), dpi=150)
    ax.plot(epochs, test_error, '-', color='blue', lw=2, label='Test Error')
    ax.plot(epochs, train_error, '--', color='blue', lw=2, alpha=0.5, label='Train Error')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Error Rate')
    ax.set_title(f'Error vs Epoch (k={k})')
    ax.set_ylim(bottom=0)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    output_path = Path(output_dir) / f'k{k}_error_vs_epoch.png'
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

    # Plot 2: Loss over epochs
    fig, ax = plt.subplots(1, 1, figsize=(10, 6), dpi=150)
    ax.plot(epochs, test_loss, '-', color='red', lw=2, label='Test Loss')
    ax.plot(epochs, train_loss, '--', color='red', lw=2, alpha=0.5, label='Train Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Cross-Entropy Loss')
    ax.set_title(f'Loss vs Epoch (k={k})')
    ax.set_ylim(bottom=0)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    output_path = Path(output_dir) / f'k{k}_loss_vs_epoch.png'
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

    # Plot 3: Combined - error and loss on same plot (dual y-axis)
    fig, ax1 = plt.subplots(1, 1, figsize=(10, 6), dpi=150)

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Error Rate', color='blue')
    ax1.plot(epochs, test_error, '-', color='blue', lw=2, label='Test Error')
    ax1.plot(epochs, train_error, '--', color='blue', lw=2, alpha=0.5, label='Train Error')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.set_ylim(bottom=0)

    ax2 = ax1.twinx()
    ax2.set_ylabel('Cross-Entropy Loss', color='red')
    ax2.plot(epochs, test_loss, '-', color='red', lw=2, label='Test Loss')
    ax2.plot(epochs, train_loss, '--', color='red', lw=2, alpha=0.5, label='Train Loss')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.set_ylim(bottom=0)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    ax1.set_title(f'Training Curves (k={k})')
    ax1.grid(True, alpha=0.3)

    output_path = Path(output_dir) / f'k{k}_combined.png'
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot training curves for a single k value"
    )
    parser.add_argument(
        "metrics_path",
        type=str,
        help="Path to metrics directory or metrics_k{k}.jsonl file"
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
        help="Directory to save plots"
    )
    args = parser.parse_args()

    plot_single_k(args.metrics_path, args.k, args.output_dir)


if __name__ == "__main__":
    main()
