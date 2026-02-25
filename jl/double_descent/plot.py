#!/usr/bin/env python3
"""Plotting utilities for Deep Double Descent experiments.

Usage:
    python -m jl.double_descent.plot ./output/metrics.jsonl --output-dir ./data
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np


def load_metrics(metrics_path: str) -> List[Dict]:
    """Load metrics from JSONL file."""
    metrics = []
    with open(metrics_path, 'r') as f:
        for line in f:
            if line.strip():
                metrics.append(json.loads(line))
    return metrics


def get_final_epoch_data(metrics: List[Dict]) -> Dict:
    """Extract metrics from the final epoch for each width k.

    Returns:
        Dict with keys: k_values, train_error, test_error, train_loss, test_loss
    """
    # Find max epoch
    max_epoch = max(m['epoch'] for m in metrics)

    # Filter to final epoch and sort by k
    final_metrics = [m for m in metrics if m['epoch'] == max_epoch]
    final_metrics.sort(key=lambda m: m['k'])

    return {
        'k_values': [m['k'] for m in final_metrics],
        'train_error': [m['train_error'] for m in final_metrics],
        'test_error': [m['test_error'] for m in final_metrics],
        'train_loss': [m['train_loss'] for m in final_metrics],
        'test_loss': [m['test_loss'] for m in final_metrics],
    }


def get_epoch_data(metrics: List[Dict]) -> Dict:
    """Extract metrics for all epochs and widths.

    Returns:
        Dict with arrays shaped [n_widths, n_epochs]
    """
    # Get unique k values and epochs
    k_values = sorted(set(m['k'] for m in metrics))
    epochs = sorted(set(m['epoch'] for m in metrics))

    n_widths = len(k_values)
    n_epochs = len(epochs)

    # Create arrays
    train_error = np.zeros((n_widths, n_epochs))
    test_error = np.zeros((n_widths, n_epochs))
    train_loss = np.zeros((n_widths, n_epochs))
    test_loss = np.zeros((n_widths, n_epochs))

    # Index mappings
    k_to_idx = {k: i for i, k in enumerate(k_values)}
    epoch_to_idx = {e: i for i, e in enumerate(epochs)}

    # Fill arrays
    for m in metrics:
        ki = k_to_idx[m['k']]
        ei = epoch_to_idx[m['epoch']]
        train_error[ki, ei] = m['train_error']
        test_error[ki, ei] = m['test_error']
        train_loss[ki, ei] = m['train_loss']
        test_loss[ki, ei] = m['test_loss']

    return {
        'k_values': k_values,
        'epochs': epochs,
        'train_error': train_error,
        'test_error': test_error,
        'train_loss': train_loss,
        'test_loss': test_loss,
    }


def plot_double_descent_error(
    metrics_path: str,
    output_dir: str,
    noise_level: float = 0.15,
) -> None:
    """Plot 1: Test/Train error vs k (final epoch) - reproduces Figure 1.

    Args:
        metrics_path: Path to metrics.jsonl file.
        output_dir: Directory to save plot.
        noise_level: Label noise fraction for title.
    """
    metrics = load_metrics(metrics_path)
    data = get_final_epoch_data(metrics)

    fig, ax = plt.subplots(1, 1, figsize=(10, 6), dpi=150)

    ax.plot(data['k_values'], data['test_error'], '-', color='blue',
            lw=2, label='Test Error')
    ax.plot(data['k_values'], data['train_error'], '--', color='blue',
            lw=2, alpha=0.5, label='Train Error')

    ax.set_xlabel('ResNet18 width parameter k')
    ax.set_ylabel('Error Rate')
    ax.set_title(f'Double Descent: ResNet18 on CIFAR-10 ({int(noise_level*100)}% label noise)')
    ax.set_ylim(bottom=0)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    output_path = Path(output_dir) / 'double_descent_error.png'
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Saved error plot to {output_path}")


def plot_double_descent_loss(
    metrics_path: str,
    output_dir: str,
    noise_level: float = 0.15,
) -> None:
    """Plot 2: Test/Train loss vs k (final epoch).

    Shows loss/error divergence at interpolation threshold.
    """
    metrics = load_metrics(metrics_path)
    data = get_final_epoch_data(metrics)

    fig, ax = plt.subplots(1, 1, figsize=(10, 6), dpi=150)

    ax.plot(data['k_values'], data['test_loss'], '-', color='red',
            lw=2, label='Test Loss')
    ax.plot(data['k_values'], data['train_loss'], '--', color='red',
            lw=2, alpha=0.5, label='Train Loss')

    ax.set_xlabel('ResNet18 width parameter k')
    ax.set_ylabel('Cross-Entropy Loss')
    ax.set_title(f'Loss vs Width: ResNet18 on CIFAR-10 ({int(noise_level*100)}% label noise)')
    ax.set_ylim(bottom=0)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    output_path = Path(output_dir) / 'double_descent_loss.png'
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Saved loss plot to {output_path}")


def plot_error_vs_epoch_heatmap(
    metrics_path: str,
    output_dir: str,
    noise_level: float = 0.15,
) -> None:
    """Plot 3: Test error vs (k, epoch) heatmap - reproduces Figure 2."""
    metrics = load_metrics(metrics_path)
    data = get_epoch_data(metrics)

    fig, ax = plt.subplots(1, 1, figsize=(12, 8), dpi=150)

    # Create heatmap
    im = ax.imshow(
        data['test_error'],
        aspect='auto',
        origin='lower',
        cmap='viridis',
    )

    # Labels
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Width k')

    # Set tick labels
    n_epochs = len(data['epochs'])
    n_widths = len(data['k_values'])

    # X-axis: show every 500th epoch
    x_step = max(1, n_epochs // 8)
    x_ticks = list(range(0, n_epochs, x_step))
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([str(data['epochs'][i]) for i in x_ticks])

    # Y-axis: show all widths if few, else subsample
    if n_widths <= 20:
        ax.set_yticks(range(n_widths))
        ax.set_yticklabels([str(k) for k in data['k_values']])
    else:
        y_step = max(1, n_widths // 10)
        y_ticks = list(range(0, n_widths, y_step))
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([str(data['k_values'][i]) for i in y_ticks])

    ax.set_title(f'Test Error vs (Width, Epoch): CIFAR-10 ({int(noise_level*100)}% noise)')

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Test Error')

    output_path = Path(output_dir) / 'double_descent_error_heatmap.png'
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Saved error heatmap to {output_path}")


def plot_loss_vs_epoch_heatmap(
    metrics_path: str,
    output_dir: str,
    noise_level: float = 0.15,
) -> None:
    """Plot 4: Test loss vs (k, epoch) heatmap."""
    metrics = load_metrics(metrics_path)
    data = get_epoch_data(metrics)

    fig, ax = plt.subplots(1, 1, figsize=(12, 8), dpi=150)

    # Use log scale for loss heatmap
    loss_data = np.log10(data['test_loss'] + 1e-10)

    im = ax.imshow(
        loss_data,
        aspect='auto',
        origin='lower',
        cmap='plasma',
    )

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Width k')

    # Set tick labels
    n_epochs = len(data['epochs'])
    n_widths = len(data['k_values'])

    x_step = max(1, n_epochs // 8)
    x_ticks = list(range(0, n_epochs, x_step))
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([str(data['epochs'][i]) for i in x_ticks])

    if n_widths <= 20:
        ax.set_yticks(range(n_widths))
        ax.set_yticklabels([str(k) for k in data['k_values']])
    else:
        y_step = max(1, n_widths // 10)
        y_ticks = list(range(0, n_widths, y_step))
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([str(data['k_values'][i]) for i in y_ticks])

    ax.set_title(f'Test Loss vs (Width, Epoch): CIFAR-10 ({int(noise_level*100)}% noise)')

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Log10(Test Loss)')

    output_path = Path(output_dir) / 'double_descent_loss_heatmap.png'
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Saved loss heatmap to {output_path}")


def plot_error_loss_comparison(
    metrics_path: str,
    output_dir: str,
    noise_level: float = 0.15,
) -> None:
    """Plot 5: Combined error and loss plot showing they are at odds."""
    metrics = load_metrics(metrics_path)
    data = get_final_epoch_data(metrics)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), dpi=150, sharex=True)

    # Error plot
    ax1.plot(data['k_values'], data['test_error'], '-', color='blue',
             lw=2, label='Test Error')
    ax1.plot(data['k_values'], data['train_error'], '--', color='blue',
             lw=2, alpha=0.5, label='Train Error')
    ax1.set_ylabel('Error Rate')
    ax1.set_title(f'Double Descent: Error vs Loss are at odds (CIFAR-10, {int(noise_level*100)}% noise)')
    ax1.set_ylim(bottom=0)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Loss plot
    ax2.plot(data['k_values'], data['test_loss'], '-', color='red',
             lw=2, label='Test Loss')
    ax2.plot(data['k_values'], data['train_loss'], '--', color='red',
             lw=2, alpha=0.5, label='Train Loss')
    ax2.set_xlabel('ResNet18 width parameter k')
    ax2.set_ylabel('Cross-Entropy Loss')
    ax2.set_ylim(bottom=0)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = Path(output_dir) / 'double_descent_error_vs_loss.png'
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Saved error vs loss comparison to {output_path}")


def plot_all(
    metrics_path: str,
    output_dir: str,
    noise_level: float = 0.15,
) -> None:
    """Generate all plots."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    plot_double_descent_error(metrics_path, output_dir, noise_level)
    plot_double_descent_loss(metrics_path, output_dir, noise_level)
    plot_error_loss_comparison(metrics_path, output_dir, noise_level)
    plot_error_vs_epoch_heatmap(metrics_path, output_dir, noise_level)
    plot_loss_vs_epoch_heatmap(metrics_path, output_dir, noise_level)

    print(f"\nAll plots saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot Deep Double Descent results"
    )
    parser.add_argument(
        "metrics_path",
        type=str,
        help="Path to metrics.jsonl file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data",
        help="Directory to save plots"
    )
    parser.add_argument(
        "--noise-level",
        type=float,
        default=0.15,
        help="Label noise fraction (for plot titles)"
    )
    args = parser.parse_args()

    plot_all(args.metrics_path, args.output_dir, args.noise_level)


if __name__ == "__main__":
    main()
