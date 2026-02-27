#!/usr/bin/env python3
"""Plot step-wise training curves for a single d_model.

Usage:
    python -m jl.double_descent.transformer.plot_single_d_model ./output --d-model 128 --output-dir ./data
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt


def load_metrics_for_d_model(metrics_dir: str, d_model: int) -> List[Dict]:
    """Load metrics from metrics_d{d_model}.jsonl file.

    Args:
        metrics_dir: Directory containing metrics_d*.jsonl files.
        d_model: Model dimension value.

    Returns:
        List of metrics dictionaries for the specified d_model.
    """
    path = Path(metrics_dir) / f"metrics_d{d_model}.jsonl"

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


def get_stepwise_data(metrics: List[Dict]) -> Dict:
    """Extract step-wise metrics.

    Only includes entries with valid_loss (evaluation intervals).

    Returns:
        Dict with keys: steps, train_loss, valid_loss, train_acc, valid_acc
    """
    # Filter to entries with valid_loss (evaluation intervals)
    with_valid = [m for m in metrics if 'valid_loss' in m]

    # Sort by step
    with_valid = sorted(with_valid, key=lambda m: m['step'])

    return {
        'steps': [m['step'] for m in with_valid],
        'train_loss': [m['train_loss'] for m in with_valid],
        'valid_loss': [m['valid_loss'] for m in with_valid],
        'train_acc': [m['train_acc'] for m in with_valid],
        'valid_acc': [m['valid_acc'] for m in with_valid],
    }


def plot_single_d_model(
    metrics_dir: str,
    d_model: int,
    output_dir: str,
) -> None:
    """Plot step-wise loss and accuracy for a single d_model.

    Creates a single figure with 2 subplots:
    - Top: Train/Valid loss vs step
    - Bottom: Train/Valid accuracy vs step

    Args:
        metrics_dir: Directory containing metrics_d*.jsonl files.
        d_model: Model dimension value.
        output_dir: Directory to save plot.
    """
    metrics = load_metrics_for_d_model(metrics_dir, d_model)
    data = get_stepwise_data(metrics)

    if not data['steps']:
        raise ValueError(f"No step-wise metrics found for d_model={d_model}")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), dpi=150, sharex=True)

    # Loss plot
    ax1.plot(data['steps'], data['valid_loss'], '-', color='blue',
             lw=2, label='Valid Loss')
    ax1.plot(data['steps'], data['train_loss'], '--', color='blue',
             lw=2, alpha=0.5, label='Train Loss')
    ax1.set_ylabel('Cross-Entropy Loss')
    ax1.set_title(f'Transformer Training (d_model={d_model})')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Accuracy plot
    ax2.plot(data['steps'], data['valid_acc'], '-', color='green',
             lw=2, label='Valid Accuracy')
    ax2.plot(data['steps'], data['train_acc'], '--', color='green',
             lw=2, alpha=0.5, label='Train Accuracy')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Token-level Accuracy')
    ax2.set_ylim(0, 1)
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = Path(output_dir) / f'transformer_d{d_model}_training.png'
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot Transformer training curves for a single d_model"
    )
    parser.add_argument(
        "metrics_dir",
        type=str,
        help="Directory containing metrics_d*.jsonl files"
    )
    parser.add_argument(
        "--d-model",
        type=int,
        required=True,
        help="Model dimension d_model to plot"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data",
        help="Directory to save plot"
    )
    args = parser.parse_args()

    plot_single_d_model(args.metrics_dir, args.d_model, args.output_dir)


if __name__ == "__main__":
    main()
