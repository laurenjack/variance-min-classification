#!/usr/bin/env python3
"""Plot step-wise training curves for a single d_model and sample size.

Usage:
    python -m jl.double_descent.transformer.plot_single_d_model ./output --d-model 128 --samples 18k --output-dir ./data
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt


def load_metrics_for_d_model(metrics_dir: str, d_model: int, samples_k: str) -> List[Dict]:
    """Load metrics from metrics_d{d_model}_{samples_k}.jsonl file.

    Args:
        metrics_dir: Directory containing metrics_d*_*k.jsonl files.
        d_model: Model dimension value.
        samples_k: Sample size string (e.g., '4k' or '18k').

    Returns:
        List of metrics dictionaries for the specified d_model and sample size.
    """
    path = Path(metrics_dir) / f"metrics_d{d_model}_{samples_k}.jsonl"

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
    Excludes final BLEU entries which have train_loss=0.

    Returns:
        Dict with keys: steps, train_loss, valid_loss, train_acc, valid_acc
    """
    # Filter to entries with valid_loss (evaluation intervals)
    # Exclude final BLEU entries (which have test_bleu and train_loss=0)
    with_valid = [m for m in metrics if 'valid_loss' in m and 'test_bleu' not in m]

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
    samples_k: str,
    output_dir: str,
) -> None:
    """Plot step-wise loss and accuracy for a single d_model and sample size.

    Creates a single figure with 2 subplots:
    - Top: Train/Valid loss vs step
    - Bottom: Train/Valid accuracy vs step

    Args:
        metrics_dir: Directory containing metrics_d*_*k.jsonl files.
        d_model: Model dimension value.
        samples_k: Sample size string (e.g., '4k' or '18k').
        output_dir: Directory to save plot.
    """
    metrics = load_metrics_for_d_model(metrics_dir, d_model, samples_k)
    data = get_stepwise_data(metrics)

    if not data['steps']:
        raise ValueError(f"No step-wise metrics found for d_model={d_model}, samples={samples_k}")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), dpi=150, sharex=True)

    # Loss plot
    ax1.plot(data['steps'], data['valid_loss'], '-', color='blue',
             lw=2, label='Valid Loss')
    ax1.plot(data['steps'], data['train_loss'], '--', color='blue',
             lw=2, alpha=0.5, label='Train Loss')
    ax1.set_ylabel('Cross-Entropy Loss')
    ax1.set_title(f'Transformer Training (d_model={d_model}, {samples_k.upper()} samples)')
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
    output_path = Path(output_dir) / f'transformer_d{d_model}_{samples_k}_training.png'
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot Transformer training curves for a single d_model and sample size"
    )
    parser.add_argument(
        "metrics_dir",
        type=str,
        help="Directory containing metrics_d*_*k.jsonl files"
    )
    parser.add_argument(
        "--d-model",
        type=int,
        required=True,
        help="Model dimension d_model to plot"
    )
    parser.add_argument(
        "--samples",
        type=str,
        required=True,
        choices=['4k', '18k'],
        help="Sample size (4k or 18k)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data",
        help="Directory to save plot"
    )
    args = parser.parse_args()

    plot_single_d_model(args.metrics_dir, args.d_model, args.samples, args.output_dir)


if __name__ == "__main__":
    main()
