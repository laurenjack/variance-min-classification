#!/usr/bin/env python3
"""Plotting utilities for Transformer Double Descent experiments.

Usage:
    # All plots for all d_model values (final metrics: loss, accuracy, BLEU)
    python -m jl.transformer_dd.plot ./output --output-dir ./data

    # Step-wise training curves for a specific d_model
    python -m jl.transformer_dd.plot ./output/metrics_d64.jsonl --output-dir ./data
    # OR
    python -m jl.transformer_dd.plot ./output --d-model 64 --output-dir ./data
"""

import argparse
import glob
import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np


def load_metrics(metrics_path: str) -> List[Dict]:
    """Load metrics from JSONL file(s).

    Args:
        metrics_path: Either a path to a single .jsonl file, or a directory
                     containing metrics_d*.jsonl files.

    Returns:
        List of metrics dictionaries.
    """
    path = Path(metrics_path)
    metrics = []

    if path.is_dir():
        # Load all metrics_d*.jsonl files from directory
        pattern = str(path / "metrics_d*.jsonl")
        files = sorted(glob.glob(pattern))
        if not files:
            raise FileNotFoundError(f"No metrics_d*.jsonl files found in {path}")
        for file_path in files:
            with open(file_path, 'r') as f:
                for line in f:
                    if line.strip():
                        metrics.append(json.loads(line))
    else:
        # Load single file
        with open(path, 'r') as f:
            for line in f:
                if line.strip():
                    metrics.append(json.loads(line))

    return metrics


def get_final_metrics(metrics: List[Dict]) -> Dict:
    """Extract final metrics for each d_model value.

    Returns:
        Dict with keys: d_model_values, train_loss, test_loss, train_acc, test_acc,
                       train_bleu, test_bleu
    """
    # Group by d_model, take last entry with test_bleu (final evaluation)
    by_d_model = {}
    for m in metrics:
        d_model = m['d_model']
        # Only consider entries with test_bleu (final evaluation)
        if 'test_bleu' in m:
            by_d_model[d_model] = m

    # Sort by d_model
    d_model_values = sorted(by_d_model.keys())

    return {
        'd_model_values': d_model_values,
        'train_loss': [by_d_model[d].get('train_loss', 0) for d in d_model_values],
        'test_loss': [by_d_model[d]['test_loss'] for d in d_model_values],
        'train_acc': [by_d_model[d].get('train_acc', 0) for d in d_model_values],
        'test_acc': [by_d_model[d]['test_acc'] for d in d_model_values],
        'train_bleu': [by_d_model[d]['train_bleu'] for d in d_model_values],
        'test_bleu': [by_d_model[d]['test_bleu'] for d in d_model_values],
        'valid_loss': [by_d_model[d]['valid_loss'] for d in d_model_values],
        'valid_acc': [by_d_model[d]['valid_acc'] for d in d_model_values],
    }


def get_stepwise_metrics(metrics: List[Dict], d_model: int) -> Dict:
    """Extract step-wise metrics for a specific d_model.

    Returns:
        Dict with keys: steps, train_loss, valid_loss, train_acc, valid_acc, lr
    """
    # Filter to this d_model
    filtered = [m for m in metrics if m['d_model'] == d_model]

    # Sort by step
    filtered.sort(key=lambda m: m['step'])

    # Extract metrics (only entries with valid_loss, which are at eval intervals)
    with_valid = [m for m in filtered if 'valid_loss' in m]

    return {
        'steps': [m['step'] for m in with_valid],
        'train_loss': [m['train_loss'] for m in with_valid],
        'valid_loss': [m['valid_loss'] for m in with_valid],
        'train_acc': [m['train_acc'] for m in with_valid],
        'valid_acc': [m['valid_acc'] for m in with_valid],
        'lr': [m['lr'] for m in with_valid],
    }


def plot_loss_vs_dmodel(
    metrics_path: str,
    output_dir: str,
) -> None:
    """Plot 1: Loss vs d_model (final metrics).

    Args:
        metrics_path: Path to metrics directory or file.
        output_dir: Directory to save plot.
    """
    metrics = load_metrics(metrics_path)
    data = get_final_metrics(metrics)

    fig, ax = plt.subplots(1, 1, figsize=(10, 6), dpi=150)

    ax.plot(data['d_model_values'], data['test_loss'], '-', color='blue',
            lw=2, label='Test Loss')
    ax.plot(data['d_model_values'], data['valid_loss'], '--', color='blue',
            lw=2, alpha=0.5, label='Valid Loss')

    ax.set_xlabel('Transformer embedding dimension (d_model)')
    ax.set_ylabel('Cross-Entropy Loss')
    ax.set_title('Transformer Double Descent: Loss vs Model Width')
    ax.set_ylim(bottom=0)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    output_path = Path(output_dir) / 'transformer_dd_loss.png'
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Saved loss plot to {output_path}")


def plot_accuracy_vs_dmodel(
    metrics_path: str,
    output_dir: str,
) -> None:
    """Plot 2: Accuracy vs d_model (final metrics).

    Args:
        metrics_path: Path to metrics directory or file.
        output_dir: Directory to save plot.
    """
    metrics = load_metrics(metrics_path)
    data = get_final_metrics(metrics)

    fig, ax = plt.subplots(1, 1, figsize=(10, 6), dpi=150)

    ax.plot(data['d_model_values'], data['test_acc'], '-', color='green',
            lw=2, label='Test Accuracy')
    ax.plot(data['d_model_values'], data['valid_acc'], '--', color='green',
            lw=2, alpha=0.5, label='Valid Accuracy')

    ax.set_xlabel('Transformer embedding dimension (d_model)')
    ax.set_ylabel('Token-level Accuracy')
    ax.set_title('Transformer Double Descent: Accuracy vs Model Width')
    ax.set_ylim(0, 1)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    output_path = Path(output_dir) / 'transformer_dd_accuracy.png'
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Saved accuracy plot to {output_path}")


def plot_bleu_vs_dmodel(
    metrics_path: str,
    output_dir: str,
) -> None:
    """Plot 3: BLEU vs d_model (final metrics).

    Args:
        metrics_path: Path to metrics directory or file.
        output_dir: Directory to save plot.
    """
    metrics = load_metrics(metrics_path)
    data = get_final_metrics(metrics)

    fig, ax = plt.subplots(1, 1, figsize=(10, 6), dpi=150)

    ax.plot(data['d_model_values'], data['test_bleu'], '-', color='purple',
            lw=2, label='Test BLEU')
    ax.plot(data['d_model_values'], data['train_bleu'], '--', color='purple',
            lw=2, alpha=0.5, label='Train BLEU')

    ax.set_xlabel('Transformer embedding dimension (d_model)')
    ax.set_ylabel('BLEU Score')
    ax.set_title('Transformer Double Descent: BLEU vs Model Width')
    ax.set_ylim(bottom=0)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    output_path = Path(output_dir) / 'transformer_dd_bleu.png'
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Saved BLEU plot to {output_path}")


def plot_training_curves(
    metrics_path: str,
    output_dir: str,
    d_model: Optional[int] = None,
) -> None:
    """Plot step-wise training curves for a specific d_model.

    Single figure with 2 subplots:
    - Top: Train loss & Valid loss vs Step
    - Bottom: Train accuracy & Valid accuracy vs Step

    Args:
        metrics_path: Path to metrics directory or single file.
        output_dir: Directory to save plot.
        d_model: Specific d_model to plot (inferred from file if not provided).
    """
    metrics = load_metrics(metrics_path)

    # Infer d_model if not provided
    if d_model is None:
        d_models = set(m['d_model'] for m in metrics)
        if len(d_models) == 1:
            d_model = list(d_models)[0]
        else:
            raise ValueError(
                f"Multiple d_model values found: {sorted(d_models)}. "
                "Please specify --d-model."
            )

    data = get_stepwise_metrics(metrics, d_model)

    if not data['steps']:
        print(f"No step-wise metrics found for d_model={d_model}")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), dpi=150, sharex=True)

    # Loss plot
    ax1.plot(data['steps'], data['train_loss'], '-', color='blue',
             lw=2, label='Train Loss')
    ax1.plot(data['steps'], data['valid_loss'], '-', color='red',
             lw=2, label='Valid Loss')
    ax1.set_ylabel('Cross-Entropy Loss')
    ax1.set_title(f'Training Curves: d_model={d_model}')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Accuracy plot
    ax2.plot(data['steps'], data['train_acc'], '-', color='blue',
             lw=2, label='Train Accuracy')
    ax2.plot(data['steps'], data['valid_acc'], '-', color='red',
             lw=2, label='Valid Accuracy')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Token-level Accuracy')
    ax2.set_ylim(0, 1)
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = Path(output_dir) / f'transformer_dd_training_d{d_model}.png'
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Saved training curves to {output_path}")


def plot_all(
    metrics_path: str,
    output_dir: str,
) -> None:
    """Generate all final metrics plots (loss, accuracy, BLEU vs d_model)."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    plot_loss_vs_dmodel(metrics_path, output_dir)
    plot_accuracy_vs_dmodel(metrics_path, output_dir)
    plot_bleu_vs_dmodel(metrics_path, output_dir)

    print(f"\nAll plots saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot Transformer Double Descent results"
    )
    parser.add_argument(
        "metrics_path",
        type=str,
        help="Path to metrics directory (containing metrics_d*.jsonl) or single .jsonl file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data",
        help="Directory to save plots"
    )
    parser.add_argument(
        "--d-model",
        type=int,
        default=None,
        help="Specific d_model value for step-wise training curves"
    )
    args = parser.parse_args()

    path = Path(args.metrics_path)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Determine plot type based on input
    if path.is_file() or args.d_model is not None:
        # Single d_model: plot training curves
        plot_training_curves(args.metrics_path, args.output_dir, args.d_model)
    else:
        # Directory: plot all final metrics
        plot_all(args.metrics_path, args.output_dir)


if __name__ == "__main__":
    main()
