#!/usr/bin/env python3
"""Plot final metrics across varying d_model values.

Usage:
    python -m jl.double_descent.transformer.plot_vary_d_model ./output --min-d-model 64 --max-d-model 512 --output-dir ./data
"""

import argparse
import glob
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt


def load_metrics_for_d_range(metrics_dir: str, min_d: int, max_d: int) -> List[Dict]:
    """Load metrics from metrics_d*.jsonl files within the specified d_model range.

    Args:
        metrics_dir: Directory containing metrics_d*.jsonl files.
        min_d: Minimum d_model value (inclusive).
        max_d: Maximum d_model value (inclusive).

    Returns:
        List of metrics dictionaries for d_model values in [min_d, max_d].
    """
    path = Path(metrics_dir)
    pattern = str(path / "metrics_d*.jsonl")
    files = sorted(glob.glob(pattern))

    if not files:
        raise FileNotFoundError(f"No metrics_d*.jsonl files found in {path}")

    metrics = []
    for file_path in files:
        # Extract d_model value from filename (metrics_d64.jsonl -> 64)
        filename = Path(file_path).stem
        d_str = filename.replace("metrics_d", "")
        try:
            d_model = int(d_str)
        except ValueError:
            continue

        if min_d <= d_model <= max_d:
            with open(file_path, 'r') as f:
                for line in f:
                    if line.strip():
                        metrics.append(json.loads(line))

    if not metrics:
        raise ValueError(f"No metrics found for d_model in [{min_d}, {max_d}]")

    return metrics


def get_final_metrics(metrics: List[Dict]) -> Dict:
    """Extract final metrics for each d_model value.

    Only considers entries with test_bleu (final evaluation).

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
    }


def plot_vary_d_model(
    metrics_dir: str,
    min_d: int,
    max_d: int,
    output_dir: str,
) -> None:
    """Plot final loss, accuracy, and BLEU vs d_model.

    Creates a single figure with 3 subplots:
    - Top: Train/Test loss vs d_model
    - Middle: Train/Test accuracy vs d_model
    - Bottom: Train/Test BLEU vs d_model

    Args:
        metrics_dir: Directory containing metrics_d*.jsonl files.
        min_d: Minimum d_model value (inclusive).
        max_d: Maximum d_model value (inclusive).
        output_dir: Directory to save plot.
    """
    metrics = load_metrics_for_d_range(metrics_dir, min_d, max_d)
    data = get_final_metrics(metrics)

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), dpi=150, sharex=True)

    # Loss plot
    ax1.plot(data['d_model_values'], data['test_loss'], '-o', color='blue',
             lw=2, label='Test Loss')
    ax1.plot(data['d_model_values'], data['train_loss'], '--o', color='blue',
             lw=2, alpha=0.5, label='Train Loss')
    ax1.set_ylabel('Cross-Entropy Loss')
    ax1.set_title('Transformer Double Descent: IWSLT14 de-en')
    ax1.set_ylim(bottom=0)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Accuracy plot
    ax2.plot(data['d_model_values'], data['test_acc'], '-o', color='green',
             lw=2, label='Test Accuracy')
    ax2.plot(data['d_model_values'], data['train_acc'], '--o', color='green',
             lw=2, alpha=0.5, label='Train Accuracy')
    ax2.set_ylabel('Token-level Accuracy')
    ax2.set_ylim(0, 1)
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)

    # BLEU plot
    ax3.plot(data['d_model_values'], data['test_bleu'], '-o', color='purple',
             lw=2, label='Test BLEU')
    ax3.plot(data['d_model_values'], data['train_bleu'], '--o', color='purple',
             lw=2, alpha=0.5, label='Train BLEU')
    ax3.set_xlabel('Transformer embedding dimension (d_model)')
    ax3.set_ylabel('BLEU Score')
    ax3.set_ylim(bottom=0)
    ax3.legend(loc='lower right')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = Path(output_dir) / f'transformer_vary_d_{min_d}_to_{max_d}.png'
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot Transformer double descent: final metrics vs d_model"
    )
    parser.add_argument(
        "metrics_dir",
        type=str,
        help="Directory containing metrics_d*.jsonl files"
    )
    parser.add_argument(
        "--min-d-model",
        type=int,
        required=True,
        help="Minimum d_model value (inclusive)"
    )
    parser.add_argument(
        "--max-d-model",
        type=int,
        required=True,
        help="Maximum d_model value (inclusive)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data",
        help="Directory to save plot"
    )
    args = parser.parse_args()

    plot_vary_d_model(args.metrics_dir, args.min_d_model, args.max_d_model, args.output_dir)


if __name__ == "__main__":
    main()
