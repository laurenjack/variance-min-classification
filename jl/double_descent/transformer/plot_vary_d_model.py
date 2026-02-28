#!/usr/bin/env python3
"""Plot final metrics across varying d_model values, overlaying 4K and 18K samples.

Usage:
    python -m jl.double_descent.transformer.plot_vary_d_model ./output --output-dir ./data
"""

import argparse
import glob
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


def load_all_metrics(metrics_dir: str) -> Dict[str, List[Dict]]:
    """Load metrics from all metrics_d*_*k.jsonl files, grouped by sample size.

    Args:
        metrics_dir: Directory containing metrics_d*_*k.jsonl files.

    Returns:
        Dict with keys '4k' and '18k', each containing list of metrics dicts.
    """
    path = Path(metrics_dir)
    pattern = str(path / "metrics_d*_*k.jsonl")
    files = sorted(glob.glob(pattern))

    if not files:
        raise FileNotFoundError(f"No metrics_d*_*k.jsonl files found in {path}")

    # Group by sample size
    metrics_by_samples = {'4k': [], '18k': []}

    for file_path in files:
        # Extract sample size from filename (metrics_d64_18k.jsonl -> 18k)
        filename = Path(file_path).stem
        match = re.search(r'_(\d+k)$', filename)
        if not match:
            continue
        samples_k = match.group(1)

        if samples_k not in metrics_by_samples:
            continue

        with open(file_path, 'r') as f:
            for line in f:
                if line.strip():
                    metrics_by_samples[samples_k].append(json.loads(line))

    # Check we have data
    for k, v in metrics_by_samples.items():
        if not v:
            print(f"Warning: No metrics found for {k} samples")

    return metrics_by_samples


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
    output_dir: str,
) -> None:
    """Plot final loss, accuracy, and BLEU vs d_model with 4K and 18K overlaid.

    Creates a single figure with 3 subplots:
    - Top: Test loss vs d_model (4K and 18K overlaid)
    - Middle: Test accuracy vs d_model (4K and 18K overlaid)
    - Bottom: Test BLEU vs d_model (4K and 18K overlaid)

    Args:
        metrics_dir: Directory containing metrics_d*_*k.jsonl files.
        output_dir: Directory to save plot.
    """
    metrics_by_samples = load_all_metrics(metrics_dir)

    # Get final metrics for each sample size
    data_by_samples = {}
    for samples_k, metrics in metrics_by_samples.items():
        if metrics:
            data_by_samples[samples_k] = get_final_metrics(metrics)

    if not data_by_samples:
        raise ValueError("No final metrics found in any files")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), dpi=150, sharex=True)

    # Colors for each sample size
    colors = {'4k': 'red', '18k': 'blue'}
    labels = {'4k': '4K samples', '18k': '18K samples'}

    # Loss plot (test only)
    for samples_k, data in data_by_samples.items():
        ax1.plot(data['d_model_values'], data['test_loss'], '-o', color=colors[samples_k],
                 lw=2, label=labels[samples_k])
    ax1.set_ylabel('Test Cross-Entropy Loss')
    ax1.set_title('Transformer Double Descent: IWSLT14 de-en')
    ax1.set_ylim(bottom=0)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Accuracy plot (test only)
    for samples_k, data in data_by_samples.items():
        ax2.plot(data['d_model_values'], data['test_acc'], '-o', color=colors[samples_k],
                 lw=2, label=labels[samples_k])
    ax2.set_ylabel('Test Token-level Accuracy')
    ax2.set_ylim(0, 1)
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)

    # BLEU plot (test only)
    for samples_k, data in data_by_samples.items():
        ax3.plot(data['d_model_values'], data['test_bleu'], '-o', color=colors[samples_k],
                 lw=2, label=labels[samples_k])
    ax3.set_xlabel('Transformer embedding dimension (d_model)')
    ax3.set_ylabel('Test BLEU Score')
    ax3.set_ylim(bottom=0)
    ax3.legend(loc='lower right')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = Path(output_dir) / 'transformer_double_descent.png'
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot Transformer double descent: final metrics vs d_model (4K and 18K overlaid)"
    )
    parser.add_argument(
        "metrics_dir",
        type=str,
        help="Directory containing metrics_d*_*k.jsonl files"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data",
        help="Directory to save plot"
    )
    args = parser.parse_args()

    plot_vary_d_model(args.metrics_dir, args.output_dir)


if __name__ == "__main__":
    main()
