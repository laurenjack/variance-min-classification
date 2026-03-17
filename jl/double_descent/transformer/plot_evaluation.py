#!/usr/bin/env python3
"""Plot evaluation results for Transformer main (non-variance) runs.

Produces a figure with 3 subplots:
- Top: Train/Test loss vs d_model, with ECE on right y-axis (dual axis)
- Middle: Test accuracy vs d_model
- Bottom: Test BLEU vs d_model

Usage:
    python -m jl.double_descent.transformer.plot_evaluation \
        ./output/transformer/03-01-1010/evaluation.jsonl \
        --output-dir ./data
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt


def load_evaluation(eval_path: str) -> List[Dict]:
    """Load evaluation results from JSONL file.

    Returns:
        List of dicts sorted by d_model.
    """
    results = []
    with open(eval_path, "r") as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))

    if not results:
        raise FileNotFoundError(f"No evaluation results found in {eval_path}")

    return sorted(results, key=lambda r: r["d_model"])


def plot_evaluation(eval_path: str, output_dir: str) -> None:
    """Plot loss/accuracy/BLEU vs d_model, with ECE on loss subplot.

    Creates a single figure with 3 subplots:
    - Top: Train/Test loss vs d_model (left axis), ECE vs d_model (right axis)
    - Middle: Test accuracy vs d_model
    - Bottom: Test BLEU vs d_model

    Args:
        eval_path: Path to evaluation.jsonl file.
        output_dir: Directory to save plot.
    """
    results = load_evaluation(eval_path)

    d_model_values = [r["d_model"] for r in results]
    train_loss = [r["train_loss"] for r in results]
    test_loss = [r["test_loss"] for r in results]
    test_acc = [r["test_acc"] for r in results]
    test_bleu = [r["test_bleu"] for r in results]
    ece = [r["ece"] for r in results]

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), dpi=150, sharex=True)

    # Loss plot (top) with ECE on right y-axis
    ax1.plot(d_model_values, test_loss, '-', color='blue', lw=2, label='Test Loss')
    ax1.plot(d_model_values, train_loss, '--', color='blue', lw=2, alpha=0.5, label='Train Loss')
    ax1.set_ylabel('Cross-Entropy Loss')
    ax1.set_title('Transformer Double Descent: IWSLT14 de-en')
    ax1.set_ylim(bottom=0)
    ax1.grid(True, alpha=0.3)

    # ECE on right y-axis (dual axis)
    ax1_ece = ax1.twinx()
    ax1_ece.plot(d_model_values, ece, '-', color='green', lw=2, label='ECE')
    ax1_ece.set_ylabel('Expected Calibration Error', color='green')
    ax1_ece.tick_params(axis='y', labelcolor='green')
    ax1_ece.set_ylim(bottom=0)

    # Combined legend for loss subplot
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_ece.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    # Accuracy plot (middle)
    ax2.plot(d_model_values, test_acc, '-', color='blue', lw=2, label='Test Accuracy')
    ax2.set_ylabel('Test Token-level Accuracy')
    ax2.set_ylim(0, 1)
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)

    # BLEU plot (bottom)
    ax3.plot(d_model_values, test_bleu, '-', color='blue', lw=2, label='Test BLEU')
    ax3.set_xlabel('Transformer embedding dimension (d_model)')
    ax3.set_ylabel('Test BLEU Score')
    ax3.set_ylim(bottom=0)
    ax3.legend(loc='lower right')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = Path(output_dir) / 'transformer_evaluation.png'
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot Transformer evaluation: loss/accuracy/BLEU vs d_model with ECE"
    )
    parser.add_argument(
        "eval_path",
        type=str,
        help="Path to evaluation.jsonl file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save plot (default: same directory as eval_path)",
    )
    args = parser.parse_args()

    output_dir = args.output_dir if args.output_dir else str(Path(args.eval_path).parent)
    plot_evaluation(args.eval_path, output_dir)


if __name__ == "__main__":
    main()
