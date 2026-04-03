#!/usr/bin/env python3
"""Plot original vs fine-tuned test loss, error/ECE, and BLEU.

Reads fine_tune_evaluation.jsonl files produced by fine_tune_evaluation.py
and plots a side-by-side comparison. No GPU required.

Usage:
    python -m jl.double_descent.plot_fine_tune \
        --resnet-eval ./data/resnet18/long_double_descent/fine_tuned/lambda_1e-03/fine_tune_evaluation.jsonl \
        --transformer-eval ./data/transformer/long_double_descent_36K/fine_tuned/lambda_1e-03/fine_tune_evaluation.jsonl \
        --output-dir ./data
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt


def _load_fine_tune_eval(eval_path: str) -> List[Dict]:
    """Load fine_tune_evaluation.jsonl as a sorted list of dicts."""
    entries = []
    with open(eval_path) as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))
    return entries


def plot_fine_tune_comparison(
    resnet_entries: Optional[List[Dict]],
    transformer_entries: Optional[List[Dict]],
    output_dir: str,
) -> None:
    """Plot side-by-side original vs fine-tuned metrics.

    Top row: test loss for both architectures.
    Bottom row: test error + ECE (dual axis) for ResNet, BLEU for Transformer.
    """
    num_cols = sum(1 for d in [resnet_entries, transformer_entries] if d is not None)
    if num_cols == 0:
        raise ValueError("No data to plot")

    fig, axes = plt.subplots(2, num_cols, figsize=(7 * num_cols, 10), dpi=150)
    if num_cols == 1:
        axes = axes.reshape(2, 1)

    col_idx = 0

    if resnet_entries is not None:
        entries = sorted(resnet_entries, key=lambda e: e["k"])
        k_vals = [e["k"] for e in entries]
        orig_loss = [e["original_loss"] for e in entries]
        ft_loss = [e["fine_tuned_loss"] for e in entries]
        orig_err = [e["original_error"] for e in entries]
        ft_err = [e["fine_tuned_error"] for e in entries]
        orig_ece = [e["original_ece"] for e in entries]
        ft_ece = [e["fine_tuned_ece"] for e in entries]

        # Top: loss
        ax_loss = axes[0, col_idx]
        ax_loss.plot(k_vals, orig_loss, "-o", color="blue", lw=2, label="Original")
        ax_loss.plot(k_vals, ft_loss, "--s", color="red", lw=2, label="Fine-tuned")
        ax_loss.set_xlabel("ResNet18 width parameter k")
        ax_loss.set_ylabel("Test Cross-Entropy Loss")
        ax_loss.set_title("ResNet18 on CIFAR-10 (15% label noise)")
        ax_loss.legend(loc="upper right")
        ax_loss.grid(True, alpha=0.3)
        ax_loss.set_ylim(bottom=0)

        # Bottom: error (left axis) + ECE (right axis)
        ax_err = axes[1, col_idx]
        l1, = ax_err.plot(k_vals, orig_err, "-o", color="blue", lw=2, label="Original Error")
        l2, = ax_err.plot(k_vals, ft_err, "--s", color="red", lw=2, label="Fine-tuned Error")
        ax_err.set_xlabel("ResNet18 width parameter k")
        ax_err.set_ylabel("Test Error")
        ax_err.grid(True, alpha=0.3)
        ax_err.set_ylim(bottom=0)

        ax_ece = ax_err.twinx()
        l3, = ax_ece.plot(k_vals, orig_ece, "-^", color="blue", lw=1.5, alpha=0.6, label="Original ECE")
        l4, = ax_ece.plot(k_vals, ft_ece, "--v", color="red", lw=1.5, alpha=0.6, label="Fine-tuned ECE")
        ax_ece.set_ylabel("ECE")
        ax_ece.set_ylim(bottom=0)

        ax_err.legend(handles=[l1, l2, l3, l4], loc="upper right")

        col_idx += 1

    if transformer_entries is not None:
        entries = sorted(transformer_entries, key=lambda e: e["d_model"])
        d_vals = [e["d_model"] for e in entries]
        orig_loss = [e["original_loss"] for e in entries]
        ft_loss = [e["fine_tuned_loss"] for e in entries]
        orig_bleu = [e["original_bleu"] for e in entries]
        ft_bleu = [e["fine_tuned_bleu"] for e in entries]

        # Top: loss
        ax_loss = axes[0, col_idx]
        ax_loss.plot(d_vals, orig_loss, "-o", color="blue", lw=2, label="Original")
        ax_loss.plot(d_vals, ft_loss, "--s", color="red", lw=2, label="Fine-tuned")
        ax_loss.set_xlabel("Transformer embedding dimension d_model")
        ax_loss.set_ylabel("Test Cross-Entropy Loss")
        ax_loss.set_title("Transformer on IWSLT14 de-en")
        ax_loss.legend(loc="upper right")
        ax_loss.grid(True, alpha=0.3)
        ax_loss.set_ylim(bottom=0)

        # Bottom: BLEU
        ax_bleu = axes[1, col_idx]
        ax_bleu.plot(d_vals, orig_bleu, "-o", color="blue", lw=2, label="Original")
        ax_bleu.plot(d_vals, ft_bleu, "--s", color="red", lw=2, label="Fine-tuned")
        ax_bleu.set_xlabel("Transformer embedding dimension d_model")
        ax_bleu.set_ylabel("Test BLEU")
        ax_bleu.legend(loc="lower right")
        ax_bleu.grid(True, alpha=0.3)
        ax_bleu.set_ylim(bottom=0)

    plt.tight_layout()
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_path = Path(output_dir) / "fine_tune_comparison.png"
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot original vs fine-tuned metrics from evaluation JSONL files"
    )
    parser.add_argument(
        "--resnet-eval",
        type=str,
        default=None,
        help="Path to ResNet18 fine_tune_evaluation.jsonl",
    )
    parser.add_argument(
        "--transformer-eval",
        type=str,
        default=None,
        help="Path to Transformer fine_tune_evaluation.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data",
        help="Directory to save plot",
    )
    args = parser.parse_args()

    if args.resnet_eval is None and args.transformer_eval is None:
        parser.error("At least one of --resnet-eval or --transformer-eval is required")

    resnet_entries = None
    transformer_entries = None

    if args.resnet_eval:
        resnet_entries = _load_fine_tune_eval(args.resnet_eval)

    if args.transformer_eval:
        transformer_entries = _load_fine_tune_eval(args.transformer_eval)

    plot_fine_tune_comparison(resnet_entries, transformer_entries, args.output_dir)


if __name__ == "__main__":
    main()
