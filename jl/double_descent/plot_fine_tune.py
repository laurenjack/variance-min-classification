#!/usr/bin/env python3
"""Plot original vs fine-tuned vs temperature-scaled test metrics.

Reads fine_tune_evaluation.jsonl and/or temperature_scaled_evaluation.jsonl
files produced by fine_tune_evaluation.py and plots a comparison. No GPU required.

Usage:
    python -m jl.double_descent.plot_fine_tune \
        --resnet-ft-eval ./data/resnet18/long_double_descent/fine_tuned/lambda_1e-03/fine_tune_evaluation.jsonl \
        --resnet-ts-eval ./data/resnet18/long_double_descent/temperature_scaled/temperature_scaled_evaluation.jsonl \
        --transformer-ft-eval ./data/transformer/long_double_descent_36K/fine_tuned/lambda_1e-03/fine_tune_evaluation.jsonl \
        --transformer-ts-eval ./data/transformer/long_double_descent_36K/temperature_scaled/temperature_scaled_evaluation.jsonl \
        --output-dir ./data
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt


def _load_eval(eval_path: str) -> List[Dict]:
    """Load an evaluation JSONL as a list of dicts."""
    entries = []
    with open(eval_path) as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))
    return entries


def plot_fine_tune_comparison(
    resnet_ft: Optional[List[Dict]],
    resnet_ts: Optional[List[Dict]],
    transformer_ft: Optional[List[Dict]],
    transformer_ts: Optional[List[Dict]],
    output_dir: str,
    output_name: str = "fine_tune_comparison.png",
) -> None:
    """Plot comparison of calibration approaches.

    Top row: test loss. Bottom row: error+ECE (ResNet) or BLEU (Transformer).
    Up to 3 lines per subplot: original, fine-tuned, temperature-scaled.
    """
    has_resnet = resnet_ft is not None or resnet_ts is not None
    has_transformer = transformer_ft is not None or transformer_ts is not None
    num_cols = sum([has_resnet, has_transformer])
    if num_cols == 0:
        raise ValueError("No data to plot")

    fig, axes = plt.subplots(2, num_cols, figsize=(7 * num_cols, 10), dpi=150)
    if num_cols == 1:
        axes = axes.reshape(2, 1)

    col_idx = 0

    if has_resnet:
        # Merge data from ft and ts evals (both have original_ fields)
        # Use ft as primary source for original + fine-tuned, ts for temperature-scaled
        ft_by_k = {e["k"]: e for e in sorted(resnet_ft, key=lambda e: e["k"])} if resnet_ft else {}
        ts_by_k = {e["k"]: e for e in sorted(resnet_ts, key=lambda e: e["k"])} if resnet_ts else {}
        all_k = sorted(set(list(ft_by_k.keys()) + list(ts_by_k.keys())))

        # Original metrics (from whichever source is available)
        orig_source = ft_by_k if ft_by_k else ts_by_k
        orig_loss = [orig_source[k]["original_loss"] for k in all_k]
        orig_err = [orig_source[k]["original_error"] for k in all_k]
        orig_ece = [orig_source[k]["original_ece"] for k in all_k]

        # Top: loss
        ax_loss = axes[0, col_idx]
        ax_loss.plot(all_k, orig_loss, "-o", color="blue", lw=2, label="Original")
        if ft_by_k:
            ft_loss = [ft_by_k[k]["fine_tuned_loss"] for k in all_k]
            ax_loss.plot(all_k, ft_loss, "--s", color="red", lw=2, label="Fine-tuned")
        if ts_by_k:
            ts_loss = [ts_by_k[k]["ts_loss"] for k in all_k]
            ax_loss.plot(all_k, ts_loss, ":D", color="green", lw=2, label="Temp-scaled")
        ax_loss.set_xlabel("ResNet18 width parameter k")
        ax_loss.set_ylabel("Test Cross-Entropy Loss")
        ax_loss.set_title("ResNet18 on CIFAR-10 (15% label noise)")
        ax_loss.legend(loc="upper right")
        ax_loss.grid(True, alpha=0.3)
        ax_loss.set_ylim(bottom=0)

        # Bottom: error (left) + ECE (right)
        ax_err = axes[1, col_idx]
        err_handles = []
        l, = ax_err.plot(all_k, orig_err, "-o", color="blue", lw=2, label="Original Error")
        err_handles.append(l)
        if ft_by_k:
            ft_err = [ft_by_k[k]["fine_tuned_error"] for k in all_k]
            l, = ax_err.plot(all_k, ft_err, "--s", color="red", lw=2, label="Fine-tuned Error")
            err_handles.append(l)
        if ts_by_k:
            ts_err = [ts_by_k[k]["ts_error"] for k in all_k]
            l, = ax_err.plot(all_k, ts_err, ":D", color="green", lw=2, label="Temp-scaled Error")
            err_handles.append(l)
        ax_err.set_xlabel("ResNet18 width parameter k")
        ax_err.set_ylabel("Test Error")
        ax_err.grid(True, alpha=0.3)
        ax_err.set_ylim(bottom=0)

        ax_ece = ax_err.twinx()
        l, = ax_ece.plot(all_k, orig_ece, "-^", color="blue", lw=1.5, alpha=0.6, label="Original ECE")
        err_handles.append(l)
        if ft_by_k:
            ft_ece = [ft_by_k[k]["fine_tuned_ece"] for k in all_k]
            l, = ax_ece.plot(all_k, ft_ece, "--v", color="red", lw=1.5, alpha=0.6, label="Fine-tuned ECE")
            err_handles.append(l)
        if ts_by_k:
            ts_ece = [ts_by_k[k]["ts_ece"] for k in all_k]
            l, = ax_ece.plot(all_k, ts_ece, ":x", color="green", lw=1.5, alpha=0.6, label="Temp-scaled ECE")
            err_handles.append(l)
        ax_ece.set_ylabel("ECE")
        ax_ece.set_ylim(bottom=0)
        ax_err.legend(handles=err_handles, loc="upper right")

        col_idx += 1

    if has_transformer:
        ft_by_d = {e["d_model"]: e for e in sorted(transformer_ft, key=lambda e: e["d_model"])} if transformer_ft else {}
        ts_by_d = {e["d_model"]: e for e in sorted(transformer_ts, key=lambda e: e["d_model"])} if transformer_ts else {}
        all_d = sorted(set(list(ft_by_d.keys()) + list(ts_by_d.keys())))

        orig_source = ft_by_d if ft_by_d else ts_by_d
        orig_loss = [orig_source[d]["original_loss"] for d in all_d]

        # Top: loss
        ax_loss = axes[0, col_idx]
        ax_loss.plot(all_d, orig_loss, "-o", color="blue", lw=2, label="Original")
        if ft_by_d:
            ft_loss = [ft_by_d[d]["fine_tuned_loss"] for d in all_d]
            ax_loss.plot(all_d, ft_loss, "--s", color="red", lw=2, label="Fine-tuned")
        if ts_by_d:
            ts_loss = [ts_by_d[d]["ts_loss"] for d in all_d]
            ax_loss.plot(all_d, ts_loss, ":D", color="green", lw=2, label="Temp-scaled")
        ax_loss.set_xlabel("Transformer embedding dimension d_model")
        ax_loss.set_ylabel("Test Cross-Entropy Loss")
        ax_loss.set_title("Transformer on IWSLT14 de-en")
        ax_loss.legend(loc="upper right")
        ax_loss.grid(True, alpha=0.3)
        ax_loss.set_ylim(bottom=0)

        # Bottom: BLEU
        orig_bleu_source = ft_by_d if ft_by_d else ts_by_d
        orig_bleu = [orig_bleu_source[d]["original_bleu"] for d in all_d]

        ax_bleu = axes[1, col_idx]
        ax_bleu.plot(all_d, orig_bleu, "-o", color="blue", lw=2, label="Original")
        if ft_by_d:
            ft_bleu = [ft_by_d[d]["fine_tuned_bleu"] for d in all_d]
            ax_bleu.plot(all_d, ft_bleu, "--s", color="red", lw=2, label="Fine-tuned")
        if ts_by_d:
            ts_bleu = [ts_by_d[d]["ts_bleu"] for d in all_d]
            ax_bleu.plot(all_d, ts_bleu, ":D", color="green", lw=2, label="Temp-scaled")
        ax_bleu.set_xlabel("Transformer embedding dimension d_model")
        ax_bleu.set_ylabel("Test BLEU")
        ax_bleu.legend(loc="lower right")
        ax_bleu.grid(True, alpha=0.3)
        ax_bleu.set_ylim(bottom=0)

    plt.tight_layout()
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_path = Path(output_dir) / output_name
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot original vs fine-tuned vs temperature-scaled metrics"
    )
    parser.add_argument(
        "--resnet-ft-eval",
        type=str,
        default=None,
        help="Path to ResNet18 fine_tune_evaluation.jsonl",
    )
    parser.add_argument(
        "--resnet-ts-eval",
        type=str,
        default=None,
        help="Path to ResNet18 temperature_scaled_evaluation.jsonl",
    )
    parser.add_argument(
        "--transformer-ft-eval",
        type=str,
        default=None,
        help="Path to Transformer fine_tune_evaluation.jsonl",
    )
    parser.add_argument(
        "--transformer-ts-eval",
        type=str,
        default=None,
        help="Path to Transformer temperature_scaled_evaluation.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data",
        help="Directory to save plot",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="fine_tune_comparison.png",
        help="Output filename (default: fine_tune_comparison.png)",
    )
    args = parser.parse_args()

    if all(v is None for v in [args.resnet_ft_eval, args.resnet_ts_eval,
                                args.transformer_ft_eval, args.transformer_ts_eval]):
        parser.error("At least one eval file is required")

    resnet_ft = _load_eval(args.resnet_ft_eval) if args.resnet_ft_eval else None
    resnet_ts = _load_eval(args.resnet_ts_eval) if args.resnet_ts_eval else None
    transformer_ft = _load_eval(args.transformer_ft_eval) if args.transformer_ft_eval else None
    transformer_ts = _load_eval(args.transformer_ts_eval) if args.transformer_ts_eval else None

    plot_fine_tune_comparison(resnet_ft, resnet_ts, transformer_ft, transformer_ts, args.output_dir, args.output_name)


if __name__ == "__main__":
    main()
