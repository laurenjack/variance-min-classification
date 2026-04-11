#!/usr/bin/env python3
"""Plot evaluation results for Transformer main (non-variance) runs.

Produces a figure with 2 subplots:
- Top: Test BLEU vs d_model
- Bottom: Train/Test loss vs d_model

Usage:
    # Uncalibrated evaluation
    python -m jl.double_descent.transformer.plot_evaluation \
        ./output/transformer/03-01-1010/evaluation.jsonl \
        --output-dir ./data

    # Temperature-scaled evaluation (test/BLEU curves come from the TS JSONL,
    # train curves come from the uncalibrated evaluation.jsonl; location is
    # auto-inferred as <ts_path>.parent.parent/evaluation.jsonl unless
    # overridden with --orig-eval-path)
    python -m jl.double_descent.transformer.plot_evaluation \
        ./output/transformer/03-01-1010/temperature_scaled/temperature_scaled_evaluation.jsonl \
        --temperature-scaled
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

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


def _render(
    d_model_values: Sequence[int],
    train_loss: Sequence[float],
    test_loss: Sequence[float],
    test_bleu: Sequence[float],
    title: str,
    output_path: Path,
) -> None:
    """Render the two-subplot evaluation figure from plain arrays.

    This is the pure plotting body: takes arrays + title + output path,
    produces the figure. Used by both the uncalibrated and temperature-
    scaled entry points.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Academic paper style (matches resnet18/plot_evaluation.py)
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['DejaVu Serif', 'Times New Roman', 'Times'],
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'legend.frameon': True,
        'legend.framealpha': 0.9,
        'legend.edgecolor': '0.8',
        'axes.linewidth': 0.8,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.major.size': 4,
        'ytick.major.size': 4,
        'xtick.minor.size': 2,
        'ytick.minor.size': 2,
        'xtick.top': True,
        'ytick.right': True,
    })

    # Tableau palette (paper-friendly)
    color_loss = '#d62728'   # red
    color_bleu = '#2ca02c'   # green

    # Shared line style kwargs
    line_kw = dict(
        linewidth=1.8,
        solid_capstyle='round',
        solid_joinstyle='round',
        dash_capstyle='round',
        dash_joinstyle='round',
        antialiased=True,
    )
    test_marker = dict(marker='o', markersize=5, markeredgewidth=1.1,
                       markerfacecolor='white')
    train_marker = dict(marker='o', markersize=3.5, markeredgewidth=0)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(7, 6.5), dpi=200, sharex=True
    )

    # BLEU plot (top)
    ax1.plot(d_model_values, test_bleu, linestyle='-',
             color=color_bleu, markeredgecolor=color_bleu,
             label='Test BLEU', zorder=3, **line_kw, **test_marker)
    ax1.set_ylabel('Test BLEU Score')
    ax1.set_title(title)
    ax1.set_ylim(bottom=0)
    ax1.legend(loc='lower right')
    ax1.grid(True, which='major', linestyle=':', linewidth=0.5, alpha=0.5)
    ax1.minorticks_on()

    # Loss plot (bottom) — draw train first, test on top
    ax2.plot(d_model_values, train_loss, linestyle=(0, (4, 2)),
             color=color_loss, alpha=0.55, label='Train Loss',
             zorder=2, **line_kw, **train_marker)
    ax2.plot(d_model_values, test_loss, linestyle='-',
             color=color_loss, markeredgecolor=color_loss,
             label='Test Loss', zorder=3, **line_kw, **test_marker)
    ax2.set_xlabel(r'Transformer embedding dimension $d_{\mathrm{model}}$')
    ax2.set_ylabel('Cross-Entropy Loss')
    ax2.set_ylim(bottom=0)
    ax2.legend(loc='upper right')
    ax2.grid(True, which='major', linestyle=':', linewidth=0.5, alpha=0.5)
    ax2.minorticks_on()

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to {output_path}")


def plot_evaluation(eval_path: str, output_dir: str) -> None:
    """Plot uncalibrated BLEU + loss vs d_model.

    Args:
        eval_path: Path to evaluation.jsonl file.
        output_dir: Directory to save plot.
    """
    results = load_evaluation(eval_path)

    d_model_values = [r["d_model"] for r in results]
    train_loss = [r["train_loss"] for r in results]
    test_loss = [r["test_loss"] for r in results]
    test_bleu = [r["test_bleu"] for r in results]

    title = 'Transformer Double Descent: IWSLT14 de-en'
    output_path = Path(output_dir) / 'transformer_evaluation.png'
    _render(d_model_values, train_loss, test_loss, test_bleu,
            title, output_path)


def plot_temperature_scaled(
    ts_eval_path: str,
    orig_eval_path: str,
    output_dir: str,
) -> None:
    """Plot temperature-scaled evaluation.

    Test curves and BLEU come from the TS JSONL (ts_loss, ts_error, ts_bleu);
    train loss comes from the uncalibrated evaluation.jsonl (temperature
    scaling does not affect train-set metrics).

    Args:
        ts_eval_path: Path to temperature_scaled_evaluation.jsonl.
        orig_eval_path: Path to the uncalibrated evaluation.jsonl (for train loss).
        output_dir: Directory to save plot.
    """
    ts_results = load_evaluation(ts_eval_path)
    orig_results = load_evaluation(orig_eval_path)
    orig_by_d = {r["d_model"]: r for r in orig_results}

    d_model_values = []
    train_loss = []
    test_loss = []
    test_bleu = []
    for r in ts_results:
        d = r["d_model"]
        if d not in orig_by_d:
            raise KeyError(
                f"d_model={d} present in TS JSONL but missing from {orig_eval_path}; "
                "train curves cannot be drawn for this d_model."
            )
        d_model_values.append(d)
        test_loss.append(r["ts_loss"])
        test_bleu.append(r["ts_bleu"])
        train_loss.append(orig_by_d[d]["train_loss"])

    title = 'Transformer Double Descent: IWSLT14 de-en, Temperature Scaled'
    output_path = Path(output_dir) / 'transformer_evaluation.png'
    _render(d_model_values, train_loss, test_loss, test_bleu,
            title, output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Plot Transformer evaluation: loss/accuracy/BLEU vs d_model"
    )
    parser.add_argument(
        "eval_path",
        type=str,
        help="Path to evaluation JSONL file (uncalibrated or temperature-scaled)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save plot (default: same directory as eval_path)",
    )
    parser.add_argument(
        "--temperature-scaled",
        action="store_true",
        help="Interpret eval_path as a temperature-scaled JSONL. Train curves "
             "are pulled from the uncalibrated evaluation.jsonl (auto-inferred "
             "unless --orig-eval-path is given).",
    )
    parser.add_argument(
        "--orig-eval-path",
        type=str,
        default=None,
        help="Path to uncalibrated evaluation.jsonl (for train curves in "
             "--temperature-scaled mode). Default: <ts_path>.parent.parent/evaluation.jsonl",
    )
    args = parser.parse_args()

    ts_path = Path(args.eval_path)
    output_dir = args.output_dir if args.output_dir else str(ts_path.parent)

    if args.temperature_scaled:
        orig_eval_path = (
            args.orig_eval_path
            if args.orig_eval_path
            else str(ts_path.parent.parent / "evaluation.jsonl")
        )
        plot_temperature_scaled(args.eval_path, orig_eval_path, output_dir)
    else:
        plot_evaluation(args.eval_path, output_dir)


if __name__ == "__main__":
    main()
