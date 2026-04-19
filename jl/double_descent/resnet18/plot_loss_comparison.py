#!/usr/bin/env python3
"""Compare test metrics across calibration methods for ResNet18 double descent.

Produces a figure with 2 subplots:
- Top: Test error vs k (vanilla, temperature-scaled, early stopping)
- Bottom: Test loss vs k (vanilla, temperature-scaled, early stopping)

Usage:
    python -m jl.double_descent.resnet18.plot_loss_comparison \
        ./data/resnet18/04-11-1602
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


def load_jsonl(path: Path) -> List[Dict]:
    results = []
    with open(path) as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    return sorted(results, key=lambda r: r["k"])


def early_stop_test_metrics(metrics_dir: Path) -> Dict[int, Tuple[float, float]]:
    """For each k, return (test_error, test_loss) at the epoch with lowest val_loss."""
    result = {}
    for path in metrics_dir.glob("metrics_k*.jsonl"):
        match = re.search(r"metrics_k(\d+)\.jsonl", path.name)
        if not match:
            continue
        k = int(match.group(1))

        best_val_loss = float("inf")
        best_row = None
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                if "val_loss" not in row:
                    continue
                if row["val_loss"] < best_val_loss:
                    best_val_loss = row["val_loss"]
                    best_row = row

        if best_row is not None:
            result[k] = (best_row["test_error"], best_row["test_loss"])
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Compare test metrics: vanilla vs temperature-scaled vs early stopping"
    )
    parser.add_argument(
        "run_dir",
        type=str,
        help="Training run directory containing evaluation.jsonl, "
             "temperature_scaled/, and metrics_k*.jsonl files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save plot (default: same as run_dir)",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir) if args.output_dir else run_dir

    eval_results = load_jsonl(run_dir / "evaluation.jsonl")
    es_metrics = early_stop_test_metrics(run_dir)

    k_values = [r["k"] for r in eval_results]
    vanilla_error = [r["test_error"] for r in eval_results]
    vanilla_loss = [r["test_loss"] for r in eval_results]
    ts_error = [r["ts_error"] for r in eval_results]
    ts_loss = [r["ts_loss"] for r in eval_results]
    es_error = [es_metrics[k][0] for k in k_values]
    es_loss = [es_metrics[k][1] for k in k_values]

    # Academic paper style
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

    line_kw = dict(
        linewidth=1.8,
        solid_capstyle='round',
        solid_joinstyle='round',
        dash_capstyle='round',
        dash_joinstyle='round',
        antialiased=True,
    )
    marker_kw = dict(marker='o', markersize=5, markeredgewidth=1.1,
                     markerfacecolor='white')

    color_vanilla = '#d62728'   # red
    color_ts = '#1f77b4'        # blue
    color_es = '#2ca02c'        # green

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 6.5), dpi=200, sharex=True)

    # Error plot (top)
    ax1.plot(k_values, vanilla_error, linestyle='-',
             color=color_vanilla, markeredgecolor=color_vanilla,
             label='Vanilla', **line_kw, **marker_kw)
    ax1.plot(k_values, ts_error, linestyle='-',
             color=color_ts, markeredgecolor=color_ts,
             label='Temperature Scaled', **line_kw, **marker_kw)
    ax1.plot(k_values, es_error, linestyle='-',
             color=color_es, markeredgecolor=color_es,
             label='Early Stopping', **line_kw, **marker_kw)
    ax1.set_ylabel('Test Error')
    ax1.set_title('ResNet18 on CIFAR-10 (15% label noise)')
    ax1.set_ylim(bottom=0)
    ax1.legend(loc='upper right')
    ax1.grid(True, which='major', linestyle=':', linewidth=0.5, alpha=0.5)
    ax1.minorticks_on()

    # Loss plot (bottom)
    ax2.plot(k_values, vanilla_loss, linestyle='-',
             color=color_vanilla, markeredgecolor=color_vanilla,
             label='Vanilla', **line_kw, **marker_kw)
    ax2.plot(k_values, ts_loss, linestyle='-',
             color=color_ts, markeredgecolor=color_ts,
             label='Temperature Scaled', **line_kw, **marker_kw)
    ax2.plot(k_values, es_loss, linestyle='-',
             color=color_es, markeredgecolor=color_es,
             label='Early Stopping', **line_kw, **marker_kw)
    ax2.set_xlabel('ResNet18 width parameter $k$')
    ax2.set_ylabel('Test Cross-Entropy Loss')
    ax2.set_ylim(bottom=0)
    ax2.legend(loc='upper right')
    ax2.grid(True, which='major', linestyle=':', linewidth=0.5, alpha=0.5)
    ax2.minorticks_on()

    output_path = output_dir / 'resnet18_loss_comparison.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()
