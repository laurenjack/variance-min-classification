#!/usr/bin/env python3
"""Plot early-stopping evaluation for ResNet18 double descent.

For each k, selects the epoch with the lowest validation loss and plots
the test error and test loss at that epoch. Same 2-subplot layout as
plot_evaluation.py.

Usage:
    python -m jl.double_descent.resnet18.plot_early_stopping \
        ./data/resnet18/04-11-1602 --output-dir ./data/resnet18/04-11-1602
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

from jl.double_descent.resnet18.plot_evaluation import _render


def find_early_stop_metrics(metrics_dir: Path) -> List[Dict]:
    """For each metrics_k*.jsonl file, find the epoch with lowest val_loss.

    Returns:
        List of dicts with keys: k, train_error, test_error, train_loss,
        test_loss, best_epoch, best_val_loss — sorted by k.
    """
    results = []
    for path in sorted(metrics_dir.glob("metrics_k*.jsonl")):
        match = re.search(r"metrics_k(\d+)\.jsonl", path.name)
        if not match:
            continue
        k = int(match.group(1))

        best_row = None
        best_val_loss = float("inf")
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

        if best_row is None:
            continue

        results.append({
            "k": k,
            "train_error": best_row["train_error"],
            "test_error": best_row["test_error"],
            "train_loss": best_row["train_loss"],
            "test_loss": best_row["test_loss"],
            "best_epoch": best_row["epoch"],
            "best_val_loss": best_val_loss,
        })

    return sorted(results, key=lambda r: r["k"])


def main():
    parser = argparse.ArgumentParser(
        description="Plot ResNet18 early-stopping evaluation: error/loss vs k"
    )
    parser.add_argument(
        "metrics_dir",
        type=str,
        help="Directory containing metrics_k*.jsonl files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save plot (default: same as metrics_dir)",
    )
    parser.add_argument(
        "--noise-level",
        type=float,
        default=0.15,
        help="Label noise fraction (for plot title)",
    )
    args = parser.parse_args()

    metrics_dir = Path(args.metrics_dir)
    output_dir = args.output_dir if args.output_dir else str(metrics_dir)

    results = find_early_stop_metrics(metrics_dir)
    if not results:
        raise FileNotFoundError(
            f"No metrics_k*.jsonl files with val_loss found in {metrics_dir}"
        )

    for r in results:
        print(f"  k={r['k']:3d}  best_epoch={r['best_epoch']:4d}  "
              f"val_loss={r['best_val_loss']:.4f}  "
              f"test_error={r['test_error']:.4f}")

    k_values = [r["k"] for r in results]
    train_error = [r["train_error"] for r in results]
    test_error = [r["test_error"] for r in results]
    train_loss = [r["train_loss"] for r in results]
    test_loss = [r["test_loss"] for r in results]

    noise_pct = int(args.noise_level * 100)
    title = f"ResNet18 on CIFAR-10, Early Stopping ({noise_pct}% label noise)"
    output_path = Path(output_dir) / "resnet18_early_stopping.png"

    _render(k_values, train_error, test_error, train_loss, test_loss,
            title, output_path)


if __name__ == "__main__":
    main()
