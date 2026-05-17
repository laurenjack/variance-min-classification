#!/usr/bin/env python3
"""Plot clean-trained vs baseline transformer test loss on low-entropy
test tokens, per d_model.

- Clean side: pulled from each metrics_d{d}_36k.jsonl (last record's
  test_loss / test_acc — these are already masked to low-entropy because
  the clean trainer evaluates that way).
- Baseline side: pulled from baseline_lowent_eval_seed{seed}.jsonl
  (per-d_model {test_loss_low, test_acc_low, ...}) produced by
  eval_baseline_on_low_entropy_test.py.

Usage:
    python -m jl.double_descent.transformer.plot_clean_vs_baseline_lowent \
        --clean-dir ./data/transformer_clean/05-16-1317 \
        --baseline-jsonl ./data/transformer_baseline_lowent/baseline_lowent_eval_seed674931.jsonl \
        --output-dir ./data/transformer_baseline_lowent
"""

import argparse
import json
import logging
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)
CLEAN_RE = re.compile(r"metrics_d(\d+)_(\d+)k\.jsonl$")


def load_clean(clean_dir: Path):
    """Return {d_model: test_loss_low}. Reads the LAST record of each
    metrics file, which is the trainer's final-eval line."""
    files = sorted(clean_dir.glob("metrics_d*_*k.jsonl"))
    rows = {}
    for f in files:
        m = CLEAN_RE.search(f.name)
        if not m:
            continue
        d = int(m.group(1))
        lines = f.read_text().strip().splitlines()
        if not lines:
            continue
        # walk back to the line with "test_loss" (final-eval record)
        last_with_test = None
        for line in reversed(lines):
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if "test_loss" in rec:
                last_with_test = rec
                break
        if last_with_test:
            rows[d] = {
                "test_loss": last_with_test["test_loss"],
                "test_acc": last_with_test.get("test_acc"),
            }
    return rows


def load_baseline(jsonl: Path):
    """Return {d_model: {test_loss_low, test_acc_low, ...}}."""
    rows = {}
    for line in jsonl.read_text().strip().splitlines():
        rec = json.loads(line)
        rows[rec["d_model"]] = rec
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--clean-dir", required=True)
    p.add_argument("--baseline-jsonl", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--metric", default="loss", choices=["loss", "error"],
                   help="loss = CE; error = 1 - top-1 accuracy.")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(name)s] %(message)s",
                        datefmt="%H:%M:%S")

    clean = load_clean(Path(args.clean_dir))
    base = load_baseline(Path(args.baseline_jsonl))
    d_models = sorted(set(clean) & set(base))
    if not d_models:
        raise RuntimeError(f"No overlapping d_models. clean={list(clean)} base={list(base)}")

    if args.metric == "loss":
        y_clean = [clean[d]["test_loss"] for d in d_models]
        y_base = [base[d]["test_loss_low"] for d in d_models]
        ylab = "Test CE on low-entropy tokens (nats)"
    else:
        y_clean = [1 - (clean[d]["test_acc"] or 0) for d in d_models]
        y_base = [1 - (base[d]["test_acc_low"] or 0) for d in d_models]
        ylab = "Test error on low-entropy tokens"

    plt.rcParams.update({"font.family": "serif", "font.size": 11})
    fig, ax = plt.subplots(1, 1, figsize=(9, 5), dpi=150)
    ax.plot(d_models, y_base, "o-", color="#1f77b4", markersize=5,
            markerfacecolor="white", markeredgewidth=1.4, linewidth=1.5,
            label="Baseline (trained on all tokens)")
    ax.plot(d_models, y_clean, "s-", color="#d62728", markersize=5,
            markerfacecolor="white", markeredgewidth=1.4, linewidth=1.5,
            label="Clean (high-entropy training tokens masked out)")
    ax.set_xlabel("d_model")
    ax.set_ylabel(ylab)
    ax.set_title(
        "Transformer: clean vs baseline test "
        + ("CE" if args.metric == "loss" else "error")
        + " on low-entropy tokens (top-85% predictable)"
    )
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    ax.set_xscale("log")
    fig.tight_layout()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    plot_path = out / f"clean_vs_baseline_lowent_{args.metric}.png"
    fig.savefig(plot_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Wrote {plot_path}")


if __name__ == "__main__":
    main()
