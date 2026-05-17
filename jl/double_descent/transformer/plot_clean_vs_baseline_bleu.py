#!/usr/bin/env python3
"""Plot clean vs baseline BLEU on the held-out test chunk, per d_model.

Inputs are two jsonls produced by compute_bleu_held_out.py (one for clean,
one for baseline). Style matches the resnet clean-vs-baseline plot:
linear x-axis, no title, simple legend.

Usage:
    python -m jl.double_descent.transformer.plot_clean_vs_baseline_bleu \
        --clean-jsonl ./data/transformer_bleu_held_out/clean_bleu.jsonl \
        --baseline-jsonl ./data/transformer_bleu_held_out/baseline_bleu.jsonl \
        --output-dir ./data/transformer_bleu_held_out
"""

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def load_bleu(jsonl_path: Path) -> dict:
    """{d_model: test_bleu} from a compute_bleu_held_out jsonl."""
    rows = {}
    for line in jsonl_path.read_text().strip().splitlines():
        r = json.loads(line)
        rows[r["d_model"]] = r["test_bleu"]
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--clean-jsonl", required=True)
    p.add_argument("--baseline-jsonl", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--output-name", default="transformer_clean_vs_baseline_bleu.png")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(name)s] %(message)s",
                        datefmt="%H:%M:%S")

    clean = load_bleu(Path(args.clean_jsonl))
    base = load_bleu(Path(args.baseline_jsonl))
    common = sorted(set(clean) & set(base))
    if not common:
        raise RuntimeError("No overlapping d_models between the two jsonls.")
    logger.info(f"Plotting {len(common)} d_models: {common}")

    clean_y = [clean[d] for d in common]
    base_y = [base[d] for d in common]

    plt.rcParams.update({"font.family": "serif", "font.size": 11})
    fig, ax = plt.subplots(1, 1, figsize=(8, 5), dpi=150)
    ax.plot(
        common, base_y, "o-", color="#d62728",
        markersize=6, markerfacecolor="white", markeredgewidth=1.2,
        linewidth=1.5, label="Baseline (trained on all tokens)",
    )
    ax.plot(
        common, clean_y, "s-", color="#1f77b4",
        markersize=6, markerfacecolor="white", markeredgewidth=1.2,
        linewidth=1.5, label="Clean (high-entropy training tokens masked out)",
    )
    ax.set_xlabel(r"Transformer embedding dimension $d_{model}$")
    ax.set_ylabel("Test BLEU")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    plot_path = out / args.output_name
    fig.savefig(plot_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Wrote {plot_path}")


if __name__ == "__main__":
    main()
