#!/usr/bin/env python3
"""Bar chart of signed projection shares per d_model.

Reads bucket_projection_shares_d{d}_{Nk}.json files (produced by
extract_projection_shares.py) and renders grouped bars, one group per
d_model. Blue = low-entropy bucket (bottom 85%), red = high-entropy
bucket (top 15%). Y-axis is the scalar projection coefficient
⟨Δ_b, W_T⟩ / ‖W_T‖² (signed; can be negative).

A small annotation under each group shows the total
Σ_b share_b = ⟨V_grad, W_T⟩ / ‖W_T‖² — i.e. what fraction of W_T's
magnitude was actually built by gradients along W_T's direction.

Usage:
    python -m jl.double_descent.transformer.plot_shadow_projection_bars \
        ./data/transformer_shadows/MM-DD-HHmm
"""

import argparse
import json
import logging
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

PROJ_RE = re.compile(r"bucket_projection_shares_d(\d+)_(\d+)k\.json$")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source_dir", help="Directory containing bucket_projection_shares_d*.json")
    p.add_argument("--output-dir", default=None,
                   help="Where to write the plot (default: --source-dir)")
    p.add_argument("--title", default=None, help="Optional figure title override.")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(name)s] %(message)s",
                        datefmt="%H:%M:%S")

    src = Path(args.source_dir)
    out = Path(args.output_dir) if args.output_dir else src
    out.mkdir(parents=True, exist_ok=True)

    files = sorted(src.glob("bucket_projection_shares_d*_*k.json"))
    if not files:
        raise FileNotFoundError(f"No bucket_projection_shares_d*.json under {src}")

    rows = []
    for f in files:
        m = PROJ_RE.search(f.name)
        if not m:
            continue
        d = int(m.group(1))
        s = json.loads(f.read_text())
        if s.get("n_bins") != 2:
            logger.warning(f"  skipping {f.name}: expects 2 buckets, got {s.get('n_bins')}")
            continue
        rows.append({
            "d_model": d,
            "share_low": s["projection_shares"][0],
            "share_high": s["projection_shares"][1],
            "sum_share": s.get("projection_shares_sum"),
            "reference": s.get("reference_vector", "gradient_sum"),
        })
    if not rows:
        raise RuntimeError("No 2-bucket projection share JSONs found.")
    rows.sort(key=lambda r: r["d_model"])
    logger.info(f"Plotting {len(rows)} d_models: {[r['d_model'] for r in rows]}")

    d_models = [r["d_model"] for r in rows]
    share_low = np.array([r["share_low"] for r in rows])
    share_high = np.array([r["share_high"] for r in rows])

    ref = rows[0]["reference"]
    if ref == "final_weight":
        ylab = r"Projection coefficient  $\langle \Delta_b, W_T\rangle / \|W_T\|^2$"
        title_v = r"$V = W_T$"
    else:
        ylab = r"Projection share  $\langle \Delta_b, V\rangle / \|V\|^2$"
        title_v = r"$V = \sum_b \Delta_b$"

    x = np.arange(len(d_models))
    bar_w = 0.38

    plt.rcParams.update({"font.family": "serif", "font.size": 11})
    fig, ax = plt.subplots(1, 1, figsize=(max(7, 1.5 * len(d_models)), 5), dpi=150)
    ax.bar(x - bar_w / 2, share_low, bar_w,
           color="#1f77b4", alpha=0.9,
           label="Low-entropy bucket (bottom 85%)")
    ax.bar(x + bar_w / 2, share_high, bar_w,
           color="#d62728", alpha=0.9,
           label="High-entropy bucket (top 15%)")

    ax.axhline(0, color="0.2", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([f"d={d}" for d in d_models])
    ax.set_ylabel(ylab)
    ax.set_xlabel("Model width")
    ax.set_title(args.title or f"Per-bucket shadow projection ({title_v})")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(loc="best")

    # Annotate Σ shares under each group
    sums = [r.get("sum_share") for r in rows]
    if any(s is not None for s in sums):
        y_low = min(0, share_low.min(), share_high.min()) - 0.02
        for xi, s in zip(x, sums):
            if s is None:
                continue
            ax.annotate(
                f"Σ={s:.3f}", xy=(xi, y_low), ha="center", va="top",
                fontsize=9, color="0.35",
                xytext=(0, -2), textcoords="offset points",
            )

    fig.tight_layout()
    plot_path = out / "shadow_projection_bars.png"
    fig.savefig(plot_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Wrote {plot_path}")


if __name__ == "__main__":
    main()
