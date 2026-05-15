#!/usr/bin/env python3
"""Dual-axis plot: signed projection share (bars, left) and per-bucket test
CE loss (line, right).

Reads paired files per d_model:
  bucket_projection_shares_d{d}_{Nk}.json  (signed projections, sum to 1)
  bucket_test_loss_d{d}_{Nk}.json          (per-bucket mean test CE)

Usage:
    python -m jl.double_descent.transformer.plot_projection_vs_norm_shares \\
        ./data/transformer_m2m100_shadow_adamw/05-09-0627 \\
        --loss-dir ./output/bucket_test_loss
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
    p.add_argument("source_dir", help="Directory with bucket_projection_shares_d*.json")
    p.add_argument("--loss-dir", default=None,
                   help="Directory with bucket_test_loss_d*.json (default: --source-dir)")
    p.add_argument("--output-dir", default=None,
                   help="Where to write the plot (default: --source-dir)")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(name)s] %(message)s",
                        datefmt="%H:%M:%S")
    src = Path(args.source_dir)
    loss_dir = Path(args.loss_dir) if args.loss_dir else src
    out = Path(args.output_dir) if args.output_dir else src
    out.mkdir(parents=True, exist_ok=True)

    proj_files = sorted(src.glob("bucket_projection_shares_d*_*k.json"))
    if not proj_files:
        raise FileNotFoundError(f"No bucket_projection_shares_d*.json under {src}")

    pairs = []
    for pf in proj_files:
        m = PROJ_RE.search(pf.name)
        if not m:
            continue
        d_model = int(m.group(1))
        samples_k = int(m.group(2))
        loss_path = loss_dir / f"bucket_test_loss_d{d_model}_{samples_k}k.json"
        if not loss_path.exists():
            logger.warning(f"  d={d_model}: missing {loss_path.name}, skipping")
            continue
        proj = json.loads(pf.read_text())
        loss = json.loads(loss_path.read_text())
        pairs.append((d_model, proj, loss))

    if not pairs:
        raise RuntimeError("No paired (projection, test-loss) summaries found")
    pairs.sort(key=lambda x: x[0])
    logger.info(f"Plotting {len(pairs)} d_model values: {[d for d, _, _ in pairs]}")

    n_bins = pairs[0][1]["n_bins"]
    centers = np.array(pairs[0][1]["bucket_entropy_centers"])
    baseline_share = 1.0 / n_bins

    # Detect reference vector from any one of the JSONs (default to gradient_sum
    # for backward compatibility with files that pre-date the field).
    ref = pairs[0][1].get("reference_vector", "gradient_sum")
    if ref == "final_weight":
        bar_label = r"Projection coefficient  $\langle \Delta_b, W_T\rangle / \|W_T\|^2$"
        title_v = r"$V = W_T$"
        sum_label = r"$\langle V_{grad}, W_T\rangle / \|W_T\|^2$"
    else:
        bar_label = r"Projection share  $\langle \Delta_b, V\rangle / \|V\|^2$"
        title_v = r"$V = \sum_b \Delta_b$"
        sum_label = r"$\sum_b \text{share}_b$"

    plt.rcParams.update({"font.family": "serif", "font.size": 10})
    n_panels = len(pairs)
    fig, axes = plt.subplots(
        n_panels, 1, figsize=(9, 3.6 * n_panels), dpi=150, sharex=True,
    )
    if n_panels == 1:
        axes = [axes]

    x = np.arange(n_bins)

    for ax, (d, proj_s, loss_s) in zip(axes, pairs):
        proj_shares = proj_s["projection_shares"]
        bucket_losses = loss_s["bucket_test_loss"]      # may have None
        bucket_counts = loss_s["bucket_test_tokens"]
        ax.bar(x, proj_shares, width=0.7,
               color="#d62728", alpha=0.85,
               label=bar_label)
        ax.axhline(baseline_share, color="0.5", linestyle="--", linewidth=0.8, alpha=0.8,
                   label=f"Uniform (1/{n_bins})")
        ax.axhline(0, color="0.2", linewidth=0.6)
        ax.set_ylabel("Projection share", color="#d62728")
        ax.tick_params(axis="y", colors="#d62728")

        ax2 = ax.twinx()
        loss_vals = np.array(
            [float("nan") if v is None else v for v in bucket_losses]
        )
        ax2.plot(x, loss_vals, "o-", color="#1f77b4",
                 markersize=6, markerfacecolor="white", markeredgewidth=1.4,
                 linewidth=1.5, label="Test CE / token")
        ax2.set_ylabel("Per-bucket test CE (nats)", color="#1f77b4")
        ax2.tick_params(axis="y", colors="#1f77b4")

        # Combined legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        if ax is axes[0]:
            ax.legend(lines1 + lines2, labels1 + labels2,
                      loc="upper left", fontsize=9)

        overall_ce = loss_s.get("mean_loss_overall")
        share_sum = proj_s.get("projection_shares_sum")
        title_extra = []
        if overall_ce is not None:
            title_extra.append(f"overall test CE = {overall_ce:.3f}")
        if share_sum is not None:
            title_extra.append(f"{sum_label} = {share_sum:.4f}")
        suffix = ("  (" + ", ".join(title_extra) + ")") if title_extra else ""
        ax.set_title(f"d_model = {d}{suffix}")
        ax.grid(alpha=0.3, axis="y")

    axes[-1].set_xticks(x)
    bucket_counts0 = pairs[0][2]["bucket_test_tokens"]
    axes[-1].set_xticklabels(
        [f"{c:.2g}\n(n={n})" for c, n in zip(centers, bucket_counts0)],
        rotation=0, ha="center", fontsize=8,
    )
    axes[-1].set_xlabel("Bucket entropy center (nats):  -log p_oracle(y_i)  (low = easy → high = hard)")

    fig.suptitle(
        f"Per-bucket: signed projection share (bars, {title_v}) vs test CE loss (line)",
        y=1.0,
    )
    fig.tight_layout()

    plot_path = out / "bucket_projection_vs_test_loss.png"
    fig.savefig(plot_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Wrote {plot_path}")


if __name__ == "__main__":
    main()
