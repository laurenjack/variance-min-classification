#!/usr/bin/env python3
"""Two-bucket (top-15% surprising vs bottom-85% predictable) influence
analysis on a 'clean' test set, mirroring the CIFAR mislabeled-share
pipeline.

For each (training i, test t) pair, define:
    c_{i, t} = factor_i * |phi_i . phi_t|

where factor_i is one of:
  - label    : ||softmax(W_FT . f_i) - one_hot(y_i)||_2
                 (matches CIFAR cifar_influence_share.py)
  - distill  : ||softmax(W_FT . f_i) - softmax(W_orig . f_i + b_orig)||_2
                 (Yeh-Kim 3.2 residual; needs orig model path)

Per-test-point share:
    share_M(t) = sum_{i in TOP-15% train} c_{i,t}  /  sum over all i of c_{i,t}

Final metric per run:
    metric = mean over t in TEST_FILTERED of share_M(t)

TEST_FILTERED = test tokens whose oracle entropy is below the 85th-
percentile of TRAINING token oracle entropies (= "clean test", analog
of CIFAR's unmodified test split).

Baseline = 0.15  (top-15% holds 15% of train tokens by construction).
The support-vector hypothesis predicts share rises above baseline near
the interpolation peak.

Usage:
    python -m jl.double_descent.influence.transformer_influence_85_15 \\
        --run d=112:./data/transformer_m2m100/04-28-1534/distill_lam1e-6/d112_split0 \\
        --run d=360:./data/transformer_m2m100/04-28-1534/distill_lam1e-6_d360/d360_split0:<ORIG_MODEL_PATH> \\
        --train-oracle-path ./data/iwslt14.m2m100.de-en/train_split0_log_probs.pt \\
        --test-oracle-path ./data/iwslt14.m2m100.de-en/reference_logits.pt \\
        --output-dir ./data/influence_85_15/<timestamp>
"""

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from jl.double_descent.influence.transformer_influence_share_buckets import (
    align_oracle,
)


def parse_run_with_test(s: str):
    """Accept either:
      LABEL=PATH
      LABEL=PATH:ORIG_MODEL_PATH
      LABEL=PATH:ORIG_MODEL_PATH:TEST_FEATURES_PATH

    Returns: (label, source_dir, orig_model_path_or_None, test_features_path_or_None)
    """
    import argparse as _argparse
    if "=" not in s:
        raise _argparse.ArgumentTypeError(
            f"--run needs LABEL=PATH[:ORIG_MODEL[:TEST_FEATURES_PATH]], got {s!r}"
        )
    label, rest = s.split("=", 1)
    parts = rest.split(":")
    source_dir = Path(parts[0])
    orig = Path(parts[1]) if len(parts) > 1 else None
    test_feat = Path(parts[2]) if len(parts) > 2 else None
    return label, source_dir, orig, test_feat

logger = logging.getLogger(__name__)


def _recover_W_orig(orig_model_path: Path, vocab_size: int, d_model: int,
                    pad_idx: int, device: torch.device):
    """Re-instantiate original model + untie output_proj -> (W, b)."""
    from jl.double_descent.transformer.transformer_config import TDDConfig
    from jl.double_descent.transformer.transformer_model import TransformerModel
    from jl.double_descent.influence.transformer_main import untie_output_proj
    cfg = TDDConfig()
    m = TransformerModel(
        vocab_size=vocab_size, d_model=d_model,
        n_layers=cfg.n_layers, n_heads=cfg.n_heads,
        d_ff_multiplier=cfg.d_ff_multiplier, pad_idx=pad_idx,
    ).to(device)
    state = torch.load(orig_model_path, map_location=device, weights_only=True)
    m.load_state_dict(state)
    untie_output_proj(m)
    return (
        m.output_proj.weight.detach().clone(),
        m.output_proj.bias.detach().clone(),
    )


def load_assets(source_dir: Path, device: torch.device, orig_model_path: Path = None,
                pad_idx: int = 0, test_features_path: Path = None):
    """Load features_train, features_test, output_proj.  Returns dict with
    raw tensors plus both label and distill (if orig provided) train
    factors.

    If test_features_path is set, that file is used INSTEAD of
    source_dir/features_test.pt (lets you swap in an in-distribution
    held-out set, e.g. variance split 1).
    """
    train_blob = torch.load(source_dir / "features_train.pt", map_location="cpu",
                            weights_only=False)
    test_path = test_features_path if test_features_path is not None \
        else source_dir / "features_test.pt"
    test_blob = torch.load(test_path, map_location="cpu", weights_only=False)
    logger.info(f"  test features loaded from {test_path}")
    proj_state = torch.load(source_dir / "untied_output_proj.pt", map_location="cpu",
                            weights_only=True)

    f_train = train_blob["features"].float().to(device)
    y_train = train_blob["target_ids"].long().to(device)
    train_offsets = train_blob["sentence_offsets"]

    f_test = test_blob["features"].float().to(device)
    y_test = test_blob["target_ids"].long().to(device) if "target_ids" in test_blob else None
    test_offsets = test_blob.get("sentence_offsets")

    d_model = f_train.size(1)
    vocab_size = proj_state["weight"].size(0)
    output_proj = nn.Linear(d_model, vocab_size, bias=True).to(device)
    output_proj.load_state_dict({k: v.to(device) for k, v in proj_state.items()})
    output_proj.eval()

    distill_W_orig = None
    distill_b_orig = None
    if orig_model_path is not None:
        distill_W_orig, distill_b_orig = _recover_W_orig(
            orig_model_path, vocab_size, d_model, pad_idx, device,
        )
        logger.info(f"  loaded W_orig from {orig_model_path}")

    # Compute both train factors in one chunked pass
    n = f_train.size(0)
    chunk = 4096
    factor_label = torch.empty(n, device=device, dtype=torch.float32)
    factor_distill = torch.empty(n, device=device, dtype=torch.float32) \
        if distill_W_orig is not None else None

    with torch.no_grad():
        for s in range(0, n, chunk):
            e = min(s + chunk, n)
            logits = output_proj(f_train[s:e])
            probs = F.softmax(logits, dim=-1)
            # Label residual: ||probs - one_hot(y)||_2
            probs_at_y = probs.gather(-1, y_train[s:e].unsqueeze(-1)).squeeze(-1)
            sum_sq = probs.pow(2).sum(dim=-1)
            factor_label[s:e] = (sum_sq - 2 * probs_at_y + 1.0).clamp_min(0.0).sqrt()
            if distill_W_orig is not None:
                orig_probs = F.softmax(
                    f_train[s:e] @ distill_W_orig.t() + distill_b_orig, dim=-1,
                )
                factor_distill[s:e] = (probs - orig_probs).norm(dim=-1)

    return {
        "f_train": f_train,
        "f_test": f_test,
        "y_train_full": train_blob["target_ids"].long(),
        "y_test_full": test_blob["target_ids"].long() if "target_ids" in test_blob else None,
        "train_offsets": train_offsets,
        "test_offsets": test_offsets,
        "factor_label": factor_label,
        "factor_distill": factor_distill,
    }


def share_top_vs_baseline(
    f_train: torch.Tensor,
    train_factor: torch.Tensor,
    is_top: torch.Tensor,            # bool [N_train]
    f_test: torch.Tensor,
    train_chunk: int = 4096,
    test_chunk: int = 4096,
):
    """Per-test share = sum_{i: is_top} c_{i,t}  /  sum_i c_{i,t}.
    Returns: (mean_share, share_per_test_np)
    """
    n_train = f_train.size(0)
    n_test = f_test.size(0)
    device = f_train.device

    top_contrib = torch.zeros(n_test, device=device, dtype=torch.float64)
    total_contrib = torch.zeros(n_test, device=device, dtype=torch.float64)
    is_top_f = is_top.to(device).float()

    with torch.no_grad():
        for ts in range(0, n_test, test_chunk):
            te = min(ts + test_chunk, n_test)
            phi_test_block = f_test[ts:te]                       # [tc, d]
            for trs in range(0, n_train, train_chunk):
                tre = min(trs + train_chunk, n_train)
                dots = f_train[trs:tre] @ phi_test_block.t()     # [trc, tc]
                contrib = train_factor[trs:tre].unsqueeze(1) * dots.abs()
                top_mask = is_top_f[trs:tre].unsqueeze(1)
                top_contrib[ts:te] += (contrib * top_mask).sum(dim=0).double()
                total_contrib[ts:te] += contrib.sum(dim=0).double()

    safe_total = total_contrib.clamp_min(1e-30)
    share_per_test = (top_contrib / safe_total).cpu().numpy()
    return float(share_per_test.mean()), share_per_test


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", type=parse_run_with_test, required=True,
                        help="LABEL=PATH | LABEL=PATH:ORIG_MODEL_PATH | "
                             "LABEL=PATH:ORIG_MODEL_PATH:TEST_FEATURES_PATH "
                             "(orig path required for distill factor; test path "
                             "overrides source_dir/features_test.pt with a custom file "
                             "e.g. an in-distribution held-out split).")
    parser.add_argument("--train-oracle-path", required=True,
                        help="train_split0_log_probs.pt - per-token log p_oracle for TRAIN")
    parser.add_argument("--test-oracle-path", required=True,
                        help="reference_logits.pt - per-token entropy/log_probs for TEST")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--top-frac", type=float, default=0.15,
                        help="Top-entropy fraction (default 0.15 = top 15%%)")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logger.info(f"device = {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Train oracle: flat per-token log_p + sentence_offsets + target_ids
    train_oracle = torch.load(args.train_oracle_path, map_location="cpu",
                              weights_only=False)
    train_o_entropy = (-train_oracle["log_probs"].float())
    cutoff = float(np.quantile(train_o_entropy.numpy(), 1.0 - args.top_frac))
    logger.info(
        f"Train oracle: {train_o_entropy.numel():,} tokens; "
        f"{int((1-args.top_frac)*100)}th-percentile entropy = {cutoff:.4f} nats"
    )

    # Test oracle: either flat [N] log_probs (variance-split oracle) or
    # 2D [N, V] log_probs (reference_logits.pt has the full distribution).
    # Normalize to flat per-token log p_oracle(y_target) for align_oracle.
    test_oracle_raw = torch.load(args.test_oracle_path, map_location="cpu",
                                 weights_only=False)
    test_lp = test_oracle_raw["log_probs"].float()
    if test_lp.dim() == 2:
        log_p_y = test_lp.gather(
            1, test_oracle_raw["target_ids"].long().unsqueeze(1),
        ).squeeze(1)
        logger.info("Test oracle: 2D log_probs [N, V] -> gathered log p(y).")
    else:
        log_p_y = test_lp
        logger.info("Test oracle: flat [N] log_probs (per-target).")
    test_oracle = {
        "log_probs": log_p_y,
        "target_ids": test_oracle_raw["target_ids"],
        "sentence_offsets": test_oracle_raw["sentence_offsets"],
    }
    logger.info(f"Test oracle: {log_p_y.numel():,} tokens "
                f"(median surprise = {(-log_p_y).median().item():.3f} nats)")

    all_results = []
    for run_tuple in args.run:
        label, source_dir, orig_model_path, test_features_path = run_tuple
        logger.info(
            f"\n=== {label}: {source_dir} (orig={orig_model_path}, "
            f"test_features={test_features_path}) ==="
        )

        a = load_assets(
            source_dir, device,
            orig_model_path=orig_model_path,
            test_features_path=test_features_path,
        )

        # --- Align TRAIN entropy with feature ordering ---
        train_aligned_log_p, n_mis_train = align_oracle(
            a["y_train_full"], a["train_offsets"], train_oracle,
        )
        valid_train = ~torch.isnan(train_aligned_log_p)
        n_train_valid = int(valid_train.sum().item())
        train_entropy_aligned = (-train_aligned_log_p).to(device)
        is_top_train = (train_entropy_aligned >= cutoff).to(device)
        logger.info(
            f"  train: {n_train_valid}/{train_aligned_log_p.numel()} aligned, "
            f"top-{int(args.top_frac*100)}%: {int(is_top_train.sum().item())} "
            f"({100*float(is_top_train.float().mean().item()):.2f}%)"
        )
        if n_mis_train > 0:
            logger.warning(f"  TRAIN alignment mismatches: {n_mis_train} sentences (set entropy NaN)")

        # Subset to aligned-only train tokens
        f_train = a["f_train"][valid_train]
        is_top_train_v = is_top_train[valid_train]
        factor_label = a["factor_label"][valid_train]
        factor_distill = a["factor_distill"][valid_train] if a["factor_distill"] is not None else None

        # --- Align TEST entropy with feature ordering ---
        # test_oracle should have sentence_offsets + target_ids that match feature_test
        test_aligned_log_p, n_mis_test = align_oracle(
            a["y_test_full"], a["test_offsets"], test_oracle,
        ) if a["y_test_full"] is not None and a["test_offsets"] is not None else (None, 0)

        if test_aligned_log_p is not None:
            valid_test = ~torch.isnan(test_aligned_log_p)
            test_entropy_aligned = (-test_aligned_log_p).to(device)
            # Filter test by entropy < cutoff (= bottom-85% of TRAIN threshold)
            keep_test = (valid_test.to(device)) & (test_entropy_aligned < cutoff)
            n_test_valid = int(valid_test.sum().item())
            n_test_kept = int(keep_test.sum().item())
            logger.info(
                f"  test:  {n_test_valid}/{test_aligned_log_p.numel()} aligned, "
                f"clean (entropy < {cutoff:.3f}): {n_test_kept} "
                f"({100*n_test_kept/max(1, n_test_valid):.2f}%)"
            )
            if n_mis_test > 0:
                logger.warning(f"  TEST alignment mismatches: {n_mis_test} sentences")
            f_test = a["f_test"][keep_test]
        else:
            logger.warning("  TEST has no offsets/target_ids in features_test; using all test tokens unfiltered.")
            f_test = a["f_test"]
            keep_test = None

        # --- Compute shares for both factor variants ---
        record = {
            "label": label,
            "source_dir": str(source_dir),
            "orig_model_path": str(orig_model_path) if orig_model_path else None,
            "test_features_path": str(test_features_path) if test_features_path else None,
            "top_frac": args.top_frac,
            "entropy_cutoff": cutoff,
            "n_train_aligned": n_train_valid,
            "n_train_top": int(is_top_train_v.sum().item()),
            "n_test_kept": int(keep_test.sum().item()) if keep_test is not None else a["f_test"].size(0),
            "baseline": args.top_frac,
        }
        # Label residual
        mean_label, _ = share_top_vs_baseline(
            f_train, factor_label, is_top_train_v, f_test,
        )
        record["share_top_label"] = mean_label
        logger.info(f"  share[label]   = {mean_label:.4f}  (baseline {args.top_frac:.4f})")
        # Distill residual (if available)
        if factor_distill is not None:
            mean_distill, _ = share_top_vs_baseline(
                f_train, factor_distill, is_top_train_v, f_test,
            )
            record["share_top_distill"] = mean_distill
            logger.info(f"  share[distill] = {mean_distill:.4f}  (baseline {args.top_frac:.4f})")
        else:
            record["share_top_distill"] = None

        all_results.append(record)

        del a, f_train, f_test, factor_label
        if factor_distill is not None:
            del factor_distill
        torch.cuda.empty_cache() if device.type == "cuda" else None

    # Save
    summary = {
        "top_frac": args.top_frac,
        "entropy_cutoff": cutoff,
        "runs": all_results,
    }
    json_path = output_dir / "share_85_15.json"
    json_path.write_text(json.dumps(summary, indent=2))
    logger.info(f"\nWrote {json_path}")

    # Plot share vs d_model (parse digits from label, e.g. "d112" -> 112)
    plt.rcParams.update({"font.family": "serif", "font.size": 11})
    fig, ax = plt.subplots(1, 1, figsize=(8, 5), dpi=150)
    import re
    ds = []
    s_label = []
    s_distill = []
    for r in all_results:
        m = re.search(r"\d+", r["label"])
        if not m:
            continue
        d = int(m.group(0))
        ds.append(d)
        s_label.append(r["share_top_label"])
        s_distill.append(r.get("share_top_distill"))
    order = np.argsort(ds)
    ds = np.array(ds)[order]
    s_label = np.array(s_label)[order]
    s_distill_arr = np.array([np.nan if v is None else v for v in s_distill])[order]

    ax.plot(ds, s_label, "o-", color="#d62728", markersize=6,
            markerfacecolor="white", markeredgewidth=1.4, linewidth=1.6,
            label="factor = label residual  ||p − one_hot||")
    if not np.all(np.isnan(s_distill_arr)):
        ax.plot(ds, s_distill_arr, "s--", color="#1f77b4", markersize=6,
                markerfacecolor="white", markeredgewidth=1.4, linewidth=1.4,
                alpha=0.9, label="factor = distill residual  ||p − p_orig||")
    ax.axhline(args.top_frac, color="0.5", linestyle=":", linewidth=0.9, alpha=0.85,
               label=f"Baseline ({int(args.top_frac*100)}%)")
    ax.set_xlabel("d_model")
    ax.set_ylabel(
        r"$\mathrm{share}_M(t) = \frac{\sum_{i \in \mathrm{top}\,15\%} c_{i,t}}{\sum_i c_{i,t}}$, "
        "mean over clean test t"
    )
    ax.set_title(
        f"Influence share of top-{int(args.top_frac*100)}% entropy training tokens\n"
        f"on bottom-{int((1-args.top_frac)*100)}% entropy test tokens (clean test)"
    )
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    plot_path = output_dir / "share_85_15.png"
    fig.savefig(plot_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Wrote {plot_path}")


if __name__ == "__main__":
    main()
