#!/usr/bin/env python3
"""Variant of the influence score that uses ONLY the correct-token residual
magnitude in the per-train factor, instead of the L2 norm across all logits.

    influence_i = (1 - p_model(y_i))  ·  RMS_t(phi_i · phi_t)

vs. the standard

    influence_i = ||r_i||_2          ·  RMS_t(phi_i · phi_t)

Reuses the saved features (features_train.pt, features_test.pt) and fitted
output projection (untied_output_proj.pt) from a prior transformer_main run,
so we skip the L-BFGS fine-tune (the underlying fit is unchanged, only the
aggregation differs).

Usage:
    python -m jl.double_descent.influence.transformer_correct_token_influence \\
        --source-dir <existing run dir with features+untied_output_proj> \\
        --output-dir <new dir for the correct-token variant>
"""

import argparse
import json
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from jl.double_descent.influence.transformer_main import (
    compute_influence_scores_chunked,  # used only for shape (we override the per-train factor)
)

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True,
                        help="Existing run dir with features_train.pt, features_test.pt, untied_output_proj.pt")
    parser.add_argument("--output-dir", required=True,
                        help="Where to write the new influence_train.pt + summary")
    parser.add_argument("--feature-chunk", type=int, default=4096)
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
    logger.info(f"Using device: {device}")

    src = Path(args.source_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Load features + targets + offsets
    logger.info(f"Loading features from {src}")
    train_blob = torch.load(src / "features_train.pt", map_location="cpu", weights_only=False)
    test_blob = torch.load(src / "features_test.pt", map_location="cpu", weights_only=False)
    proj_state = torch.load(src / "untied_output_proj.pt", map_location="cpu", weights_only=True)

    phi_train = train_blob["features"].float().to(device)        # [N_train, d_model]
    y_train = train_blob["target_ids"].long().to(device)         # [N_train]
    offsets_train = train_blob["sentence_offsets"]
    phi_test = test_blob["features"].float().to(device)          # [N_test, d_model]

    n_train = phi_train.size(0)
    n_test = phi_test.size(0)
    d_model = phi_train.size(1)
    vocab_size = proj_state["weight"].size(0)
    logger.info(
        f"phi_train: {phi_train.shape}, phi_test: {phi_test.shape}, vocab: {vocab_size}"
    )

    # Build the untied output projection from the saved state dict
    output_proj = nn.Linear(d_model, vocab_size, bias=True).to(device)
    output_proj.load_state_dict({k: v.to(device) for k, v in proj_state.items()})
    output_proj.eval()

    # ---- Per-train correct-token residual magnitude: 1 - p_model(y_i) ----
    logger.info("Computing per-train p_model(y_i)...")
    p_correct = torch.empty(n_train, device=device, dtype=torch.float32)
    with torch.no_grad():
        for s in range(0, n_train, args.feature_chunk):
            e = min(s + args.feature_chunk, n_train)
            logits = output_proj(phi_train[s:e])                      # [chunk, V]
            log_probs = F.log_softmax(logits, dim=-1)
            log_p = log_probs.gather(-1, y_train[s:e].unsqueeze(-1)).squeeze(-1)
            p_correct[s:e] = log_p.exp()
    correct_residual = 1.0 - p_correct                                  # [N_train], in [0, 1]
    logger.info(
        f"correct-residual: mean={correct_residual.mean().item():.4f}, "
        f"min={correct_residual.min().item():.4e}, max={correct_residual.max().item():.4f}"
    )

    # ---- RMS_t(phi_i · phi_t), reuse the existing chunked function ----
    # compute_influence_scores_chunked takes grad_norms × RMS_sim and normalizes.
    # To get RMS_sim alone, pass grad_norms = ones; the returned tensor will be
    # RMS_sim divided by mean(RMS_sim). We undo the normalization, then multiply
    # by correct_residual, then re-normalize so mean influence = 1.
    logger.info("Computing RMS test-similarity per train token...")
    ones = torch.ones(n_train, device=device, dtype=torch.float32)
    with torch.no_grad():
        # Use the helper but recover the un-normalized RMS_sim
        # by recomputing it directly here (cheaper than re-importing internals).
        sum_sq = torch.zeros(n_train, device=device, dtype=torch.float32)
        train_chunk = 16384
        test_chunk = 16384
        for ts in range(0, n_test, test_chunk):
            te = min(ts + test_chunk, n_test)
            phi_test_block = phi_test[ts:te]
            for trs in range(0, n_train, train_chunk):
                tre = min(trs + train_chunk, n_train)
                dots = phi_train[trs:tre] @ phi_test_block.t()
                sum_sq[trs:tre] += dots.pow(2).sum(dim=1)
        rms_sim = (sum_sq / n_test).sqrt()

    raw = correct_residual * rms_sim
    influence = raw / raw.mean()
    logger.info(
        f"influence (correct-token × RMS-sim, normalized to mean=1): "
        f"mean={influence.mean().item():.4f}, std={influence.std().item():.4f}, "
        f"min={influence.min().item():.4e}, max={influence.max().item():.4f}"
    )

    # ---- Save outputs ----
    torch.save(
        {
            "influence": influence.cpu().float(),
            "correct_residual": correct_residual.cpu().float(),
            "rms_sim": rms_sim.cpu().float(),
            "p_correct": p_correct.cpu().float(),
            "target_ids": y_train.cpu().short(),
            "sentence_offsets": offsets_train,
            "formula": "(1 - p_model(y_i)) * RMS_t(phi_i . phi_t), mean-normalized",
        },
        out / "influence_train.pt",
    )

    summary = {
        "formula": "(1 - p_model(y_i)) * RMS_t(phi_i . phi_t), mean-normalized",
        "n_train_tokens": int(n_train),
        "n_test_tokens": int(n_test),
        "correct_residual": {
            "mean": float(correct_residual.mean().item()),
            "min": float(correct_residual.min().item()),
            "max": float(correct_residual.max().item()),
        },
        "influence": {
            "mean": float(influence.mean().item()),
            "std": float(influence.std().item()),
            "min": float(influence.min().item()),
            "max": float(influence.max().item()),
        },
    }
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Wrote outputs to {out}")


if __name__ == "__main__":
    main()
