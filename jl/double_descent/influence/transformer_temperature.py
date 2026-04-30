#!/usr/bin/env python3
"""Temperature-scaling decomposition for support-vector-theory analysis.

Re-parameterize the original logits as Φ = β · z where z = Φ_orig (or
Φ_orig / ||W||). Fix z, fine-tune the scalar β under L2 regularization:

    L(β) = (1/n) Σ_i CE(softmax(β · z_i), y_i) + λ β²

At the L2-CE stationary point in β:

    β* = (1/(2λn)) Σ_i (y_i - q_i) · z_i      where q_i = softmax(β* z_i)

This admits a per-train-token decomposition: each training point i contributes
the scalar α_i = (y_i - q_i) · z_i / (2λn) to β*. We use |α_i| as a per-train-
token influence measure for support-vector-theory bucket analysis.

(See critique in influence.md / discussion: this measure is a 1D reduction of
the full Yeh-Kim per-(i,t) decomposition. It loses the f_i · f_t kernel
structure and so won't reproduce the Yeh-Kim FLIP — but it answers the
related "global per-train-point calibration influence" question cleanly.)

Usage:
    python -m jl.double_descent.influence.transformer_temperature \\
        --model-path output/.../model_d40_36k.pt \\
        --d-model 40 --split-id 0 \\
        --data-path ./data/iwslt14.m2m100.de-en \\
        --output-dir output/.../temperature_lam1e-5/d40_split0 \\
        --lambda-l2 1e-5
"""

import argparse
import json
import logging
from functools import partial
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from jl.double_descent.transformer.transformer_config import TDDConfig
from jl.double_descent.transformer.transformer_data import (
    M2M100Vocab,
    collate_fn,
    load_m2m100_iwslt14_variance_split,
)
from jl.double_descent.transformer.transformer_model import TransformerModel
from jl.double_descent.influence.transformer_main import (
    extract_decoder_features,
    untie_output_proj,
)

logger = logging.getLogger(__name__)


@torch.no_grad()
def _newton_pass(beta, W, b, features, targets, chunk_size):
    """Single forward pass: returns (g_ce, hess_ce, n) where
        g_ce  = Σ_i (q_i - y_i) · z_i
        hess_ce = Σ_i Var_{q_i}[z_i]    (positive-definite contribution)
    Loss gradient = g_ce/n + 2λβ;   Loss Hessian = hess_ce/n + 2λ.
    """
    n = features.size(0)
    g_ce = 0.0
    h_ce = 0.0
    for s in range(0, n, chunk_size):
        e = min(s + chunk_size, n)
        f_chunk = features[s:e].float()
        z_chunk = f_chunk @ W.t() + b      # [chunk, V]   z = original logits
        logits_beta = beta * z_chunk
        q = F.softmax(logits_beta, dim=-1)
        y = targets[s:e]
        # g: Σ (q_i - y_i) · z_i = Σ_i (Σ_j q_{i,j} z_{i,j} - z_{i, y_i})
        eq_z = (q * z_chunk).sum(dim=-1)              # [chunk] = Σ_j q_j z_j
        z_at_y = z_chunk[torch.arange(e - s, device=z_chunk.device), y]
        g_ce += float((eq_z - z_at_y).sum().item())
        # hessian: Σ Var_q[z] = Σ (Σ q z² - (Σ q z)²)
        eq_z2 = (q * z_chunk * z_chunk).sum(dim=-1)
        h_ce += float((eq_z2 - eq_z * eq_z).sum().item())
    return g_ce, h_ce, n


@torch.no_grad()
def _alpha_pass(beta, W, b, features, targets, chunk_size):
    """At converged β, compute α_i = (y_i - q_i) · z_i for each i. Returns [N]."""
    n = features.size(0)
    alphas = torch.empty(n, device=features.device, dtype=torch.float32)
    for s in range(0, n, chunk_size):
        e = min(s + chunk_size, n)
        f_chunk = features[s:e].float()
        z_chunk = f_chunk @ W.t() + b
        q = F.softmax(beta * z_chunk, dim=-1)
        eq_z = (q * z_chunk).sum(dim=-1)
        y = targets[s:e]
        z_at_y = z_chunk[torch.arange(e - s, device=z_chunk.device), y]
        alphas[s:e] = (z_at_y - eq_z).float()
    return alphas


def fit_beta_newton(W, b, features, targets, lambda_l2, chunk_size,
                    beta_init=None, max_iter=20, tol_grad=1e-10):
    """1D Newton on β. Returns (β*, history)."""
    if beta_init is None:
        # ||W|| Frobenius is the natural scale.
        beta = float(W.float().norm().item())
        # But the "current" model has β = 1.0 (unscaled logits). Start from 1.
        beta = 1.0
    else:
        beta = float(beta_init)

    history = []
    for it in range(max_iter):
        g_ce, h_ce, n = _newton_pass(beta, W, b, features, targets, chunk_size)
        grad = g_ce / n + 2.0 * lambda_l2 * beta
        hess = h_ce / n + 2.0 * lambda_l2
        history.append({"iter": it, "beta": beta, "grad": grad, "hess": hess})
        logger.info(
            f"Newton iter {it}: β={beta:.6e}, grad={grad:.3e}, hess={hess:.3e}"
        )
        if abs(grad) < tol_grad:
            break
        # Newton step (Hessian is positive, so direction is toward minimum)
        step = grad / hess
        beta = beta - step
    return beta, history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--d-model", type=int, required=True)
    parser.add_argument("--split-id", type=int, default=0)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--lambda-l2", type=float, default=1e-5)
    parser.add_argument("--feature-chunk", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-iter", type=int, default=20)
    parser.add_argument("--tol-grad", type=float, default=1e-10)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(name)s] %(message)s",
                        datefmt="%H:%M:%S")

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Using device: {device}")

    # 1. Load data + model
    train_dataset, _, vocab = load_m2m100_iwslt14_variance_split(
        args.data_path, args.split_id, max_train_sentences=36000,
    )
    config = TDDConfig()
    model = TransformerModel(
        vocab_size=vocab.size, d_model=args.d_model,
        n_layers=config.n_layers, n_heads=config.n_heads,
        d_ff_multiplier=config.d_ff_multiplier, pad_idx=vocab.pad_idx,
    ).to(device)
    state = torch.load(args.model_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    logger.info(f"Loaded checkpoint from {args.model_path}")

    # Untie so we have the *projection* weights as W, b explicitly.
    untie_output_proj(model)
    W = model.output_proj.weight.detach().clone().to(device)
    b = model.output_proj.bias.detach().clone().to(device)

    # 2. Extract features (post-decoder_norm, pre-projection)
    logger.info("Extracting train decoder features...")
    f_train, y_train, offsets_train = extract_decoder_features(
        model, train_dataset, vocab, device, batch_size=args.batch_size,
    )
    f_train = f_train.to(device)
    y_train = y_train.long().to(device)
    logger.info(f"Train: {f_train.size(0)} tokens, d={f_train.size(1)}")

    # 3. 1D Newton on β
    beta_star, history = fit_beta_newton(
        W, b, f_train, y_train, args.lambda_l2, args.feature_chunk,
        max_iter=args.max_iter, tol_grad=args.tol_grad,
    )
    logger.info(f"β* = {beta_star:.8f}")

    # 4. Compute α_i for each train token at β*
    alphas = _alpha_pass(beta_star, W, b, f_train, y_train, args.feature_chunk)
    logger.info(
        f"|α| stats: mean={alphas.abs().mean().item():.4e}, "
        f"max={alphas.abs().max().item():.4e}, "
        f"sum={alphas.sum().item():.4e}, "
        f"|sum/(2λn) = {alphas.sum().item()/(2*args.lambda_l2*alphas.size(0)):.4e} (should ≈ β*)"
    )

    # 5. Save outputs
    torch.save(
        {
            "alphas": alphas.cpu(),
            "target_ids": y_train.short().cpu(),
            "sentence_offsets": offsets_train,
            "beta_star": beta_star,
            "lambda_l2": args.lambda_l2,
            "d_model": args.d_model,
            "split_id": args.split_id,
        },
        output_dir / "temperature_alphas.pt",
    )

    summary = {
        "model_path": str(args.model_path),
        "d_model": args.d_model,
        "split_id": args.split_id,
        "lambda_l2": args.lambda_l2,
        "n_train_tokens": int(f_train.size(0)),
        "beta_star": beta_star,
        "alpha_stats": {
            "mean_abs": float(alphas.abs().mean().item()),
            "max_abs": float(alphas.abs().max().item()),
            "sum": float(alphas.sum().item()),
        },
        "newton_history": history,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    logger.info(f"Wrote outputs to {output_dir}")
    logger.info("Done.")


if __name__ == "__main__":
    main()
