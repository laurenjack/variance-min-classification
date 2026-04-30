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
    M2M100TranslationDataset,
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
def _ft_metrics(beta, W, b, features, targets, chunk_size):
    """At a given β, compute test loss (CE/token) plus KL(orig || FT) per token,
    plus Pearson r on flattened logits and probabilities (orig vs FT model).
    Used to quantify how much the temperature scaling diverges from β=1.
    """
    n = features.size(0)
    C = W.size(0)
    device = features.device
    ce_sum = 0.0
    kl_sum = 0.0
    n_elem = 0
    sx_l = sy_l = sxx_l = syy_l = sxy_l = 0.0
    sx_p = sy_p = sxx_p = syy_p = sxy_p = 0.0
    for s in range(0, n, chunk_size):
        e = min(s + chunk_size, n)
        f_chunk = features[s:e].float()
        z_chunk = f_chunk @ W.t() + b               # original logits
        log_p_orig = F.log_softmax(z_chunk, dim=-1)
        p_orig = log_p_orig.exp()
        z_ft = beta * z_chunk
        log_p_ft = F.log_softmax(z_ft, dim=-1)
        p_ft = log_p_ft.exp()
        # CE on target (FT predictions)
        y = targets[s:e]
        ce_sum += -log_p_ft[torch.arange(e - s, device=device), y].sum().item()
        # KL(orig || FT) per-token, summed
        kl_sum += (p_orig * (log_p_orig - log_p_ft)).sum(dim=-1).sum().item()
        # Pearson sums
        la = z_chunk.double().flatten()
        lr = z_ft.double().flatten()
        pa = p_orig.double().flatten()
        pr = p_ft.double().flatten()
        sx_l += la.sum().item();   sy_l += lr.sum().item()
        sxx_l += (la * la).sum().item(); syy_l += (lr * lr).sum().item()
        sxy_l += (la * lr).sum().item()
        sx_p += pa.sum().item();   sy_p += pr.sum().item()
        sxx_p += (pa * pa).sum().item(); syy_p += (pr * pr).sum().item()
        sxy_p += (pa * pr).sum().item()
        n_elem += la.numel()

    def pearson(sx, sy, sxx, syy, sxy, ne):
        num = ne * sxy - sx * sy
        den = ((ne * sxx - sx * sx) * (ne * syy - sy * sy)) ** 0.5
        return num / den if den > 0 else float("nan")

    return {
        "n_tokens": n,
        "ce_per_token": ce_sum / n,
        "kl_orig_to_ft_per_token": kl_sum / n,
        "pearson_logits_orig_vs_ft": pearson(sx_l, sy_l, sxx_l, syy_l, sxy_l, n_elem),
        "pearson_probs_orig_vs_ft": pearson(sx_p, sy_p, sxx_p, syy_p, sxy_p, n_elem),
    }


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


def _grad_at(beta, W, b, features, targets, lambda_l2, chunk_size):
    g_ce, _, n = _newton_pass(beta, W, b, features, targets, chunk_size)
    return g_ce / n + 2.0 * lambda_l2 * beta


def fit_beta_safe(W, b, features, targets, lambda_l2, chunk_size,
                  max_bracket_iter=20, max_bisect_iter=60, tol=1e-10):
    """Bracket-then-bisect 1D solver on g(β) = 0.

    Robust: handles ill-conditioned curvature where naive Newton diverges.
    Each iter is one chunked forward pass (~3s). Total ~30-50 evals per d_model.
    """
    history = []

    def call(b_):
        g = _grad_at(b_, W, b, features, targets, lambda_l2, chunk_size)
        history.append({"beta": float(b_), "grad": float(g)})
        return g

    g_at_1 = call(1.0)
    logger.info(f"  g(β=1) = {g_at_1:.3e}")

    # Decide direction. β* is where g flips sign.
    if abs(g_at_1) < tol:
        return 1.0, history

    if g_at_1 > 0:
        # Loss increasing in β at β=1 → β* < 1. Bracket downward.
        hi, g_hi = 1.0, g_at_1
        lo = 0.5
        for _ in range(max_bracket_iter):
            g_lo = call(lo)
            if g_lo <= 0:
                break
            hi, g_hi = lo, g_lo
            lo /= 2.0
        else:
            raise RuntimeError(f"Failed to bracket downward; β* may be < {lo}")
    else:
        # β* > 1. Bracket upward.
        lo, g_lo = 1.0, g_at_1
        hi = 2.0
        for _ in range(max_bracket_iter):
            g_hi = call(hi)
            if g_hi >= 0:
                break
            lo, g_lo = hi, g_hi
            hi *= 2.0
        else:
            raise RuntimeError(f"Failed to bracket upward; β* may be > {hi}")

    logger.info(f"  bracketed: lo={lo:.4e} (g={g_lo:.3e}), hi={hi:.4e} (g={g_hi:.3e})")

    # Bisect. Stop when grad is below tol OR interval is below tol.
    for it in range(max_bisect_iter):
        mid = 0.5 * (lo + hi)
        g_mid = call(mid)
        if abs(g_mid) < tol or (hi - lo) < tol * max(1.0, abs(mid)):
            return mid, history
        if g_mid > 0:
            hi, g_hi = mid, g_mid
        else:
            lo, g_lo = mid, g_mid
        if it % 5 == 0:
            logger.info(f"  bisect iter {it}: β={mid:.6e}, g={g_mid:.3e}, "
                        f"interval=[{lo:.4e}, {hi:.4e}]")
    return 0.5 * (lo + hi), history


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
    train_dataset, _valid_dataset, test_dataset, vocab = load_m2m100_iwslt14_variance_split(
        data_dir=args.data_path,
        split_id=args.split_id,
        num_splits=4,
        samples_per_split=36000,
    )
    config = TDDConfig()
    model = TransformerModel(
        vocab_size=len(vocab), d_model=args.d_model,
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

    # 2. Extract features (post-decoder_norm, pre-projection) for train + test
    logger.info("Extracting train decoder features...")
    f_train, y_train, offsets_train = extract_decoder_features(
        model, train_dataset, vocab, device, batch_size=args.batch_size,
    )
    f_train = f_train.to(device)
    y_train = y_train.long().to(device)
    logger.info(f"Train: {f_train.size(0)} tokens, d={f_train.size(1)}")

    logger.info("Extracting test decoder features...")
    f_test, y_test, _offsets_test = extract_decoder_features(
        model, test_dataset, vocab, device, batch_size=args.batch_size,
    )
    f_test = f_test.to(device)
    y_test = y_test.long().to(device)
    logger.info(f"Test:  {f_test.size(0)} tokens")

    # 3. Bracket + bisect on β (robust to ill-conditioning that breaks Newton)
    beta_star, history = fit_beta_safe(
        W, b, f_train, y_train, args.lambda_l2, args.feature_chunk,
        tol=args.tol_grad,
    )
    logger.info(f"β* = {beta_star:.8f}")

    # 4. Compute α_i for each train token at β*
    alphas = _alpha_pass(beta_star, W, b, f_train, y_train, args.feature_chunk)
    beta_recon = alphas.sum().item() / (2 * args.lambda_l2 * alphas.size(0))
    logger.info(
        f"|α| stats: mean={alphas.abs().mean().item():.4e}, "
        f"max={alphas.abs().max().item():.4e}, "
        f"β* = {beta_star:.6e}, β_recon = {beta_recon:.6e}, "
        f"rel_diff = {abs(beta_recon - beta_star) / max(abs(beta_star), 1e-30):.3e}"
    )

    # 5. Q1: how much does the FT model differ from the original?
    #    Compute test loss at β=1 (original) and β=β* (FT) plus KL between them.
    logger.info("Computing original vs FT test-side metrics...")
    metrics_orig = _ft_metrics(1.0, W, b, f_test, y_test, args.feature_chunk)
    metrics_ft = _ft_metrics(beta_star, W, b, f_test, y_test, args.feature_chunk)
    test_ce_orig = metrics_orig["ce_per_token"]
    test_ce_ft = metrics_ft["ce_per_token"]
    test_kl_orig_vs_ft = metrics_ft["kl_orig_to_ft_per_token"]
    logger.info(
        f"Test loss: orig={test_ce_orig:.4f}, FT={test_ce_ft:.4f}, "
        f"|delta|={abs(test_ce_orig - test_ce_ft):.4f}; "
        f"KL(orig||FT)/token = {test_kl_orig_vs_ft:.3e}, "
        f"r_logits = {metrics_ft['pearson_logits_orig_vs_ft']:.6f}, "
        f"r_probs = {metrics_ft['pearson_probs_orig_vs_ft']:.6f}"
    )

    # 6. Save outputs
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
        "n_test_tokens": int(f_test.size(0)),
        "beta_star": beta_star,
        "beta_recon_from_alphas": beta_recon,
        "alpha_stats": {
            "mean_abs": float(alphas.abs().mean().item()),
            "max_abs": float(alphas.abs().max().item()),
            "sum": float(alphas.sum().item()),
        },
        "test_metrics_original": metrics_orig,
        "test_metrics_ft": metrics_ft,
        "test_loss_delta": abs(test_ce_orig - test_ce_ft),
        "newton_history": history,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    logger.info(f"Wrote outputs to {output_dir}")
    logger.info("Done.")


if __name__ == "__main__":
    main()
