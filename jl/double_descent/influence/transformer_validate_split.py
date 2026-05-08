#!/usr/bin/env python3
"""Validate decomposition on TRAIN and TEST simultaneously (Yeh-Kim Fig. 1 style).

Builds W_recon from the training residuals (the kernel identity is anchored at
the training stationary point), then computes KL/MSE/Pearson r on BOTH train
and test feature sets — letting us reproduce Yeh-Kim's "almost 1 for both train
and test" verification.

Usage:
    python -m jl.double_descent.influence.transformer_validate_split \\
        --source-dir <distill run dir, e.g. .../d40_split0> \\
        --orig-model-path <model_d{N}_36k.pt> \\
        --d-model 40 \\
        --data-path ./data/iwslt14.m2m100.de-en \\
        --lambda-l2 1e-5
"""

import argparse
import json
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from jl.double_descent.transformer.transformer_config import TDDConfig
from jl.double_descent.transformer.transformer_data import M2M100Vocab
from jl.double_descent.transformer.transformer_model import TransformerModel
from jl.double_descent.influence.transformer_main import untie_output_proj

logger = logging.getLogger(__name__)


def _pearson(sx, sy, sxx, syy, sxy, n_e):
    num = n_e * sxy - sx * sy
    den = ((n_e * sxx - sx * sx) * (n_e * syy - sy * sy)) ** 0.5
    return num / den if den > 0 else float("nan")


@torch.no_grad()
def _build_W_recon(features, linear, lambda_l2, distill_W_orig, distill_b_orig,
                   chunk_size=4096):
    n = features.size(0)
    d = features.size(1)
    C = linear.out_features
    device = features.device
    work_dtype = features.dtype
    W_recon = torch.zeros(C, d, device=device, dtype=work_dtype)
    b_recon = torch.zeros(C, device=device, dtype=work_dtype)
    for s in range(0, n, chunk_size):
        e = min(s + chunk_size, n)
        f = features[s:e]
        logits = f @ linear.weight.t() + linear.bias
        probs = F.softmax(logits, dim=-1)
        orig_logits = f @ distill_W_orig.t() + distill_b_orig
        probs.sub_(F.softmax(orig_logits, dim=-1))
        residuals = probs
        W_recon += residuals.t() @ f
        b_recon += residuals.sum(dim=0)
        del residuals, probs, logits, orig_logits
    scale = -1.0 / (2.0 * lambda_l2 * n)
    W_recon *= scale
    b_recon *= scale
    return W_recon, b_recon


@torch.no_grad()
def _eval_recon(features, linear, W_recon, b_recon, chunk_size=4096):
    n = features.size(0)
    C = linear.out_features
    device = features.device
    kl_sum = 0.0
    mse_sum = 0.0
    n_elem = 0
    sx_l = sy_l = sxx_l = syy_l = sxy_l = 0.0
    sx_p = sy_p = sxx_p = syy_p = sxy_p = 0.0
    for s in range(0, n, chunk_size):
        e = min(s + chunk_size, n)
        f = features[s:e]
        logits_actual = f @ linear.weight.t() + linear.bias
        logits_recon = f @ W_recon.t() + b_recon
        log_p_actual = F.log_softmax(logits_actual, dim=-1)
        p_actual = log_p_actual.exp()
        log_p_recon = F.log_softmax(logits_recon, dim=-1)
        p_recon = log_p_recon.exp()
        kl = (p_actual * (log_p_actual - log_p_recon)).sum(dim=-1).sum().item()
        mse = (logits_recon - logits_actual).pow(2).sum().item()
        kl_sum += kl
        mse_sum += mse
        la = logits_actual.double().flatten()
        lr = logits_recon.double().flatten()
        pa = p_actual.double().flatten()
        pr = p_recon.double().flatten()
        sx_l += la.sum().item();   sy_l += lr.sum().item()
        sxx_l += (la * la).sum().item(); syy_l += (lr * lr).sum().item()
        sxy_l += (la * lr).sum().item()
        sx_p += pa.sum().item();   sy_p += pr.sum().item()
        sxx_p += (pa * pa).sum().item(); syy_p += (pr * pr).sum().item()
        sxy_p += (pa * pr).sum().item()
        n_elem += la.numel()
    return {
        "n_tokens": n,
        "kl_div_per_token": kl_sum / n,
        "logit_mse": mse_sum / (n * C),
        "pearson_logits": _pearson(sx_l, sy_l, sxx_l, syy_l, sxy_l, n_elem),
        "pearson_probs": _pearson(sx_p, sy_p, sxx_p, syy_p, sxy_p, n_elem),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--orig-model-path", required=True)
    parser.add_argument("--d-model", type=int, required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--lambda-l2", type=float, required=True)
    parser.add_argument("--feature-chunk", type=int, default=4096)
    parser.add_argument("--use-float64", action="store_true")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(name)s] %(message)s",
                        datefmt="%H:%M:%S")

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")

    src = Path(args.source_dir)
    vocab = M2M100Vocab(str(Path(args.data_path) / "vocab_mapping.json"))

    work_dtype = torch.float64 if args.use_float64 else torch.float32
    train_blob = torch.load(src / "features_train.pt", map_location="cpu", weights_only=False)
    test_blob = torch.load(src / "features_test.pt", map_location="cpu", weights_only=False)
    proj_state = torch.load(src / "untied_output_proj.pt", map_location="cpu", weights_only=True)
    f_train = train_blob["features"].to(work_dtype).to(device)
    f_test = test_blob["features"].to(work_dtype).to(device)

    d_model = f_train.size(1)
    vocab_size = proj_state["weight"].size(0)
    assert d_model == args.d_model

    output_proj = nn.Linear(d_model, vocab_size, bias=True).to(device).to(work_dtype)
    output_proj.load_state_dict({k: v.to(device).to(work_dtype) for k, v in proj_state.items()})
    output_proj.eval()

    config = TDDConfig()
    model = TransformerModel(
        vocab_size=vocab_size, d_model=args.d_model,
        n_layers=config.n_layers, n_heads=config.n_heads,
        d_ff_multiplier=config.d_ff_multiplier, pad_idx=vocab.pad_idx,
    ).to(device)
    state = torch.load(args.orig_model_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    untie_output_proj(model)
    distill_W_orig = model.output_proj.weight.detach().clone().to(work_dtype)
    distill_b_orig = model.output_proj.bias.detach().clone().to(work_dtype)
    del model

    W_recon, b_recon = _build_W_recon(
        f_train, output_proj, args.lambda_l2, distill_W_orig, distill_b_orig,
        chunk_size=args.feature_chunk,
    )

    train_stats = _eval_recon(f_train, output_proj, W_recon, b_recon,
                              chunk_size=args.feature_chunk)
    test_stats = _eval_recon(f_test, output_proj, W_recon, b_recon,
                             chunk_size=args.feature_chunk)
    max_W = (output_proj.weight - W_recon).abs().max().item()
    max_b = (output_proj.bias - b_recon).abs().max().item()

    out = {
        "d_model": args.d_model,
        "lambda_l2": args.lambda_l2,
        "max_W_diff": max_W,
        "max_b_diff": max_b,
        "train": train_stats,
        "test": test_stats,
    }
    out_path = src / "validation_split.json"
    out_path.write_text(json.dumps(out, indent=2))
    logger.info(f"d={args.d_model}: train r_probs={train_stats['pearson_probs']:.6f}, "
                f"test r_probs={test_stats['pearson_probs']:.6f}, "
                f"train KL={train_stats['kl_div_per_token']:.2e}, "
                f"test KL={test_stats['kl_div_per_token']:.2e}")
    logger.info(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
