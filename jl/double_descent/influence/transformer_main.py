#!/usr/bin/env python3
"""Training-point influence decomposition for an M2M100-tokenized Transformer.

Loads a single trained Transformer (e.g. d_model=96, variance split 0) and:
  1. Untying step: replace tied output_proj with a fresh nn.Linear(bias=True)
     initialized from the input embedding (so input-side embeddings don't move
     when we fine-tune the output side).
  2. Extracts per-token decoder features (post decoder_norm, pre output_proj)
     for the training and test sets, masking padding.
  3. L-BFGS fine-tunes only the output projection on (CE + lambda * ||W||^2 +
     lambda * ||b||^2), using a chunked closure (vocab is ~18K so a single
     [N_tokens, vocab] tensor would OOM).
  4. Validates the Yeh & Kim (2018) decomposition: KL between actual and
     reconstructed softmax should be ~0.
  5. Computes per-training-token influence scores against the test set:
        score_i = ||r_i||_2 * mean_t(phi_train[i] . phi_test[t])
     normalized to mean 1.
  6. Records: original vs fine-tuned test loss, decomposition KL/MSE, L-BFGS
     gradient norm, and per-token influence with sentence offsets.

Usage:
    python -m jl.double_descent.influence.transformer_main \\
        --model-path ./data/transformer_m2m100_variance/04-02-1520/model_d96_split0.pt \\
        --d-model 96 --split-id 0 \\
        --data-path ./data/iwslt14.m2m100.de-en \\
        --output-dir ./data/transformer_m2m100_variance/04-02-1520/influence/d96_split0 \\
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
    M2M100TranslationDataset,
    M2M100Vocab,
    collate_fn,
    load_m2m100_iwslt14_variance_split,
)
from jl.double_descent.transformer.transformer_model import TransformerModel

logger = logging.getLogger(__name__)


def untie_output_proj(model: TransformerModel) -> None:
    """Replace tied output_proj (bias=False, weight tied to embedding) with a
    standalone nn.Linear(bias=True). Bias initializes to zero so the model's
    forward pass is identical to before until L-BFGS runs.
    """
    d_model = model.d_model
    vocab_size = model.embedding.num_embeddings
    device = model.embedding.weight.device
    dtype = model.embedding.weight.dtype

    new_proj = nn.Linear(d_model, vocab_size, bias=True).to(device=device, dtype=dtype)
    with torch.no_grad():
        new_proj.weight.copy_(model.embedding.weight.detach())
        new_proj.bias.zero_()
    model.output_proj = new_proj


@torch.no_grad()
def evaluate_test_loss(
    model: TransformerModel,
    dataset: M2M100TranslationDataset,
    vocab: M2M100Vocab,
    device: torch.device,
    batch_size: int = 64,
) -> float:
    """Token-level cross-entropy on a dataset, ignoring padding."""
    model.eval()
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        collate_fn=partial(collate_fn, pad_idx=vocab.pad_idx),
    )
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_idx, reduction="sum")
    total_loss = 0.0
    total_tokens = 0
    for src, tgt in loader:
        src = src.to(device); tgt = tgt.to(device)
        logits = model(src, tgt[:, :-1])
        target = tgt[:, 1:].contiguous()
        loss = criterion(logits.view(-1, logits.size(-1)), target.view(-1))
        mask = target != vocab.pad_idx
        total_loss += loss.item()
        total_tokens += int(mask.sum().item())
    return total_loss / max(total_tokens, 1)


@torch.no_grad()
def extract_decoder_features(
    model: TransformerModel,
    dataset: M2M100TranslationDataset,
    vocab: M2M100Vocab,
    device: torch.device,
    batch_size: int = 64,
):
    """Forward each (src, tgt) pair, hook decoder_norm output, return:
        features:        [N_tokens, d_model] post-decoder_norm hidden states
        targets:         [N_tokens]          target token compact IDs
        sentence_offsets:[n_sent + 1]        cumulative position counts
    Padding positions are dropped.
    """
    model.eval()
    captured = {}

    def hook(_module, _inp, out):
        captured["out"] = out

    handle = model.decoder_norm.register_forward_hook(hook)

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        collate_fn=partial(collate_fn, pad_idx=vocab.pad_idx),
    )

    feats = []
    targs = []
    offsets = [0]

    try:
        for src, tgt in loader:
            src = src.to(device); tgt = tgt.to(device)
            _ = model(src, tgt[:, :-1])
            phi = captured["out"]               # [B, T-1, d_model]
            target = tgt[:, 1:]                 # [B, T-1]
            mask = target != vocab.pad_idx      # [B, T-1]

            for b in range(phi.shape[0]):
                m = mask[b]
                n = int(m.sum().item())
                if n > 0:
                    feats.append(phi[b][m].cpu())
                    targs.append(target[b][m].cpu())
                offsets.append(offsets[-1] + n)
    finally:
        handle.remove()

    features = torch.cat(feats, dim=0).contiguous()
    targets = torch.cat(targs, dim=0).contiguous()
    offsets = torch.tensor(offsets, dtype=torch.long)
    return features, targets, offsets


def l2_finetune_chunked(
    linear: nn.Linear,
    features: torch.Tensor,
    targets: torch.Tensor,
    lambda_l2: float,
    max_iter: int = 500,
    tolerance_grad: float = 1e-7,
    chunk_size: int = 4096,
) -> dict:
    """L-BFGS fine-tune of an nn.Linear (bias=True) on (features, targets).

    Loss = (1/n) * CE(features W^T + b, targets) + lambda * (||W||^2 + ||b||^2)

    Closure backprops in chunks to keep peak memory bounded by
    chunk_size * vocab_size, since features × W^T materializes [N, vocab] which
    would OOM at N ~ 1M, vocab ~ 18K.
    """
    assert linear.bias is not None
    n = features.size(0)
    device = features.device

    W = linear.weight.detach().clone().to(device).requires_grad_(True)
    b = linear.bias.detach().clone().to(device).requires_grad_(True)

    optimizer = torch.optim.LBFGS(
        [W, b],
        max_iter=max_iter,
        tolerance_grad=tolerance_grad,
        tolerance_change=0,
        line_search_fn="strong_wolfe",
        history_size=100,
    )

    eval_count = [0]
    last_log = [None]

    def closure():
        optimizer.zero_grad()
        ce_total = torch.zeros((), device=device)
        for s in range(0, n, chunk_size):
            e = min(s + chunk_size, n)
            chunk_phi = features[s:e]
            chunk_y = targets[s:e]
            chunk_logits = chunk_phi @ W.t() + b
            # sum over chunk, divide by n at end → matches mean over all tokens
            ce_chunk = F.cross_entropy(chunk_logits, chunk_y, reduction="sum") / n
            ce_chunk.backward()
            ce_total = ce_total + ce_chunk.detach()

        l2 = lambda_l2 * (W.pow(2).sum() + b.pow(2).sum())
        l2.backward()
        loss = ce_total + l2.detach()
        eval_count[0] += 1
        last_log[0] = (float(ce_total.item()), float(l2.item()))
        return loss

    final_loss = optimizer.step(closure)
    grad_norm = torch.cat([W.grad.flatten(), b.grad.flatten()]).norm().item()
    final_ce, final_l2 = last_log[0]

    with torch.no_grad():
        linear.weight.copy_(W)
        linear.bias.copy_(b)

    logger.info(
        f"L-BFGS: {eval_count[0]} closure evals, final loss={final_loss.item():.6f} "
        f"(CE={final_ce:.6f}, L2={final_l2:.6f}), grad_norm={grad_norm:.2e}"
    )
    return {
        "final_loss": float(final_loss.item()),
        "final_ce": final_ce,
        "final_l2": final_l2,
        "grad_norm": grad_norm,
        "n_closure_evals": eval_count[0],
        "lambda_l2": lambda_l2,
    }


@torch.no_grad()
def validate_decomposition_chunked(
    features: torch.Tensor,
    targets: torch.Tensor,
    linear: nn.Linear,
    lambda_l2: float,
    chunk_size: int = 4096,
) -> dict:
    """Verify W ≈ -1/(2 lambda n) * R^T Phi, b ≈ -1/(2 lambda n) * sum(R).

    Computes residuals, the reconstructed (W, b), and the KL between the
    actual and reconstructed softmax — all chunked over tokens.
    """
    n = features.size(0)
    d = features.size(1)
    C = linear.out_features
    device = features.device
    scale = -1.0 / (2.0 * lambda_l2 * n)

    W_recon = torch.zeros(C, d, device=device, dtype=torch.float32)
    b_recon = torch.zeros(C, device=device, dtype=torch.float32)

    # Pass 1: build reconstruction
    for s in range(0, n, chunk_size):
        e = min(s + chunk_size, n)
        phi = features[s:e].float()
        y = targets[s:e]
        logits = phi @ linear.weight.float().t() + linear.bias.float()
        probs = F.softmax(logits, dim=-1)
        # residual = probs - one_hot(y), in place
        probs[torch.arange(e - s, device=device), y] -= 1.0
        residuals = probs  # [chunk, C]
        W_recon += residuals.t() @ phi
        b_recon += residuals.sum(dim=0)
        del residuals, probs, logits

    W_recon *= scale
    b_recon *= scale

    # Pass 2: compute KL between actual and reconstructed softmax (chunked)
    kl_sum = 0.0
    mse_sum = 0.0
    for s in range(0, n, chunk_size):
        e = min(s + chunk_size, n)
        phi = features[s:e].float()
        logits_actual = phi @ linear.weight.float().t() + linear.bias.float()
        logits_recon = phi @ W_recon.t() + b_recon
        log_p_actual = F.log_softmax(logits_actual, dim=-1)
        p_actual = log_p_actual.exp()
        log_p_recon = F.log_softmax(logits_recon, dim=-1)
        # KL(p_actual || p_recon) = sum p_actual * (log p_actual - log p_recon)
        kl = (p_actual * (log_p_actual - log_p_recon)).sum(dim=-1).sum().item()
        mse = (logits_recon - logits_actual).pow(2).sum().item()
        kl_sum += kl
        mse_sum += mse

    kl_mean = kl_sum / n
    mse_mean = mse_sum / (n * C)
    max_W_diff = (linear.weight.float() - W_recon).abs().max().item()
    max_b_diff = (linear.bias.float() - b_recon).abs().max().item()

    logger.info(
        f"Decomposition validation: KL/token={kl_mean:.2e}, "
        f"logit_MSE={mse_mean:.2e}, max|W-W_recon|={max_W_diff:.2e}, "
        f"max|b-b_recon|={max_b_diff:.2e}"
    )
    return {
        "kl_div_per_token": kl_mean,
        "logit_mse": mse_mean,
        "max_W_diff": max_W_diff,
        "max_b_diff": max_b_diff,
    }


@torch.no_grad()
def compute_residual_norms_chunked(
    features: torch.Tensor,
    targets: torch.Tensor,
    linear: nn.Linear,
    chunk_size: int = 4096,
) -> torch.Tensor:
    """Per-token L2 norm of residual r_i = softmax(W phi_i + b) - e_{y_i}.

    Avoids materializing the full [N, vocab] residual matrix.
    """
    n = features.size(0)
    device = features.device
    norms = torch.empty(n, device=device, dtype=torch.float32)
    for s in range(0, n, chunk_size):
        e = min(s + chunk_size, n)
        phi = features[s:e].float()
        logits = phi @ linear.weight.float().t() + linear.bias.float()
        probs = F.softmax(logits, dim=-1)
        probs[torch.arange(e - s, device=device), targets[s:e]] -= 1.0
        norms[s:e] = probs.norm(dim=-1)
    return norms


@torch.no_grad()
def compute_influence_scores_chunked(
    phi_train: torch.Tensor,
    phi_test: torch.Tensor,
    grad_norms: torch.Tensor,
    train_chunk: int = 16384,
    test_chunk: int = 16384,
) -> torch.Tensor:
    """influence_i = ||r_i|| * mean_t(phi_train[i] . phi_test[t]), normalized
    so mean(influence) = 1. Computed with two-level chunking.
    """
    n_train = phi_train.size(0)
    n_test = phi_test.size(0)
    device = phi_train.device

    mean_sim = torch.zeros(n_train, device=device, dtype=torch.float32)
    for ts in range(0, n_test, test_chunk):
        te = min(ts + test_chunk, n_test)
        phi_test_block = phi_test[ts:te].float()
        for trs in range(0, n_train, train_chunk):
            tre = min(trs + train_chunk, n_train)
            dots = phi_train[trs:tre].float() @ phi_test_block.t()
            mean_sim[trs:tre] += dots.sum(dim=1)
    mean_sim /= n_test

    raw = grad_norms.float() * mean_sim
    return raw / raw.mean()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True,
                        help="Path to model_d{N}_split{K}.pt")
    parser.add_argument("--d-model", type=int, required=True)
    parser.add_argument("--split-id", type=int, required=True)
    parser.add_argument("--data-path", required=True,
                        help="Directory with M2M100-preprocessed IWSLT data")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--lambda-l2", type=float, default=1e-5)
    parser.add_argument("--max-iter", type=int, default=3000)
    parser.add_argument("--num-splits", type=int, default=4)
    parser.add_argument("--samples-per-split", type=int, default=36000)
    parser.add_argument("--subsample-seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--feature-chunk", type=int, default=4096,
                        help="Chunk size for L-BFGS / validation passes")
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

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load data (split-0 train + full test) using the same shuffle/slice as training
    logger.info(f"Loading data from {args.data_path} (variance split {args.split_id})...")
    train_dataset, _valid_dataset, test_dataset, vocab = load_m2m100_iwslt14_variance_split(
        data_dir=args.data_path,
        split_id=args.split_id,
        num_splits=args.num_splits,
        samples_per_split=args.samples_per_split,
        subsample_seed=args.subsample_seed,
    )
    logger.info(f"Train sentences: {len(train_dataset)}, test sentences: {len(test_dataset)}")
    logger.info(f"Compact vocab size: {len(vocab)}")

    # 2. Build model and load checkpoint (still tied)
    config = TDDConfig()
    model = TransformerModel(
        vocab_size=len(vocab),
        d_model=args.d_model,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        d_ff_multiplier=config.d_ff_multiplier,
        pad_idx=vocab.pad_idx,
    ).to(device)
    state = torch.load(args.model_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    logger.info(f"Loaded checkpoint from {args.model_path}")

    # 3. Original test loss (with tied output_proj)
    original_test_loss = evaluate_test_loss(model, test_dataset, vocab, device, args.batch_size)
    logger.info(f"Original test loss (tied): {original_test_loss:.6f}")

    # 4. Untie output_proj (bias=0 init → forward pass unchanged)
    untie_output_proj(model)
    untie_test_loss = evaluate_test_loss(model, test_dataset, vocab, device, args.batch_size)
    logger.info(f"Test loss after untie (sanity, should match): {untie_test_loss:.6f}")
    assert abs(untie_test_loss - original_test_loss) < 1e-5, (
        f"Untie changed loss: {original_test_loss} -> {untie_test_loss}"
    )

    # 5. Extract decoder features for train + test
    logger.info("Extracting train decoder features...")
    phi_train, y_train, offsets_train = extract_decoder_features(
        model, train_dataset, vocab, device, args.batch_size
    )
    logger.info(
        f"Train: {phi_train.shape[0]} tokens, d={phi_train.shape[1]}, "
        f"sentences={len(offsets_train) - 1}"
    )

    logger.info("Extracting test decoder features...")
    phi_test, y_test, offsets_test = extract_decoder_features(
        model, test_dataset, vocab, device, args.batch_size
    )
    logger.info(
        f"Test:  {phi_test.shape[0]} tokens, d={phi_test.shape[1]}, "
        f"sentences={len(offsets_test) - 1}"
    )

    # Move to device for L-BFGS / chunked passes
    phi_train_dev = phi_train.to(device)
    y_train_dev = y_train.to(device).long()
    phi_test_dev = phi_test.to(device)

    # Save features for downstream analysis
    torch.save(
        {
            "features": phi_train.half(), "target_ids": y_train.short(),
            "sentence_offsets": offsets_train,
        },
        output_dir / "features_train.pt",
    )
    torch.save(
        {
            "features": phi_test.half(), "target_ids": y_test.short(),
            "sentence_offsets": offsets_test,
        },
        output_dir / "features_test.pt",
    )

    # 6. L-BFGS fine-tune the untied output projection
    logger.info(f"L-BFGS fine-tune (lambda={args.lambda_l2}, max_iter={args.max_iter})...")
    ft_stats = l2_finetune_chunked(
        model.output_proj, phi_train_dev, y_train_dev,
        lambda_l2=args.lambda_l2, max_iter=args.max_iter,
        chunk_size=args.feature_chunk,
    )

    # 7. FT test loss after fine-tuning
    ft_test_loss = evaluate_test_loss(model, test_dataset, vocab, device, args.batch_size)
    logger.info(
        f"Test loss: original={original_test_loss:.6f}, FT={ft_test_loss:.6f}, "
        f"|delta|={abs(original_test_loss - ft_test_loss):.6f}"
    )

    # 8. Validate decomposition identity
    val_stats = validate_decomposition_chunked(
        phi_train_dev, y_train_dev, model.output_proj,
        lambda_l2=args.lambda_l2, chunk_size=args.feature_chunk,
    )

    # 9. Compute per-token influence
    logger.info("Computing per-token residual norms...")
    grad_norms = compute_residual_norms_chunked(
        phi_train_dev, y_train_dev, model.output_proj, chunk_size=args.feature_chunk,
    )
    logger.info("Computing per-token influence scores against test set...")
    influence = compute_influence_scores_chunked(
        phi_train_dev, phi_test_dev, grad_norms,
    )
    logger.info(
        f"Influence: mean={influence.mean().item():.4f}, "
        f"std={influence.std().item():.4f}, "
        f"min={influence.min().item():.4f}, max={influence.max().item():.4f}"
    )

    # 10. Save outputs
    torch.save(model.output_proj.state_dict(), output_dir / "untied_output_proj.pt")

    torch.save(
        {
            "influence": influence.cpu().float(),
            "residual_norms": grad_norms.cpu().float(),
            "target_ids": y_train.short(),
            "sentence_offsets": offsets_train,
            "d_model": args.d_model,
            "split_id": args.split_id,
        },
        output_dir / "influence_train.pt",
    )

    summary = {
        "model_path": str(args.model_path),
        "d_model": args.d_model,
        "split_id": args.split_id,
        "lambda_l2": args.lambda_l2,
        "n_train_tokens": int(phi_train.shape[0]),
        "n_test_tokens": int(phi_test.shape[0]),
        "original_test_loss": original_test_loss,
        "ft_test_loss": ft_test_loss,
        "ft_test_loss_delta": abs(original_test_loss - ft_test_loss),
        "lbfgs": ft_stats,
        "decomposition": val_stats,
    }
    with open(output_dir / "validation.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Wrote outputs to {output_dir}")
    logger.info("Done.")


if __name__ == "__main__":
    main()
