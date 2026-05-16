"""Transformer trainer that excludes high-entropy (surprising) tokens.

Companion to bucket_shadow_trainer.py: trains the same architecture
under the same recipe (TDDConfig, GPU-resident batches, AdamW with
warmup + cosine decay), but masks out the per-token CE loss at tokens
whose oracle entropy exceeds the cutoff (default 85th percentile).

The high-entropy tokens still appear in the decoder context — only
their prediction is excluded from the gradient. This is the natural
counterpart to "drop the mislabeled 15%" on ResNet.

Saves model + early-stop checkpoint only — no shadow tracking.
"""

import json
import logging
import math
import random
import time
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast

from jl.double_descent.transformer.transformer_config import TDDConfig
from jl.double_descent.transformer.transformer_data import (
    load_m2m100_iwslt14_train_chunk_test, GPUBatchedLoader, M2M100Vocab,
)
from jl.double_descent.transformer.transformer_model import (
    TransformerModel, count_parameters,
)
from jl.double_descent.transformer.bucket_shadow_trainer import (
    compute_bucket_id_sequences, BUCKET_PAD,
)


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# GPU-resident loader that also pre-pads per-token bucket ids
# ---------------------------------------------------------------------------


class _GPUBatchedBucketLoader:
    """Mirror of GPUBatchedLoader, plus a per-token bucket-id tensor.

    bucket_id_sequences must align with tgt_encoded[1:] (the prediction
    targets). At pad positions in the padded batch the bucket tensor
    carries BUCKET_PAD so callers can mask it out alongside the regular
    pad_idx mask.
    """

    def __init__(
        self,
        dataset,
        bucket_id_sequences: List[List[int]],
        pad_idx: int,
        max_tokens: int,
        device: torch.device,
        shuffle: bool = True,
        seed: int = 42,
    ):
        assert len(dataset) == len(bucket_id_sequences)
        self.pad_idx = pad_idx
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self.device = device

        lengths = dataset.get_lengths()
        sorted_indices = sorted(range(len(lengths)), key=lambda i: lengths[i])

        batches_idx: List[List[int]] = []
        current_batch: List[int] = []
        current_max_len = 0
        for idx in sorted_indices:
            length = lengths[idx]
            new_max_len = max(current_max_len, length)
            new_batch_tokens = new_max_len * (len(current_batch) + 1)
            if new_batch_tokens > max_tokens and current_batch:
                batches_idx.append(current_batch)
                current_batch = [idx]
                current_max_len = length
            else:
                current_batch.append(idx)
                current_max_len = new_max_len
        if current_batch:
            batches_idx.append(current_batch)

        self.batches: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        for idx_batch in batches_idx:
            src_seqs, tgt_seqs, bid_seqs = [], [], []
            for i in idx_batch:
                s, t = dataset[i]
                src_seqs.append(s)
                tgt_seqs.append(t)
                bid_seqs.append(bucket_id_sequences[i])
            max_src = max(len(s) for s in src_seqs)
            max_tgt = max(len(t) for t in tgt_seqs)
            max_bid = max_tgt - 1
            src_padded = torch.tensor(
                [s + [pad_idx] * (max_src - len(s)) for s in src_seqs],
                dtype=torch.long, device=device,
            )
            tgt_padded = torch.tensor(
                [t + [pad_idx] * (max_tgt - len(t)) for t in tgt_seqs],
                dtype=torch.long, device=device,
            )
            bid_padded = torch.tensor(
                [b + [BUCKET_PAD] * (max_bid - len(b)) for b in bid_seqs],
                dtype=torch.long, device=device,
            )
            self.batches.append((src_padded, tgt_padded, bid_padded))

    def __len__(self) -> int:
        return len(self.batches)

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __iter__(self):
        if self.shuffle:
            rng = random.Random(self.seed + self.epoch)
            order = list(range(len(self.batches)))
            rng.shuffle(order)
        else:
            order = range(len(self.batches))
        for i in order:
            yield self.batches[i]


# ---------------------------------------------------------------------------
# Eval helper (mirrors trainer.evaluate but reused here for clarity)
# ---------------------------------------------------------------------------


def _evaluate(model, loader, pad_idx: int, mask_high_entropy: bool = False
              ) -> Tuple[float, float]:
    """Evaluate CE loss / accuracy on the loader.

    If `mask_high_entropy=True`, the loader is a _GPUBatchedBucketLoader
    and only bucket-0 (low-entropy) tokens contribute. Otherwise it's a
    plain GPUBatchedLoader and all non-pad tokens contribute.
    """
    model.eval()
    total_loss, total_correct, total_tokens = 0.0, 0, 0
    with torch.no_grad():
        for batch in loader:
            if mask_high_entropy:
                src, tgt, bids = batch
            else:
                src, tgt = batch
                bids = None
            logits = model(src, tgt[:, :-1])
            target = tgt[:, 1:].contiguous()
            pad_mask = target != pad_idx
            if mask_high_entropy:
                keep = pad_mask & (bids == 0)
            else:
                keep = pad_mask
            per_token = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                target.view(-1),
                ignore_index=pad_idx,
                reduction="none",
            ).view_as(target)
            preds = logits.argmax(dim=-1)
            total_loss += float((per_token * keep.float()).sum().item())
            total_correct += int(((preds == target) & keep).sum().item())
            total_tokens += int(keep.sum().item())
    model.train()
    if total_tokens == 0:
        return 0.0, 0.0
    return total_loss / total_tokens, total_correct / total_tokens


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


def train_single_model_clean_only(
    gpu_id: int,
    d_model: int,
    train_samples: int,
    config: TDDConfig,
    output_path: str,
    data_path: str,
    oracle_path: str,
    cutoff_quantile: float = 0.85,
    test_oracle_path: str = None,
) -> None:
    """Train a transformer where the loss is masked at high-entropy tokens.

    Saves to output_path:
      - metrics_d{D}_{Nk}.jsonl
      - model_d{D}_{Nk}.pt
      - early_stop/model_d{D}_{Nk}.pt
    """
    device = torch.device(f"cuda:{gpu_id}")
    # Seed-locked init keyed by d_model so re-runs reproduce exactly.
    torch.manual_seed(42 + int(d_model))

    samples_k = train_samples // 1000
    output_suffix = f"{samples_k}k"
    log_name = f"clean_d{d_model}_{samples_k}k"
    process_logger = logging.getLogger(log_name)
    process_logger.setLevel(logging.INFO)
    process_logger.info(
        f"[clean-only] d_model={d_model}, {samples_k}K samples on GPU {gpu_id}"
    )

    train_dataset, valid_dataset, test_dataset, vocab = (
        load_m2m100_iwslt14_train_chunk_test(
            data_path,
            train_samples=train_samples,
            holdout_test_samples=6750,
            subsample_seed=config.subsample_seed,
        )
    )
    process_logger.info(
        f"Loaded {len(train_dataset)} train / {len(valid_dataset)} valid / "
        f"{len(test_dataset)} test; vocab={len(vocab)}"
    )

    bucket_id_sequences, edges, centers = compute_bucket_id_sequences(
        train_dataset, oracle_path, cutoff_quantile=cutoff_quantile,
    )
    train_cutoff = float(edges[1])
    process_logger.info(
        f"Train entropy cutoff (nats): {train_cutoff:.4f}; "
        f"bucket centers: {[round(float(c), 4) for c in centers]}"
    )

    # Build test bucket-id sequences using the SAME train-side cutoff value.
    test_bucket_id_sequences = None
    if test_oracle_path:
        from pathlib import Path as _P
        if not _P(test_oracle_path).exists():
            process_logger.warning(
                f"test_oracle_path={test_oracle_path} not found; "
                f"test eval will use all tokens (no high-entropy mask)."
            )
        else:
            test_bucket_id_sequences, test_edges, test_centers = (
                compute_bucket_id_sequences(
                    test_dataset, test_oracle_path,
                    cutoff_value=train_cutoff,
                )
            )
            n_low = sum(
                sum(1 for b in seq if b == 0) for seq in test_bucket_id_sequences
            )
            n_total = sum(len(seq) for seq in test_bucket_id_sequences)
            process_logger.info(
                f"Test bucket sizes: low={n_low}, high={n_total - n_low} "
                f"(low fraction = {n_low / max(1, n_total):.3f})"
            )

    model = TransformerModel(
        vocab_size=len(vocab),
        d_model=d_model,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        d_ff_multiplier=config.d_ff_multiplier,
        pad_idx=vocab.pad_idx,
    ).to(device)
    process_logger.info(f"Model parameters: {count_parameters(model):,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate,
        betas=(0.9, 0.98), eps=1e-9, weight_decay=0.01,
    )

    def lr_lambda(step: int) -> float:
        if step < config.warmup_steps:
            return step / config.warmup_steps
        progress = (step - config.warmup_steps) / max(
            1, config.max_steps - config.warmup_steps
        )
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    train_loader = _GPUBatchedBucketLoader(
        train_dataset, bucket_id_sequences, vocab.pad_idx,
        max_tokens=config.max_tokens, device=device,
        shuffle=True, seed=config.subsample_seed,
    )
    valid_loader = GPUBatchedLoader(
        valid_dataset, vocab.pad_idx, max_tokens=config.max_tokens,
        device=device, shuffle=False, seed=config.subsample_seed,
    )
    if test_bucket_id_sequences is not None:
        test_loader = _GPUBatchedBucketLoader(
            test_dataset, test_bucket_id_sequences, vocab.pad_idx,
            max_tokens=config.max_tokens, device=device,
            shuffle=False, seed=config.subsample_seed,
        )
        test_mask_high = True
    else:
        test_loader = GPUBatchedLoader(
            test_dataset, vocab.pad_idx, max_tokens=config.max_tokens,
            device=device, shuffle=False, seed=config.subsample_seed,
        )
        test_mask_high = False

    use_bf16 = getattr(config, "use_bf16", True)
    if use_bf16:
        process_logger.info("BF16 autocast enabled")

    metrics_file = Path(output_path) / f"metrics_d{d_model}_{output_suffix}.jsonl"
    if metrics_file.exists():
        metrics_file.unlink()

    es_dir = Path(output_path) / "early_stop"
    es_dir.mkdir(parents=True, exist_ok=True)
    es_model_path = es_dir / f"model_d{d_model}_{output_suffix}.pt"
    best_valid_loss = float("inf")
    best_valid_step = 0

    model.train()
    step = 0
    epoch = 0
    train_start = time.time()

    while step < config.max_steps:
        train_loader.set_epoch(epoch)
        epoch += 1
        for src, tgt, bids in train_loader:
            target = tgt[:, 1:].contiguous()
            # Loss mask = active token AND bucket 0 (low entropy / clean).
            keep_mask = (target != vocab.pad_idx) & (bids == 0)

            optimizer.zero_grad()
            with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
                logits = model(src, tgt[:, :-1])
                per_token_loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    target.reshape(-1),
                    reduction="none",
                    ignore_index=vocab.pad_idx,
                    label_smoothing=config.label_smoothing or 0.0,
                ).view_as(target)
                n_active = keep_mask.float().sum().clamp_min(1.0)
                loss = (per_token_loss * keep_mask.float()).sum() / n_active
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            step += 1

            if step % config.log_interval == 0:
                with torch.no_grad():
                    preds = logits.argmax(dim=-1)
                    train_acc_full = float(
                        ((preds == target) & (target != vocab.pad_idx)).sum().item()
                        / max(1, (target != vocab.pad_idx).sum().item())
                    )
                current_lr = scheduler.get_last_lr()[0] if scheduler else config.learning_rate
                metrics = {
                    "step": step,
                    "d_model": d_model,
                    "train_samples": train_samples,
                    "train_loss": float(loss.item()),
                    "train_acc": train_acc_full,
                    "lr": current_lr,
                    "n_active_tokens": int(n_active.item()),
                }
                if step % config.eval_interval == 0:
                    valid_loss, valid_acc = _evaluate(
                        model, valid_loader, vocab.pad_idx,
                        mask_high_entropy=False,
                    )
                    metrics["valid_loss"] = valid_loss
                    metrics["valid_acc"] = valid_acc
                    if valid_loss < best_valid_loss:
                        best_valid_loss = valid_loss
                        best_valid_step = step
                        torch.save(model.state_dict(), es_model_path)
                with open(metrics_file, "a") as f:
                    f.write(json.dumps(metrics) + "\n")
                elapsed = time.time() - train_start
                process_logger.info(
                    f"[d={d_model}] step {step}/{config.max_steps} "
                    f"loss={loss.item():.4f} lr={current_lr:.6f} "
                    f"n_active={int(n_active.item())} ({elapsed:.0f}s)"
                )
            if step >= config.max_steps:
                break

    # Final eval on test — masked to low-entropy tokens when test oracle present.
    test_loss, test_acc = _evaluate(
        model, test_loader, vocab.pad_idx, mask_high_entropy=test_mask_high,
    )
    final = {
        "step": step,
        "d_model": d_model,
        "train_samples": train_samples,
        "best_valid_loss": best_valid_loss,
        "best_valid_step": best_valid_step,
        "test_loss": test_loss,
        "test_acc": test_acc,
    }
    with open(metrics_file, "a") as f:
        f.write(json.dumps(final) + "\n")

    model_path = Path(output_path) / f"model_d{d_model}_{output_suffix}.pt"
    torch.save(model.state_dict(), model_path)
    total_time = time.time() - train_start
    process_logger.info(
        f"[d={d_model}] DONE clean-only: test_loss={test_loss:.4f} "
        f"test_acc={test_acc:.4f}  best_valid_loss={best_valid_loss:.4f} "
        f"@ step {best_valid_step}  total {total_time/3600:.2f}h"
    )
