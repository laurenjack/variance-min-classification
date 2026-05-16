"""Single-model trainer that decomposes the AdamW trajectory by oracle
entropy bucket of the target tokens.

We replace torch.optim.AdamW with a manual implementation so we can keep
**per-bucket** first moments while sharing the second moment.  The math:

    g_t = sum_b g_t^b      (the per-bucket gradients we already get
                             from 10 backward passes per step)
    m_t   = beta1 * m_{t-1}   + (1 - beta1) * g_t
    m_t^b = beta1 * m_{t-1}^b + (1 - beta1) * g_t^b
            => m_t = sum_b m_t^b   (linear in g)
    v_t   = beta2 * v_{t-1} + (1 - beta2) * g_t^2
            (NOT decomposable per bucket; shared)
    Delta W_t = -lr * m_hat_t / (sqrt(v_hat_t) + eps)
              = sum_b [ -lr * m_hat_t^b / (sqrt(v_hat_t) + eps) ]
              = sum_b Delta W_t^b
    shadow_b += Delta W_t^b

So sum_b shadow_b == sum_t Delta W_t exactly — the per-bucket shadows
additively decompose the actual W trajectory under AdamW.  Weight decay
(W -= lr * wd * W) is W's self-shrinkage and is NOT attributed to any
bucket; it's the same background force regardless of which bucket
contributed gradients.

After training, ||shadow_b||_2 (global L2 over all params) is bucket b's
cumulative push to W via the AdamW update path.
"""

import json
import logging
import math
import time
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler

from jl.double_descent.transformer.bleu import compute_bleu
from jl.double_descent.transformer.transformer_config import TDDConfig
from jl.double_descent.transformer.transformer_data import (
    M2M100Vocab,
    MaxTokensBatchSampler,
    TranslationDataset,
    Vocab,
    load_m2m100_iwslt14_variance_split,
)
from jl.double_descent.transformer.transformer_model import TransformerModel, count_parameters

logger = logging.getLogger(__name__)


BUCKET_PAD = -1  # sentinel: pad / non-aligned position


# ---------------------------------------------------------------------------
# Bucket pre-computation
# ---------------------------------------------------------------------------


def compute_bucket_id_sequences(
    train_dataset,
    oracle_path: str,
    cutoff_quantile: Optional[float] = None,
    cutoff_value: Optional[float] = None,
) -> Tuple[List[List[int]], np.ndarray, np.ndarray]:
    """For each sentence in train_dataset, build a per-token bucket-id list
    aligned to tgt_encoded[1:] (i.e., the prediction targets).

    Either `cutoff_quantile` (compute the cutoff as that quantile of this
    oracle's entropies) or `cutoff_value` (use a precomputed cutoff — e.g.
    the train-side threshold, applied to a test oracle) must be supplied.

    Two buckets:
      - bucket 0 = entropy <= cutoff (low entropy, easy / predictable)
      - bucket 1 = entropy >  cutoff (high entropy, surprising)
    """
    if (cutoff_quantile is None) == (cutoff_value is None):
        raise ValueError("Provide exactly one of cutoff_quantile / cutoff_value.")

    oracle = torch.load(oracle_path, map_location="cpu", weights_only=False)
    o_targets = oracle["target_ids"].long().numpy()
    o_log_p = oracle["log_probs"].float().numpy()
    o_offsets = oracle["sentence_offsets"].long().numpy()

    n_sent_oracle = len(o_offsets) - 1
    n_sent_train = len(train_dataset)
    if n_sent_oracle != n_sent_train:
        raise ValueError(
            f"Oracle has {n_sent_oracle} sentences but dataset has "
            f"{n_sent_train}. They must come from the same subsample."
        )

    entropy_full = -o_log_p  # nats
    cutoff = (
        float(np.quantile(entropy_full, cutoff_quantile))
        if cutoff_value is None else float(cutoff_value)
    )
    edges = np.array([entropy_full.min(), cutoff, entropy_full.max()])
    n_bins = 2
    flat_buckets = np.clip(
        np.digitize(entropy_full, edges[1:-1]), 0, n_bins - 1,
    ).astype(np.int8)

    centers = np.array([
        entropy_full[flat_buckets == b].mean() if (flat_buckets == b).any() else float("nan")
        for b in range(n_bins)
    ])

    bucket_id_sequences: List[List[int]] = []
    n_mismatch = 0
    for s in range(n_sent_train):
        tgt_encoded = train_dataset.tgt_encoded[s]   # [bos] + ids + [eos]
        # Prediction targets are tgt_encoded[1:]: ids + [eos]
        train_targets = np.asarray(tgt_encoded[1:], dtype=np.int64)
        oa, ob = int(o_offsets[s]), int(o_offsets[s + 1])
        oracle_targets = o_targets[oa:ob]
        if len(train_targets) != len(oracle_targets) or not np.array_equal(
            train_targets, oracle_targets
        ):
            n_mismatch += 1
            # Mark whole sentence as no-bucket so it never enters L_b sums.
            bucket_id_sequences.append([BUCKET_PAD] * len(train_targets))
            continue
        bucket_id_sequences.append(flat_buckets[oa:ob].tolist())

    if n_mismatch > 0:
        logger.warning(
            f"Bucket alignment: {n_mismatch}/{n_sent_train} sentences mismatch "
            f"oracle targets (skipped)."
        )
    return bucket_id_sequences, edges, centers


# ---------------------------------------------------------------------------
# Dataset wrapper that also yields the bucket-id sequence
# ---------------------------------------------------------------------------


class BucketAugmentedDataset(Dataset):
    """Wraps a TranslationDataset / M2M100TranslationDataset and tacks on
    a per-token bucket-id sequence aligned with tgt_encoded[1:]."""

    def __init__(self, base_dataset, bucket_id_sequences: List[List[int]]):
        assert len(base_dataset) == len(bucket_id_sequences)
        self.base = base_dataset
        self.bucket_id_sequences = bucket_id_sequences

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        src, tgt = self.base[idx]
        bid = self.bucket_id_sequences[idx]
        return src, tgt, bid

    def get_lengths(self) -> List[int]:
        return self.base.get_lengths()


def bucket_collate_fn(
    batch: List[Tuple[List[int], List[int], List[int]]],
    pad_idx: int,
    bucket_pad: int = BUCKET_PAD,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collate (src, tgt, bucket_ids) tuples with padding.

    bucket_ids has length len(tgt) - 1 (predictions over tgt[1:]).  After
    target padding, bucket_ids is also padded to max(tgt_len) - 1 with
    `bucket_pad`.
    """
    max_src_len = max(len(s) for s, _, _ in batch)
    max_tgt_len = max(len(t) for _, t, _ in batch)
    max_bid_len = max_tgt_len - 1

    src_padded, tgt_padded, bid_padded = [], [], []
    for src, tgt, bid in batch:
        src_padded.append(src + [pad_idx] * (max_src_len - len(src)))
        tgt_padded.append(tgt + [pad_idx] * (max_tgt_len - len(tgt)))
        # bid is aligned to tgt[1:] of length len(tgt)-1.  Pad to max_bid_len.
        pad_n = max_bid_len - len(bid)
        bid_padded.append(list(bid) + [bucket_pad] * pad_n)

    return (
        torch.tensor(src_padded, dtype=torch.long),
        torch.tensor(tgt_padded, dtype=torch.long),
        torch.tensor(bid_padded, dtype=torch.long),
    )


# ---------------------------------------------------------------------------
# Shadow weight bookkeeping
# ---------------------------------------------------------------------------


def init_shadows(model: nn.Module, n_bins: int, device: torch.device):
    """Allocate n_bins zero-init shadow tensors per trainable parameter.

    Returns a list of length n_bins; each element is a list of zero
    tensors, one per param, in the same order as model.parameters().
    """
    shadows = []
    for _ in range(n_bins):
        shadows.append([
            torch.zeros_like(p.detach(), device=device) for p in model.parameters()
        ])
    return shadows


def shadow_global_l2(shadow_list) -> float:
    """Sum of squared-L2 across all params for one bucket's shadow (sqrt'd)."""
    total = 0.0
    for s in shadow_list:
        total += s.pow(2).sum().item()
    return math.sqrt(total)


# ---------------------------------------------------------------------------
# Evaluation (matches trainer.evaluate but on TranslationDataset)
# ---------------------------------------------------------------------------


def evaluate(model, dataset, vocab, device, criterion, batch_size=64):
    from jl.double_descent.transformer.transformer_data import collate_fn
    model.eval()
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=partial(collate_fn, pad_idx=vocab.pad_idx),
    )
    total_loss, total_correct, total_tokens = 0.0, 0, 0
    with torch.no_grad():
        for src, tgt in loader:
            src, tgt = src.to(device), tgt.to(device)
            logits = model(src, tgt[:, :-1])
            target = tgt[:, 1:].contiguous()
            loss = criterion(
                logits.view(-1, logits.size(-1)),
                target.view(-1),
            )
            mask = target != vocab.pad_idx
            predictions = logits.argmax(dim=-1)
            correct = ((predictions == target) & mask).sum().item()
            total_loss += loss.item() * mask.sum().item()
            total_correct += correct
            total_tokens += mask.sum().item()
    model.train()
    if total_tokens == 0:
        return 0.0, 0.0
    return total_loss / total_tokens, total_correct / total_tokens


def log_metrics(output_path, d_model, output_suffix, metrics):
    f = Path(output_path) / f"metrics_d{d_model}_{output_suffix}.jsonl"
    with open(f, "a") as fh:
        fh.write(json.dumps(metrics) + "\n")


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------


def train_single_model_bucket_shadow(
    gpu_id: int,
    d_model: int,
    train_samples: int,
    config: TDDConfig,
    output_path: str,
    data_path: str,
    oracle_path: str,
    cutoff_quantile: float = 0.85,
) -> None:
    """Train a single Transformer with d_model embedding dim, while
    tracking per-bucket shadow gradients.

    Saves per-step bucket-share metrics + a final
    bucket_shadows_d{d}_{samples_k}k.pt with the n_bins shadow weight
    sets and a bucket_shares.json summarising the global-L2 shares.
    """
    device = torch.device(f"cuda:{gpu_id}")

    samples_k = train_samples // 1000
    output_suffix = f"{samples_k}k"
    log_name = f"shadow_d{d_model}_{samples_k}k"
    process_logger = logging.getLogger(log_name)
    process_logger.setLevel(logging.INFO)

    process_logger.info(
        f"[bucket-shadow] d_model={d_model}, {samples_k}K samples on GPU {gpu_id}"
    )

    # subsample_seed locked to 42 (TDDConfig) so train alignment with the
    # M2M100 oracle (train_split0_log_probs.pt) is preserved. Test set is a
    # held-out chunk carved from IWSLT train (indices
    # [train_samples : train_samples + 6750] of the seed=42 shuffle) so it
    # shares the train distribution — no IWSLT-test domain shift.
    from jl.double_descent.transformer.transformer_data import (
        load_m2m100_iwslt14_train_chunk_test,
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
        f"Loaded data: {len(train_dataset)} train / {len(valid_dataset)} valid / "
        f"{len(test_dataset)} test;  vocab={len(vocab)}"
    )

    n_bins = 2
    bucket_id_sequences, edges, centers = compute_bucket_id_sequences(
        train_dataset, oracle_path, cutoff_quantile=cutoff_quantile,
    )
    process_logger.info(
        f"Bucket entropy edges (nats): {[round(float(x), 4) for x in edges]}"
    )
    process_logger.info(
        f"Bucket entropy centers (nats): {[round(float(x), 4) for x in centers]}"
    )

    bucket_dataset = BucketAugmentedDataset(train_dataset, bucket_id_sequences)

    model = TransformerModel(
        vocab_size=len(vocab),
        d_model=d_model,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        d_ff_multiplier=config.d_ff_multiplier,
        pad_idx=vocab.pad_idx,
    ).to(device)
    process_logger.info(f"Model parameters: {count_parameters(model):,}")

    # Manual AdamW so we can keep per-bucket first moments.  Hyperparameters
    # match the canonical trainer: betas=(0.9, 0.98), eps=1e-9, wd=0.01.
    BETA1, BETA2 = 0.9, 0.98
    ADAM_EPS = 1e-9
    WEIGHT_DECAY = 0.01

    def lr_at(step: int) -> float:
        if step < config.warmup_steps:
            return config.learning_rate * step / config.warmup_steps
        progress = (step - config.warmup_steps) / max(
            1, config.max_steps - config.warmup_steps
        )
        return config.learning_rate * 0.5 * (1 + math.cos(math.pi * progress))

    criterion = nn.CrossEntropyLoss(
        ignore_index=vocab.pad_idx,
        label_smoothing=config.label_smoothing or 0.0,
    )

    # Per-bucket first moments + shadows (10x model size each).  Shared v.
    params = list(model.parameters())
    adam_m_per_bucket = [
        [torch.zeros_like(p, device=device) for p in params] for _ in range(n_bins)
    ]
    adam_v = [torch.zeros_like(p, device=device) for p in params]
    shadows = init_shadows(model, n_bins, device)
    process_logger.info(
        f"Allocated {n_bins} per-bucket m + {n_bins} shadow tensors "
        f"+ shared v ({2 * n_bins + 1}x model size)."
    )

    train_sampler = MaxTokensBatchSampler(
        bucket_dataset, max_tokens=config.max_tokens,
        shuffle=True, seed=config.subsample_seed,
    )
    train_loader = DataLoader(
        bucket_dataset,
        batch_sampler=train_sampler,
        collate_fn=partial(bucket_collate_fn, pad_idx=vocab.pad_idx),
    )

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

    pad_idx = vocab.pad_idx

    while step < config.max_steps:
        train_sampler.set_epoch(epoch)
        epoch += 1

        for src, tgt, bids in train_loader:
            src = src.to(device, non_blocking=True)
            tgt = tgt.to(device, non_blocking=True)
            bids = bids.to(device, non_blocking=True)

            logits = model(src, tgt[:, :-1])
            target = tgt[:, 1:].contiguous()
            B, Tm1, V = logits.shape

            per_token_loss = F.cross_entropy(
                logits.reshape(-1, V),
                target.reshape(-1),
                reduction="none",
                ignore_index=pad_idx,
                label_smoothing=config.label_smoothing or 0.0,
            )  # [B*(T-1)]
            mask_active = (target.reshape(-1) != pad_idx)
            n_active = mask_active.sum().clamp_min(1).float()
            bids_flat = bids.reshape(-1)

            current_lr = lr_at(step)

            # 10 backwards.  In the same loop:
            #   (1) update per-bucket m_b in place: m_b *= beta1; m_b += (1-beta1)*g_b
            #   (2) accumulate total_grad for the shared v update
            total_grad = [None] * len(params)
            buckets_seen = [False] * n_bins
            for b in range(n_bins):
                mask_b = (bids_flat == b)
                if not torch.any(mask_b):
                    # m_b decays even when bucket b has no tokens this step
                    with torch.no_grad():
                        for p_idx in range(len(params)):
                            adam_m_per_bucket[b][p_idx].mul_(BETA1)
                    continue
                buckets_seen[b] = True
                L_b = (per_token_loss * mask_b.float()).sum() / n_active
                grads_b = torch.autograd.grad(
                    L_b, params,
                    retain_graph=(b < n_bins - 1),
                    allow_unused=True,
                )
                with torch.no_grad():
                    for p_idx, g in enumerate(grads_b):
                        if g is None:
                            adam_m_per_bucket[b][p_idx].mul_(BETA1)
                            continue
                        # m_b ← beta1 * m_b + (1-beta1) * g_b
                        adam_m_per_bucket[b][p_idx].mul_(BETA1).add_(g, alpha=(1.0 - BETA1))
                        if total_grad[p_idx] is None:
                            total_grad[p_idx] = g.detach().clone()
                        else:
                            total_grad[p_idx].add_(g)
                del grads_b  # free per-bucket grads before next backward

            step += 1
            bias_correction1 = 1.0 - BETA1 ** step
            bias_correction2 = 1.0 - BETA2 ** step

            # Shared v + per-bucket update + W step.  Order matches
            # torch.optim.AdamW: weight decay FIRST (decoupled), then m/v update.
            with torch.no_grad():
                for p_idx, p in enumerate(params):
                    # Weight decay (W -= lr * wd * W), NOT attributed to any bucket.
                    p.mul_(1.0 - current_lr * WEIGHT_DECAY)

                    g_total = total_grad[p_idx]
                    if g_total is None:
                        # No bucket contributed a gradient to this param this step.
                        # v decays.  m_b's already decayed above.  No m/v W update.
                        adam_v[p_idx].mul_(BETA2)
                        continue
                    # v ← beta2 * v + (1-beta2) * g_total^2
                    adam_v[p_idx].mul_(BETA2).addcmul_(g_total, g_total, value=(1.0 - BETA2))
                    # denom = sqrt(v_hat) + eps
                    denom = (adam_v[p_idx] / bias_correction2).sqrt_().add_(ADAM_EPS)
                    total_update = torch.zeros_like(p)
                    for b in range(n_bins):
                        # Per-bucket update: -lr * m_hat_b / denom
                        m_hat_b = adam_m_per_bucket[b][p_idx] / bias_correction1
                        update_b = -current_lr * m_hat_b / denom
                        shadows[b][p_idx].add_(update_b)
                        total_update.add_(update_b)
                    p.add_(total_update)
                    del denom, total_update
            del total_grad

            if step % config.log_interval == 0:
                # batch-level training accuracy
                with torch.no_grad():
                    preds = logits.argmax(dim=-1)
                    mask = target != pad_idx
                    train_acc = ((preds == target) & mask).sum().item() / max(
                        1, mask.sum().item()
                    )
                # current shadow shares (for monitoring trajectory)
                with torch.no_grad():
                    shadow_norms_sq = [
                        sum(s.pow(2).sum().item() for s in shadows[b])
                        for b in range(n_bins)
                    ]
                shadow_norms = [math.sqrt(v) for v in shadow_norms_sq]
                tot = sum(shadow_norms) or 1.0
                shadow_shares = [v / tot for v in shadow_norms]

                # Compute mean batch loss for logging
                batch_loss = (per_token_loss * mask_active.float()).sum().item() / max(
                    1, mask_active.sum().item()
                )

                metrics = {
                    "step": step,
                    "d_model": d_model,
                    "train_samples": train_samples,
                    "train_loss": batch_loss,
                    "train_acc": train_acc,
                    "lr": current_lr,
                    "shadow_norms": shadow_norms,
                    "shadow_shares": shadow_shares,
                }
                if step % config.eval_interval == 0:
                    valid_loss, valid_acc = evaluate(
                        model, valid_dataset, vocab, device, criterion,
                    )
                    metrics["valid_loss"] = valid_loss
                    metrics["valid_acc"] = valid_acc
                    if valid_loss < best_valid_loss:
                        best_valid_loss = valid_loss
                        best_valid_step = step
                        torch.save(model.state_dict(), es_model_path)
                log_metrics(output_path, d_model, output_suffix, metrics)

                elapsed = time.time() - train_start
                process_logger.info(
                    f"[d={d_model}] step {step}/{config.max_steps} "
                    f"loss={batch_loss:.4f} lr={current_lr:.6f} "
                    f"shares={[round(x,3) for x in shadow_shares]} "
                    f"({elapsed:.0f}s)"
                )
            if step >= config.max_steps:
                break

    # ---- final evaluation + saving ----
    process_logger.info(f"[d={d_model}] running final evaluation ...")
    train_loss, train_acc = evaluate(model, train_dataset, vocab, device, criterion)
    valid_loss, valid_acc = evaluate(model, valid_dataset, vocab, device, criterion)
    test_loss, test_acc = evaluate(model, test_dataset, vocab, device, criterion)

    process_logger.info(f"[d={d_model}] computing BLEU ...")
    train_bleu = compute_bleu(model, train_dataset, vocab, device, max_len=128)
    test_bleu = compute_bleu(model, test_dataset, vocab, device, max_len=128)

    final_metrics = {
        "step": step,
        "d_model": d_model,
        "train_samples": train_samples,
        "train_loss": train_loss,
        "train_acc": train_acc,
        "valid_loss": valid_loss,
        "valid_acc": valid_acc,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "train_bleu": train_bleu,
        "test_bleu": test_bleu,
        "lr": lr_at(step),
    }
    log_metrics(output_path, d_model, output_suffix, final_metrics)

    # --- save shadow state + per-bucket shares ---
    shadow_norms_sq = [
        sum(s.pow(2).sum().item() for s in shadows[b]) for b in range(n_bins)
    ]
    shadow_norms = [math.sqrt(v) for v in shadow_norms_sq]
    tot = sum(shadow_norms) or 1.0
    shadow_shares = [v / tot for v in shadow_norms]

    bucket_summary = {
        "d_model": d_model,
        "train_samples": train_samples,
        "n_bins": n_bins,
        "bucket_entropy_edges": [float(e) for e in edges],
        "bucket_entropy_centers": [float(c) for c in centers],
        "shadow_norms": shadow_norms,
        "shadow_shares": shadow_shares,
        "best_valid_loss": best_valid_loss,
        "best_valid_step": best_valid_step,
    }
    summary_path = Path(output_path) / f"bucket_shares_d{d_model}_{output_suffix}.json"
    summary_path.write_text(json.dumps(bucket_summary, indent=2))
    process_logger.info(f"[d={d_model}] wrote {summary_path}")

    shadow_path = Path(output_path) / f"bucket_shadows_d{d_model}_{output_suffix}.pt"
    torch.save(
        {
            "shadows": [[s.detach().cpu() for s in shadows[b]] for b in range(n_bins)],
            "param_names": [n for n, _ in model.named_parameters()],
            "bucket_entropy_edges": edges.tolist(),
            "bucket_entropy_centers": centers.tolist(),
        },
        shadow_path,
    )
    process_logger.info(f"[d={d_model}] wrote {shadow_path} ({shadow_path.stat().st_size/1e6:.0f} MB)")

    model_path = Path(output_path) / f"model_d{d_model}_{output_suffix}.pt"
    torch.save(model.state_dict(), model_path)
    process_logger.info(f"[d={d_model}] saved final model to {model_path}")

    total_time = time.time() - train_start
    process_logger.info(
        f"[d={d_model}] DONE: test_loss={test_loss:.4f} test_bleu={test_bleu:.2f}  "
        f"shares={[round(x,3) for x in shadow_shares]}  "
        f"total {total_time/3600:.2f}h"
    )
