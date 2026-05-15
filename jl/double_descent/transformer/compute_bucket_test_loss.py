#!/usr/bin/env python3
"""Per-bucket test cross-entropy loss for trained transformer models.

For each saved small-transformer model_d{d}_{Nk}.pt:
  - Load test split (M2M100-tokenized).
  - Bucket each non-pad test target token by oracle entropy
    (= -log p_oracle(y) from test_log_probs.pt) using the SAME edges as
    the train-side bucketing (loaded from train_log_probs.pt).
  - Run model on test, accumulate CE loss + token count per bucket.
  - Output bucket_test_loss_d{d}_{Nk}.json with per-bucket mean CE and
    counts.

Usage:
    python -m jl.double_descent.transformer.compute_bucket_test_loss \\
        --data-path ./data/iwslt14.m2m100.de-en \\
        --train-oracle ./data/iwslt14.m2m100.de-en/train_split0_log_probs.pt \\
        --test-oracle  ./data/iwslt14.m2m100.de-en/test_log_probs.pt \\
        --model-dir ./data \\
        --output-dir ./output/bucket_test_loss \\
        --n-bins 10
"""

import argparse
import json
import logging
import re
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from jl.double_descent.transformer.transformer_data import (
    M2M100Vocab, load_m2m100_split_ids, M2M100TranslationDataset,
)
from jl.double_descent.transformer.transformer_model import TransformerModel

logger = logging.getLogger(__name__)

MODEL_RE = re.compile(r"model_d(\d+)_(\d+)k\.pt$")
BUCKET_PAD = -1


def collate(batch, pad_idx):
    src_list, tgt_list = zip(*batch)
    max_src = max(len(s) for s in src_list)
    max_tgt = max(len(t) for t in tgt_list)
    src = torch.full((len(batch), max_src), pad_idx, dtype=torch.long)
    tgt = torch.full((len(batch), max_tgt), pad_idx, dtype=torch.long)
    for i, (s, t) in enumerate(zip(src_list, tgt_list)):
        src[i, :len(s)] = torch.tensor(s, dtype=torch.long)
        tgt[i, :len(t)] = torch.tensor(t, dtype=torch.long)
    return src, tgt


def compute_train_edges(train_oracle_path: Path, n_bins: int) -> np.ndarray:
    o = torch.load(train_oracle_path, map_location="cpu", weights_only=False)
    entropy = -o["log_probs"].float().numpy()
    edges = np.quantile(entropy, np.linspace(0, 1, n_bins + 1))
    return edges


def bucket_test_tokens(
    test_oracle_path: Path,
    test_dataset: M2M100TranslationDataset,
    edges: np.ndarray,
    n_bins: int,
) -> List[List[int]]:
    """Per-sentence list of bucket-ids aligned with tgt_encoded[1:]."""
    o = torch.load(test_oracle_path, map_location="cpu", weights_only=False)
    o_targets = o["target_ids"].long().numpy()
    o_log_p = o["log_probs"].float().numpy()
    o_offsets = o["sentence_offsets"].long().numpy()

    n_sent_oracle = len(o_offsets) - 1
    n_sent_test = len(test_dataset)
    if n_sent_oracle != n_sent_test:
        raise ValueError(
            f"Oracle has {n_sent_oracle} sentences but test_dataset has {n_sent_test}."
        )

    entropy_full = -o_log_p
    flat_buckets = np.clip(
        np.digitize(entropy_full, edges[1:-1]), 0, n_bins - 1,
    ).astype(np.int16)

    bucket_id_seqs = []
    n_mismatch = 0
    for s in range(n_sent_test):
        tgt_encoded = test_dataset.tgt_encoded[s]
        train_targets = np.asarray(tgt_encoded[1:], dtype=np.int64)
        oa, ob = int(o_offsets[s]), int(o_offsets[s + 1])
        oracle_targets = o_targets[oa:ob]
        if len(train_targets) != len(oracle_targets) or not np.array_equal(
            train_targets, oracle_targets
        ):
            n_mismatch += 1
            bucket_id_seqs.append([BUCKET_PAD] * len(train_targets))
            continue
        bucket_id_seqs.append(flat_buckets[oa:ob].tolist())

    if n_mismatch:
        logger.warning(f"  {n_mismatch}/{n_sent_test} sentences mismatched oracle targets")
    return bucket_id_seqs


def compute_bucket_loss(
    model: TransformerModel,
    test_dataset: M2M100TranslationDataset,
    bucket_id_seqs: List[List[int]],
    pad_idx: int,
    n_bins: int,
    device: torch.device,
    batch_size: int = 64,
) -> dict:
    model.eval()
    loss_sum = torch.zeros(n_bins, dtype=torch.float64, device=device)
    tok_count = torch.zeros(n_bins, dtype=torch.long, device=device)

    indices = list(range(len(test_dataset)))
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            batch_idx = indices[start:start + batch_size]
            batch = [test_dataset[i] for i in batch_idx]
            src, tgt = collate(batch, pad_idx)
            src = src.to(device, non_blocking=True)
            tgt = tgt.to(device, non_blocking=True)

            logits = model(src, tgt[:, :-1])           # [B, T, V]
            target = tgt[:, 1:].contiguous()           # [B, T]
            log_p = F.log_softmax(logits.float(), dim=-1)
            ce = -log_p.gather(2, target.unsqueeze(-1)).squeeze(-1)   # [B, T]
            mask = (target != pad_idx)

            B, T = target.shape
            buckets = torch.full((B, T), BUCKET_PAD, dtype=torch.long, device=device)
            for j, i in enumerate(batch_idx):
                bid = bucket_id_seqs[i]
                length = min(len(bid), T)
                if length:
                    buckets[j, :length] = torch.tensor(
                        bid[:length], dtype=torch.long, device=device,
                    )

            valid = mask & (buckets != BUCKET_PAD)
            if not valid.any():
                continue
            ce_v = ce[valid].double()
            buc_v = buckets[valid]
            loss_sum.scatter_add_(0, buc_v, ce_v)
            tok_count.scatter_add_(0, buc_v, torch.ones_like(buc_v))

    loss_sum = loss_sum.cpu().numpy()
    tok_count = tok_count.cpu().numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_loss = loss_sum / np.where(tok_count > 0, tok_count, 1)
    mean_loss = np.where(tok_count > 0, mean_loss, np.nan)

    return {
        "bucket_test_loss": [float(x) if not np.isnan(x) else None for x in mean_loss],
        "bucket_test_tokens": [int(x) for x in tok_count],
        "total_tokens": int(tok_count.sum()),
        "total_loss": float(loss_sum.sum()),
        "mean_loss_overall": float(loss_sum.sum() / max(1, tok_count.sum())),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-path", required=True)
    p.add_argument("--train-oracle", required=True)
    p.add_argument("--test-oracle", required=True)
    p.add_argument("--model-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--n-bins", type=int, default=10)
    p.add_argument("--n-layers", type=int, default=6)
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--d-ff-multiplier", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(name)s] %(message)s",
                        datefmt="%H:%M:%S")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device) if args.device else (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )
    logger.info(f"Device: {device}")

    vocab = M2M100Vocab(str(Path(args.data_path) / "vocab_mapping.json"))
    test_src, test_tgt = load_m2m100_split_ids(args.data_path, "test")
    test_dataset = M2M100TranslationDataset(test_src, test_tgt, vocab)
    logger.info(f"Test sentences: {len(test_dataset)}")

    edges = compute_train_edges(Path(args.train_oracle), args.n_bins)
    logger.info(f"Train edges (nats): {[round(float(e), 4) for e in edges]}")

    bucket_id_seqs = bucket_test_tokens(
        Path(args.test_oracle), test_dataset, edges, args.n_bins,
    )
    total_test_tokens = sum(
        sum(1 for b in seq if b != BUCKET_PAD) for seq in bucket_id_seqs
    )
    logger.info(f"Bucketed {total_test_tokens} test target tokens")

    model_files = sorted(Path(args.model_dir).glob("model_d*_*k.pt"))
    if not model_files:
        raise FileNotFoundError(f"No model_d*_*k.pt under {args.model_dir}")

    for mf in model_files:
        m = MODEL_RE.search(mf.name)
        if not m:
            continue
        d_model = int(m.group(1))
        samples_k = int(m.group(2))
        logger.info(f"--- d={d_model}, {samples_k}k ---")

        model = TransformerModel(
            vocab_size=len(vocab),
            d_model=d_model,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            d_ff_multiplier=args.d_ff_multiplier,
            pad_idx=vocab.pad_idx,
        ).to(device)
        state = torch.load(mf, map_location=device, weights_only=True)
        model.load_state_dict(state)

        result = compute_bucket_loss(
            model, test_dataset, bucket_id_seqs,
            vocab.pad_idx, args.n_bins, device, batch_size=args.batch_size,
        )
        result["d_model"] = d_model
        result["train_samples"] = samples_k * 1000
        result["n_bins"] = args.n_bins
        result["bucket_entropy_edges"] = [float(e) for e in edges]

        logger.info(
            f"  overall test CE = {result['mean_loss_overall']:.4f}; "
            f"per-bucket = {[round(x, 3) if x is not None else None for x in result['bucket_test_loss']]}"
        )
        logger.info(f"  bucket counts = {result['bucket_test_tokens']}")

        out_path = out / f"bucket_test_loss_d{d_model}_{samples_k}k.json"
        out_path.write_text(json.dumps(result, indent=2))
        logger.info(f"  wrote {out_path}")

        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
