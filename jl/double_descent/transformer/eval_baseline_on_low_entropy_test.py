#!/usr/bin/env python3
"""Re-evaluate baseline transformer checkpoints on the seed=674931
held-out chunk, masked to low-entropy tokens.

The baseline was trained with TDDConfig.subsample_seed = 674931 on the
first 36k sentences of that shuffle. To stay disjoint from its training
data we carve indices [36000:42750] of the SAME shuffle as the test
chunk. We then bucket each test token by its M2M100 oracle entropy and
report mean test CE / accuracy on bucket 0 (low entropy = bottom 85%).

Cutoff is hard-coded to the clean run's value (3.0879 nats) so the
"low entropy" partition is consistent across the two runs being
compared. Both samples come from IWSLT-train distribution so the
entropy distribution should be near-identical and the partitioning
will be very close to 85:15 on the baseline chunk too.

Usage:
    python -m jl.double_descent.transformer.eval_baseline_on_low_entropy_test \
        --data-path ./data/iwslt14.m2m100.de-en \
        --test-oracle ./data/iwslt14.m2m100.de-en/test_chunk_log_probs_seed674931.pt \
        --model-dir /root/transformer_baseline_models \
        --output-dir /root/baseline_lowent_eval \
        --subsample-seed 674931
"""

import argparse
import json
import logging
import re
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from jl.double_descent.transformer.transformer_data import (
    M2M100Vocab, load_m2m100_iwslt14_train_chunk_test,
)
from jl.double_descent.transformer.transformer_model import TransformerModel
from jl.double_descent.transformer.bucket_shadow_trainer import (
    compute_bucket_id_sequences, BUCKET_PAD,
)

logger = logging.getLogger(__name__)
MODEL_RE = re.compile(r"model_d(\d+)_(\d+)k\.pt$")


def collate(batch, pad_idx):
    src_seqs, tgt_seqs = zip(*batch)
    max_src = max(len(s) for s in src_seqs)
    max_tgt = max(len(t) for t in tgt_seqs)
    src = torch.full((len(batch), max_src), pad_idx, dtype=torch.long)
    tgt = torch.full((len(batch), max_tgt), pad_idx, dtype=torch.long)
    for i, (s, t) in enumerate(zip(src_seqs, tgt_seqs)):
        src[i, :len(s)] = torch.tensor(s, dtype=torch.long)
        tgt[i, :len(t)] = torch.tensor(t, dtype=torch.long)
    return src, tgt


def eval_model(
    model, test_dataset, bucket_id_seqs, pad_idx, n_bins, device,
    batch_size=64,
):
    """Per-bucket loss + accuracy on the carved test chunk."""
    model.eval()
    loss_sum = torch.zeros(n_bins, dtype=torch.float64, device=device)
    correct = torch.zeros(n_bins, dtype=torch.long, device=device)
    tok_count = torch.zeros(n_bins, dtype=torch.long, device=device)

    indices = list(range(len(test_dataset)))
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            batch_idx = indices[start:start + batch_size]
            batch = [test_dataset[i] for i in batch_idx]
            src, tgt = collate(batch, pad_idx)
            src = src.to(device, non_blocking=True)
            tgt = tgt.to(device, non_blocking=True)
            logits = model(src, tgt[:, :-1])
            target = tgt[:, 1:].contiguous()
            log_p = F.log_softmax(logits.float(), dim=-1)
            ce = -log_p.gather(2, target.unsqueeze(-1)).squeeze(-1)
            preds = logits.argmax(dim=-1)
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
            correct_v = (preds[valid] == target[valid]).long()
            buc_v = buckets[valid]
            loss_sum.scatter_add_(0, buc_v, ce_v)
            correct.scatter_add_(0, buc_v, correct_v)
            tok_count.scatter_add_(0, buc_v, torch.ones_like(buc_v))

    loss_sum = loss_sum.cpu().numpy()
    correct = correct.cpu().numpy()
    tok_count = tok_count.cpu().numpy()
    safe = np.where(tok_count > 0, tok_count, 1)
    mean_loss = np.where(tok_count > 0, loss_sum / safe, np.nan)
    acc = np.where(tok_count > 0, correct / safe, np.nan)
    return mean_loss, acc, tok_count


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-path", required=True)
    p.add_argument("--test-oracle", required=True,
                   help="M2M100 oracle for the carved test chunk.")
    p.add_argument("--model-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--subsample-seed", type=int, default=674931,
                   help="Seed used by the baseline (default 674931).")
    p.add_argument("--train-samples", type=int, default=36000)
    p.add_argument("--holdout-samples", type=int, default=6750)
    p.add_argument("--cutoff-value", type=float, default=3.0879,
                   help="Entropy threshold (nats). Default matches the seed=42 "
                        "clean run's 85th-percentile.")
    p.add_argument("--n-layers", type=int, default=6)
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--d-ff-multiplier", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(name)s] %(message)s",
                        datefmt="%H:%M:%S")

    device = torch.device(args.device) if args.device else (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )
    logger.info(f"Device: {device}; seed={args.subsample_seed}; cutoff={args.cutoff_value}")

    train_dataset, valid_dataset, test_dataset, vocab = (
        load_m2m100_iwslt14_train_chunk_test(
            args.data_path,
            train_samples=args.train_samples,
            holdout_test_samples=args.holdout_samples,
            subsample_seed=args.subsample_seed,
        )
    )
    logger.info(
        f"Carved test chunk: {len(test_dataset)} sentences "
        f"(indices [{args.train_samples}:{args.train_samples+args.holdout_samples}] "
        f"of seed={args.subsample_seed} shuffle)"
    )

    bucket_id_seqs, edges, centers = compute_bucket_id_sequences(
        test_dataset, args.test_oracle, cutoff_value=args.cutoff_value,
    )
    n_bins = 2
    n_low = sum(sum(1 for b in seq if b == 0) for seq in bucket_id_seqs)
    n_high = sum(sum(1 for b in seq if b == 1) for seq in bucket_id_seqs)
    logger.info(
        f"Bucket sizes on test chunk: low={n_low}, high={n_high} "
        f"(low fraction = {n_low / max(1, n_low + n_high):.3f}); "
        f"cutoff = {edges[1]:.4f} nats"
    )

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    model_files = sorted(Path(args.model_dir).glob("model_d*_*k.pt"))
    if not model_files:
        raise FileNotFoundError(f"No model_d*_*k.pt under {args.model_dir}")
    logger.info(f"Re-evaluating {len(model_files)} baseline models")

    results = []
    for mf in model_files:
        m = MODEL_RE.search(mf.name)
        if not m:
            continue
        d_model = int(m.group(1))
        samples_k = int(m.group(2))
        t0 = time.time()
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

        mean_loss, acc, counts = eval_model(
            model, test_dataset, bucket_id_seqs,
            vocab.pad_idx, n_bins, device, batch_size=args.batch_size,
        )
        record = {
            "d_model": d_model,
            "train_samples": samples_k * 1000,
            "test_loss_low": float(mean_loss[0]) if not np.isnan(mean_loss[0]) else None,
            "test_loss_high": float(mean_loss[1]) if not np.isnan(mean_loss[1]) else None,
            "test_acc_low": float(acc[0]) if not np.isnan(acc[0]) else None,
            "test_acc_high": float(acc[1]) if not np.isnan(acc[1]) else None,
            "n_low": int(counts[0]),
            "n_high": int(counts[1]),
            "cutoff_value": args.cutoff_value,
            "subsample_seed": args.subsample_seed,
        }
        results.append(record)
        logger.info(
            f"d={d_model}: loss_low={record['test_loss_low']:.4f} "
            f"loss_high={record['test_loss_high']:.4f} "
            f"acc_low={record['test_acc_low']:.4f} "
            f"acc_high={record['test_acc_high']:.4f}  "
            f"({time.time() - t0:.1f}s)"
        )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    out_path = out / f"baseline_lowent_eval_seed{args.subsample_seed}.jsonl"
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    logger.info(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
