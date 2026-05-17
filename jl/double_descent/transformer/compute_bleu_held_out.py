#!/usr/bin/env python3
"""BLEU compute for a directory of transformer checkpoints, evaluated on
the held-out train chunk (NOT the IWSLT test split).

For each model_d{D}_{Nk}.pt in --model-dir, loads the model, runs greedy
generation under BF16 autocast on the held-out chunk
(indices [train_samples : train_samples + holdout_samples] of the
seed=42 shuffle — matches the loss / re-eval pipeline so test sets are
self-consistent), and writes a per-d_model jsonl line with the BLEU
score.

MPS-packed: bin-packs all (d_model) jobs across visible GPUs in waves
of --max-concurrent-per-gpu. Each worker process owns one (d_model, GPU)
pair and writes its own jsonl line; a final merge step concatenates them.

Usage:
    CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps \
    python -m jl.double_descent.transformer.compute_bleu_held_out \
        --data-path ./data/iwslt14.m2m100.de-en \
        --model-dir ./output/transformer_baseline_seed42/05-17-0454 \
        --output-path ./output/bleu_held_out/baseline_bleu.jsonl \
        --max-concurrent-per-gpu 6
"""

import argparse
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import List, Tuple

import torch
import torch.multiprocessing as mp

from jl.double_descent.transformer.bleu import compute_bleu
from jl.double_descent.transformer.transformer_data import (
    M2M100Vocab, load_m2m100_iwslt14_train_chunk_test,
)
from jl.double_descent.transformer.transformer_model import TransformerModel


logger = logging.getLogger(__name__)
MODEL_RE = re.compile(r"model_d(\d+)_(\d+)k\.pt$")


# ---------------------------------------------------------------------------
# Worker (one (d_model, GPU) per process)
# ---------------------------------------------------------------------------


def bleu_one_model(
    gpu_id: int,
    d_model: int,
    samples_k: int,
    data_path: str,
    model_dir: str,
    output_dir: str,
    n_layers: int,
    n_heads: int,
    d_ff_multiplier: int,
    train_samples: int,
    holdout_test_samples: int,
    subsample_seed: int,
    max_len: int,
    batch_size: int,
) -> None:
    """Compute BLEU on the held-out chunk for one trained model. Each
    process writes a separate JSON file (no shared-state contention).
    """
    device = torch.device(f"cuda:{gpu_id}")
    log_name = f"bleu_d{d_model}_{samples_k}k"
    logger_p = logging.getLogger(log_name)
    logger_p.setLevel(logging.INFO)
    if not logger_p.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter(
            "%(asctime)s [%(name)s] %(message)s", datefmt="%H:%M:%S"
        ))
        logger_p.addHandler(h)
    logger_p.info(f"start d={d_model} on GPU {gpu_id}")
    t0 = time.time()

    vocab = M2M100Vocab(str(Path(data_path) / "vocab_mapping.json"))
    _, _, test_dataset, _ = load_m2m100_iwslt14_train_chunk_test(
        data_path,
        train_samples=train_samples,
        holdout_test_samples=holdout_test_samples,
        subsample_seed=subsample_seed,
    )

    model = TransformerModel(
        vocab_size=len(vocab), d_model=d_model,
        n_layers=n_layers, n_heads=n_heads,
        d_ff_multiplier=d_ff_multiplier,
        pad_idx=vocab.pad_idx,
    ).to(device)
    model_path = Path(model_dir) / f"model_d{d_model}_{samples_k}k.pt"
    state = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state)

    bleu_score = compute_bleu(
        model, test_dataset, vocab, device,
        max_len=max_len, batch_size=batch_size, use_bf16=True,
    )

    record = {
        "d_model": d_model,
        "train_samples": train_samples,
        "holdout_test_samples": holdout_test_samples,
        "subsample_seed": subsample_seed,
        "n_test_sentences": len(test_dataset),
        "max_len": max_len,
        "test_bleu": float(bleu_score),
    }
    per_d_path = Path(output_dir) / f"bleu_d{d_model}_{samples_k}k.json"
    per_d_path.write_text(json.dumps(record, indent=2))

    logger_p.info(
        f"d={d_model} BLEU={bleu_score:.4f} ({time.time() - t0:.1f}s) "
        f"-> {per_d_path.name}"
    )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def discover_models(model_dir: str) -> List[Tuple[int, int]]:
    pairs = []
    for f in sorted(Path(model_dir).glob("model_d*_*k.pt")):
        m = MODEL_RE.search(f.name)
        if not m:
            continue
        pairs.append((int(m.group(1)), int(m.group(2))))
    return pairs


def bin_pack(jobs: List[Tuple[int, int]], num_gpus: int) -> List[List[Tuple[int, int]]]:
    """Greedy bin-pack jobs by d_model descending → lightest GPU first."""
    gpu_jobs = [[] for _ in range(num_gpus)]
    gpu_load = [0] * num_gpus
    for d_model, samples_k in sorted(jobs, key=lambda p: p[0], reverse=True):
        g = min(range(num_gpus), key=lambda i: gpu_load[i])
        gpu_jobs[g].append((d_model, samples_k))
        gpu_load[g] += d_model
    return gpu_jobs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-path", required=True)
    p.add_argument("--model-dir", required=True,
                   help="Directory holding model_d{D}_{Nk}.pt checkpoints.")
    p.add_argument("--output-path", required=True,
                   help="Path to the merged jsonl written at the end.")
    p.add_argument("--max-concurrent-per-gpu", type=int, default=6)
    p.add_argument("--train-samples", type=int, default=36000)
    p.add_argument("--holdout-samples", type=int, default=6750)
    p.add_argument("--subsample-seed", type=int, default=42)
    p.add_argument("--max-len", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--n-layers", type=int, default=6)
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--d-ff-multiplier", type=int, default=4)
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    num_gpus = torch.cuda.device_count()
    if num_gpus < 1:
        raise RuntimeError("No CUDA GPUs visible.")

    jobs = discover_models(args.model_dir)
    if not jobs:
        raise FileNotFoundError(f"No model_d*_*k.pt under {args.model_dir}")

    gpu_jobs = bin_pack(jobs, num_gpus)
    max_per_gpu = max(len(j) for j in gpu_jobs)
    max_concurrent = max(1, args.max_concurrent_per_gpu)
    num_waves = (max_per_gpu + max_concurrent - 1) // max_concurrent

    logger.info(f"Models to score: {len(jobs)}")
    logger.info(f"GPUs visible: {num_gpus}; max jobs/GPU = {max_per_gpu}; "
                f"max-concurrent/GPU = {max_concurrent}; waves = {num_waves}")
    for gid, jjs in enumerate(gpu_jobs):
        logger.info(f"  GPU {gid} -> {[d for d, _ in jjs]}")
    logger.info(
        f"Held-out chunk: seed={args.subsample_seed}, "
        f"[{args.train_samples}:{args.train_samples + args.holdout_samples}]"
    )

    # Per-d_model jsonl scratch dir alongside the final output.
    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    per_d_dir = out_path.parent / "_per_d"
    per_d_dir.mkdir(exist_ok=True)

    mp.set_start_method("spawn", force=True)
    total_start = time.time()
    for wave in range(num_waves):
        wave_start = time.time()
        procs = []
        for gpu_id, gpu_list in enumerate(gpu_jobs):
            start = wave * max_concurrent
            for d_model, samples_k in gpu_list[start:start + max_concurrent]:
                p_ = mp.Process(
                    target=bleu_one_model,
                    args=(gpu_id, d_model, samples_k, args.data_path,
                          args.model_dir, str(per_d_dir),
                          args.n_layers, args.n_heads, args.d_ff_multiplier,
                          args.train_samples, args.holdout_samples,
                          args.subsample_seed, args.max_len, args.batch_size),
                )
                p_.start()
                procs.append((gpu_id, d_model, p_))
        logger.info(f"Wave {wave + 1}/{num_waves}: {len(procs)} processes started")
        for gid, dm, p_ in procs:
            p_.join()
            if p_.exitcode != 0:
                logger.warning(f"  d={dm} on GPU {gid} exited {p_.exitcode}")
        logger.info(f"Wave {wave + 1} done in {(time.time() - wave_start) / 60:.1f}m")

    # Merge per-d jsons into the final jsonl.
    rows = []
    for f in sorted(per_d_dir.glob("bleu_d*_*k.json")):
        rows.append(json.loads(f.read_text()))
    rows.sort(key=lambda r: r["d_model"])
    with open(out_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    logger.info(f"Merged {len(rows)} rows -> {out_path}")
    logger.info(f"Total: {(time.time() - total_start) / 60:.1f}m")


if __name__ == "__main__":
    main()
