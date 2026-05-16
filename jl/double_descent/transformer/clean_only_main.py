#!/usr/bin/env python3
"""Standalone entry point: train transformers with high-entropy tokens masked.

Companion to transformer_main.py --track-shadows. Trains the same widths
under the same recipe (TDDConfig, GPU-resident batches, AdamW warmup +
cosine), but per-token CE loss is masked at tokens whose oracle entropy
exceeds the cutoff (default 85th percentile). The high-entropy tokens
still appear in the decoder context — only their prediction is excluded
from the gradient.

Saves model + ES checkpoint only — no shadow tracking.

Usage:
    python -m jl.double_descent.transformer.clean_only_main \
        --output-path ./output/transformer_clean/$(date +%m-%d-%H%M) \
        --data-path ./data/iwslt14.m2m100.de-en \
        --d-models 32 112 360
"""

import argparse
import logging
import os
import time

import torch
import torch.multiprocessing as mp

from jl.double_descent.transformer.transformer_config import TDDConfig
from jl.double_descent.transformer.clean_only_trainer import (
    train_single_model_clean_only,
)

TRAIN_SAMPLES = [36000]
D_MODEL_VALUES = list(range(8, 392, 8))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Transformer clean-only training (mask loss at high-entropy tokens)"
    )
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument(
        "--data-path", type=str, default="./data/iwslt14.m2m100.de-en",
    )
    parser.add_argument(
        "--oracle-path", type=str,
        default="./data/iwslt14.m2m100.de-en/train_split0_log_probs.pt",
        help="M2M100 oracle log-probs file aligned with the train subsample.",
    )
    parser.add_argument(
        "--test-oracle-path", type=str,
        default="./data/iwslt14.m2m100.de-en/test_chunk_log_probs.pt",
        help="M2M100 oracle log-probs for the held-out test chunk; enables "
             "test-side high-entropy masking. Pass empty string to disable.",
    )
    parser.add_argument(
        "--cutoff-quantile", type=float, default=0.85,
        help="Train-side entropy quantile above which tokens are masked out.",
    )
    parser.add_argument(
        "--d-models", type=int, nargs="+", default=None,
        help="Subset of d_model values to train (default: full 8..384).",
    )
    parser.add_argument(
        "--max-concurrent-per-gpu", type=int, default=16,
        help="Cap on concurrent processes per GPU (same as transformer_main).",
    )
    return parser.parse_args()


def run(args, config, d_models, num_gpus):
    total_start = time.time()
    train_samples = TRAIN_SAMPLES[0]
    samples_k = train_samples // 1000

    # Round-robin assign jobs to GPUs so each GPU gets a mix of d_models.
    jobs = [(d, None) for d in d_models]
    gpu_to_jobs = [[] for _ in range(num_gpus)]
    for i, job in enumerate(jobs):
        gpu_to_jobs[i % num_gpus].append(job)
    max_per_gpu = max(len(j) for j in gpu_to_jobs)
    max_concurrent = max(1, args.max_concurrent_per_gpu)
    num_waves = (max_per_gpu + max_concurrent - 1) // max_concurrent

    logger.info(f"Clean-only sweep: d_models={d_models}")
    logger.info(f"GPUs: {num_gpus}, max jobs/GPU: {max_per_gpu}, waves: {num_waves}")
    for gpu_id, jjs in enumerate(gpu_to_jobs):
        logger.info(f"  GPU {gpu_id} -> {jjs}")
    logger.info(f"Cutoff quantile: {args.cutoff_quantile}; oracle: {args.oracle_path}")

    mp.set_start_method('spawn', force=True)
    for wave in range(num_waves):
        wave_start = time.time()
        wave_procs = []
        for gpu_id, gpu_jobs in enumerate(gpu_to_jobs):
            start = wave * max_concurrent
            end = start + max_concurrent
            for d, _ in gpu_jobs[start:end]:
                p = mp.Process(
                    target=train_single_model_clean_only,
                    args=(gpu_id, d, train_samples, config,
                          args.output_path, args.data_path,
                          args.oracle_path),
                    kwargs={
                        "cutoff_quantile": args.cutoff_quantile,
                        "test_oracle_path": args.test_oracle_path or None,
                    },
                )
                p.start()
                wave_procs.append((gpu_id, f"d{d}", p))
        logger.info(
            f"Wave {wave+1}/{num_waves}: started {len(wave_procs)} processes"
        )
        for gpu_id, label, p in wave_procs:
            p.join()
            if p.exitcode != 0:
                logger.warning(
                    f"  Process {label} (GPU {gpu_id}) exit code {p.exitcode}"
                )
        logger.info(f"Wave {wave+1} done in {(time.time()-wave_start)/60:.1f}m")

    total = time.time() - total_start
    logger.info(f"All clean-only done. Total: {total/3600:.2f}h")
    logger.info(f"Metrics: {args.output_path}/metrics_d*_{samples_k}k.jsonl")


def main():
    args = parse_args()
    config = TDDConfig()
    num_gpus = torch.cuda.device_count()
    if num_gpus < 1:
        raise RuntimeError("No CUDA GPUs visible.")
    if not os.path.isfile(args.oracle_path):
        raise FileNotFoundError(
            f"Oracle file not found at {args.oracle_path}. Generate via "
            f"`python -m jl.double_descent.transformer.extract_m2m100_reference "
            f"--data-path {args.data_path} --output-path {args.oracle_path} "
            f"--split train --variance-split-id 0`."
        )
    d_models = args.d_models if args.d_models else D_MODEL_VALUES
    os.makedirs(args.output_path, exist_ok=True)
    run(args, config, d_models, num_gpus)


if __name__ == "__main__":
    main()
