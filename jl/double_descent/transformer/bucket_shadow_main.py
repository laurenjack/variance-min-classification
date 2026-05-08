#!/usr/bin/env python3
"""Bucket-shadow Transformer training entry point.

Trains a list of d_model values in parallel (one model per GPU per
batch) using the bucket-shadow trainer.  Each model produces:

    metrics_d{d}_{Nk}.jsonl       step-wise loss + shadow shares
    bucket_shares_d{d}_{Nk}.json  final bucket norm + share summary
    bucket_shadows_d{d}_{Nk}.pt   raw shadow weight tensors (n_bins lists)
    model_d{d}_{Nk}.pt            final main model

Default sweep is the 3 regimes from the support-vector hypothesis:
    d=32 (under-parameterized), d=112 (interpolation), d=360 (deep over-param)

Usage:
    python -m jl.double_descent.transformer.bucket_shadow_main \\
        --output-path ./output/shadow/$(date +%m-%d-%H%M) \\
        --data-path  ./data/iwslt14.m2m100.de-en \\
        --oracle-path ./data/iwslt14.m2m100.de-en/train_split0_log_probs.pt
"""

import argparse
import logging
import os
import time

import torch
import torch.multiprocessing as mp

from jl.double_descent.transformer.bucket_shadow_trainer import (
    train_single_model_bucket_shadow,
)
from jl.double_descent.transformer.transformer_config import TDDConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


DEFAULT_D_MODELS = [32, 112, 360]
TRAIN_SAMPLES = 36000


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-path", required=True)
    p.add_argument(
        "--data-path", default="./data/iwslt14.m2m100.de-en",
        help="Directory with M2M100-tokenized IWSLT data (vocab_mapping.json + .ids files).",
    )
    p.add_argument(
        "--oracle-path", required=True,
        help="Path to oracle log_probs .pt with sentence_offsets + target_ids "
             "matching the train subsample (e.g. train_split0_log_probs.pt).",
    )
    p.add_argument(
        "--d-models", type=int, nargs="+", default=None,
        help=f"d_model values (default: {DEFAULT_D_MODELS}).",
    )
    p.add_argument(
        "--n-bins", type=int, default=10,
        help="Number of entropy buckets (default 10).",
    )
    return p.parse_args()


def main():
    args = parse_args()
    config = TDDConfig()
    d_models = args.d_models if args.d_models else DEFAULT_D_MODELS

    num_gpus = torch.cuda.device_count()
    if num_gpus < 1:
        raise RuntimeError("No CUDA GPUs visible to this process.")

    os.makedirs(args.output_path, exist_ok=True)

    required_files = [
        "vocab_mapping.json",
        "train.de.ids", "train.en.ids",
        "valid.de.ids", "valid.en.ids",
        "test.de.ids", "test.en.ids",
    ]
    missing = [f for f in required_files if not os.path.isfile(os.path.join(args.data_path, f))]
    if missing:
        raise FileNotFoundError(
            f"Missing M2M100 data in {args.data_path}: {missing}\n"
            "Run: python -m jl.double_descent.transformer.prepare_m2m100_data"
        )

    if not os.path.isfile(args.oracle_path):
        raise FileNotFoundError(f"Oracle file not found: {args.oracle_path}")

    logger.info("Bucket-shadow Transformer training")
    logger.info(f"d_models: {d_models}")
    logger.info(f"GPUs: {num_gpus}")
    logger.info(f"Train samples: {TRAIN_SAMPLES}; n_bins: {args.n_bins}")
    logger.info(f"Output: {args.output_path}")
    logger.info(f"Oracle: {args.oracle_path}")

    total_start = time.time()
    num_batches = (len(d_models) + num_gpus - 1) // num_gpus
    for batch_idx in range(0, len(d_models), num_gpus):
        batch_d_models = d_models[batch_idx:batch_idx + num_gpus]
        batch_num = batch_idx // num_gpus + 1
        logger.info(f"\nBatch {batch_num}/{num_batches}: d_model = {batch_d_models}")

        mp.set_start_method("spawn", force=True)
        processes = []
        for gpu_id, d_model in enumerate(batch_d_models):
            p = mp.Process(
                target=train_single_model_bucket_shadow,
                args=(
                    gpu_id, d_model, TRAIN_SAMPLES, config,
                    args.output_path, args.data_path, args.oracle_path,
                    args.n_bins,
                ),
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
        logger.info(f"Batch {batch_num}/{num_batches} complete")

    total_time = time.time() - total_start
    logger.info(f"\nAll done. Total time: {total_time:.0f}s ({total_time/3600:.2f}h)")
    logger.info(
        f"Plot with:\n"
        f"  python -m jl.double_descent.transformer.plot_bucket_shadow {args.output_path} "
        f"--output-dir <dir>"
    )


if __name__ == "__main__":
    main()
