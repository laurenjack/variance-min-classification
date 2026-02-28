#!/usr/bin/env python3
"""Main entry point for Transformer Double Descent training.

Reproduces Figure 3 from Nakkiran et al. (2019) "Deep Double Descent":
6-layer encoder-decoder Transformer with varying embedding dimension d_model
trained on IWSLT'14 German-to-English translation.

Runs the FULL experiment automatically:
- Requires exactly 8 GPUs (fails fast otherwise)
- Trains all 48 models: 24 d_model values x 2 sample sizes
- 6 batches total, 8 models per batch

Usage:
    python -m jl.double_descent.transformer.transformer_main \
        --output-path ./output \
        --data-path ./data/iwslt14.tokenized.de-en
"""

import argparse
import logging
import os
import time

import torch
import torch.multiprocessing as mp

from jl.double_descent.transformer.transformer_config import TDDConfig
from jl.double_descent.transformer.trainer import train_single_model

# Configure logging with timestamps
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Hardcoded experiment parameters
TRAIN_SAMPLES = [18000, 4000]  # 18K first, then 4K
D_MODEL_VALUES = list(range(8, 200, 8))  # [8, 16, 24, ..., 192] - 24 values
REQUIRED_GPUS = 8


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train Transformer models for Deep Double Descent reproduction"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Path to save training metrics"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="./data/iwslt14.tokenized.de-en",
        help="Path to preprocessed IWSLT'14 data"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = TDDConfig()
    total_start = time.time()

    # Require exactly 8 GPUs
    num_gpus = torch.cuda.device_count()
    if num_gpus != REQUIRED_GPUS:
        raise RuntimeError(
            f"This experiment requires exactly {REQUIRED_GPUS} GPUs, "
            f"but found {num_gpus}. Please run on an 8-GPU instance."
        )

    logger.info("Transformer Double Descent - Full Experiment")
    logger.info(f"d_model values: {D_MODEL_VALUES}")
    logger.info(f"Train samples: {TRAIN_SAMPLES}")
    logger.info(f"Total models: {len(D_MODEL_VALUES) * len(TRAIN_SAMPLES)} = 48")
    logger.info(f"Batches: {len(D_MODEL_VALUES) // REQUIRED_GPUS * len(TRAIN_SAMPLES)} = 6")
    logger.info(f"Max steps: {config.max_steps}, Max tokens/batch: {config.max_tokens}")

    # Create output directory
    os.makedirs(args.output_path, exist_ok=True)

    # Verify data exists - check for required files
    required_files = ["train.de", "train.en", "valid.de", "valid.en", "test.de", "test.en", "code"]
    missing_files = [f for f in required_files if not os.path.isfile(os.path.join(args.data_path, f))]
    if missing_files:
        raise FileNotFoundError(
            f"Preprocessed IWSLT'14 data not found at {args.data_path}.\n"
            f"Missing files: {missing_files}\n\n"
            "Please run preprocessing first:\n"
            "  ./infra/prepare_iwslt14.sh\n"
        )

    # Outer loop: sample sizes (18K first, then 4K)
    for train_samples in TRAIN_SAMPLES:
        samples_k = train_samples // 1000
        logger.info(f"\n{'='*60}")
        logger.info(f"Starting {samples_k}K sample runs")
        logger.info(f"{'='*60}")

        # Inner loop: d_model batches (3 batches of 8)
        for batch_idx in range(0, len(D_MODEL_VALUES), REQUIRED_GPUS):
            batch_d_models = D_MODEL_VALUES[batch_idx:batch_idx + REQUIRED_GPUS]
            batch_num = batch_idx // REQUIRED_GPUS + 1

            logger.info(f"\n[{samples_k}K] Batch {batch_num}/3: d_model = {batch_d_models}")

            # Set multiprocessing start method (only needed once, but force=True handles it)
            mp.set_start_method('spawn', force=True)

            # Spawn 8 training processes
            processes = []
            for gpu_id, d_model in enumerate(batch_d_models):
                p = mp.Process(
                    target=train_single_model,
                    args=(gpu_id, d_model, train_samples, config,
                          args.output_path, args.data_path)
                )
                p.start()
                processes.append(p)

            # Wait for batch to complete
            for p in processes:
                p.join()

            logger.info(f"[{samples_k}K] Batch {batch_num}/3 complete")

    total_time = time.time() - total_start
    logger.info("\n" + "=" * 60)
    logger.info("Full experiment complete!")
    logger.info(f"Total time: {total_time:.2f}s ({total_time/3600:.2f}h)")
    logger.info(f"Metrics files: {args.output_path}/metrics_d*_*k.jsonl")


if __name__ == "__main__":
    main()
