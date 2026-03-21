#!/usr/bin/env python3
"""Main entry point for Transformer Double Descent training.

Reproduces Figure 3 from Nakkiran et al. (2019) "Deep Double Descent":
6-layer encoder-decoder Transformer with varying embedding dimension d_model
trained on IWSLT'14 German-to-English translation.

Two modes:

1. DEFAULT MODE (no flags):
   - Trains 24 models: 24 d_model values x 1 sample size (36K)
   - d_model: 8, 16, 24, ..., 192
   - 3 batches total, 8 models per batch

2. VARIANCE MODE (--variance):
   - Trains 96 models for variance analysis: 24 d_model values x 4 disjoint splits
   - d_model: 16, 32, 48, ..., 384
   - Each split is a disjoint 36K subset from the 160K training data
   - 12 batches total (2 d_model x 4 splits per batch = 8 GPUs)

Usage:
    # Default double descent experiment (d_model 8-192)
    python -m jl.double_descent.transformer.transformer_main \\
        --output-path ./output \\
        --data-path ./data/iwslt14.tokenized.de-en

    # Variance experiment
    python -m jl.double_descent.transformer.transformer_main \\
        --variance \\
        --output-path ./output \\
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
TRAIN_SAMPLES = [36000]  # 36K samples
D_MODEL_VALUES = list(range(200, 392, 8))  # [200, 208, 216, ..., 384] - 24 values
REQUIRED_GPUS = 8

# Variance experiment parameters
VARIANCE_D_MODEL_VALUES = list(range(16, 400, 16))  # [16, 32, 48, ..., 384] - 24 values
VARIANCE_SAMPLES_PER_SPLIT = 36000  # Each disjoint split has 36K samples
VARIANCE_NUM_SPLITS = 4  # 4 disjoint training sets
VARIANCE_D_MODELS_PER_BATCH = 2  # Train 2 d_model values per batch (2 * 4 splits = 8 GPUs)


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
    parser.add_argument(
        "--variance",
        action="store_true",
        help="Run variance experiment (24 d_model x 4 disjoint splits) instead of double descent"
    )
    return parser.parse_args()


def run_double_descent(args, config):
    """Run the default double descent experiment (24 d_model values, 36K samples)."""
    total_start = time.time()

    train_samples = TRAIN_SAMPLES[0]
    samples_k = train_samples // 1000
    num_batches = len(D_MODEL_VALUES) // REQUIRED_GPUS

    logger.info("Transformer Double Descent - Full Experiment")
    logger.info(f"d_model values: {D_MODEL_VALUES}")
    logger.info(f"Train samples: {train_samples}")
    logger.info(f"Total models: {len(D_MODEL_VALUES)}")
    logger.info(f"Batches: {num_batches}")
    logger.info(f"Max steps: {config.max_steps}, Max tokens/batch: {config.max_tokens}")

    logger.info(f"\n{'='*60}")
    logger.info(f"Starting {samples_k}K sample runs")
    logger.info(f"{'='*60}")

    # Loop over d_model batches (3 batches of 8)
    for batch_idx in range(0, len(D_MODEL_VALUES), REQUIRED_GPUS):
        batch_d_models = D_MODEL_VALUES[batch_idx:batch_idx + REQUIRED_GPUS]
        batch_num = batch_idx // REQUIRED_GPUS + 1

        logger.info(f"\n[{samples_k}K] Batch {batch_num}/{num_batches}: d_model = {batch_d_models}")

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

        logger.info(f"[{samples_k}K] Batch {batch_num}/{num_batches} complete")

    total_time = time.time() - total_start
    logger.info("\n" + "=" * 60)
    logger.info("Full experiment complete!")
    logger.info(f"Total time: {total_time:.2f}s ({total_time/3600:.2f}h)")
    logger.info(f"Metrics files: {args.output_path}/metrics_d*_{samples_k}k.jsonl")


def run_variance(args, config):
    """Run the variance experiment (24 d_model x 4 disjoint splits).

    Batches 2 d_model values at a time, each with 4 splits = 8 GPUs per batch.
    GPU layout per batch:
        d_model_a: split 0→GPU0, split 1→GPU1, split 2→GPU2, split 3→GPU3
        d_model_b: split 0→GPU4, split 1→GPU5, split 2→GPU6, split 3→GPU7
    """
    total_start = time.time()

    total_models = len(VARIANCE_D_MODEL_VALUES) * VARIANCE_NUM_SPLITS
    num_batches = len(VARIANCE_D_MODEL_VALUES) // VARIANCE_D_MODELS_PER_BATCH

    logger.info("Transformer Variance Experiment")
    logger.info(f"d_model values: {VARIANCE_D_MODEL_VALUES}")
    logger.info(f"Disjoint splits: {VARIANCE_NUM_SPLITS} x {VARIANCE_SAMPLES_PER_SPLIT} samples each")
    logger.info(f"Total models: {total_models}")
    logger.info(f"Batches: {num_batches} ({VARIANCE_D_MODELS_PER_BATCH} d_model x {VARIANCE_NUM_SPLITS} splits per batch)")
    logger.info(f"Max steps: {config.max_steps}, Max tokens/batch: {config.max_tokens}")

    # Loop over d_model values in pairs (2 d_model x 4 splits = 8 GPUs per batch)
    for batch_idx in range(0, len(VARIANCE_D_MODEL_VALUES), VARIANCE_D_MODELS_PER_BATCH):
        batch_d_models = VARIANCE_D_MODEL_VALUES[batch_idx:batch_idx + VARIANCE_D_MODELS_PER_BATCH]
        batch_num = batch_idx // VARIANCE_D_MODELS_PER_BATCH + 1

        logger.info(f"\n{'='*60}")
        logger.info(f"Batch {batch_num}/{num_batches}: d_model = {batch_d_models}")
        logger.info(f"Training {len(batch_d_models) * VARIANCE_NUM_SPLITS} models on disjoint {VARIANCE_SAMPLES_PER_SPLIT // 1000}K splits")
        logger.info(f"{'='*60}")

        # Set multiprocessing start method
        mp.set_start_method('spawn', force=True)

        # Spawn processes: 2 d_model x 4 splits = 8 GPUs
        processes = []
        for d_model_offset, d_model in enumerate(batch_d_models):
            for split_id in range(VARIANCE_NUM_SPLITS):
                gpu_id = d_model_offset * VARIANCE_NUM_SPLITS + split_id
                p = mp.Process(
                    target=train_single_model,
                    args=(gpu_id, d_model, VARIANCE_SAMPLES_PER_SPLIT, config,
                          args.output_path, args.data_path),
                    kwargs={'split_id': split_id, 'num_splits': VARIANCE_NUM_SPLITS}
                )
                p.start()
                processes.append(p)

        # Wait for batch to complete
        for p in processes:
            p.join()

        logger.info(f"Batch {batch_num}/{num_batches} complete")

    total_time = time.time() - total_start
    logger.info("\n" + "=" * 60)
    logger.info("Variance experiment complete!")
    logger.info(f"Total time: {total_time:.2f}s ({total_time/3600:.2f}h)")
    logger.info(f"Metrics files: {args.output_path}/metrics_d*_split*.jsonl")


def main():
    args = parse_args()
    config = TDDConfig()

    # Require exactly 8 GPUs
    num_gpus = torch.cuda.device_count()
    if num_gpus != REQUIRED_GPUS:
        raise RuntimeError(
            f"This experiment requires exactly {REQUIRED_GPUS} GPUs, "
            f"but found {num_gpus}. Please run on an 8-GPU instance."
        )

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

    # Run appropriate experiment
    if args.variance:
        run_variance(args, config)
    else:
        run_double_descent(args, config)


if __name__ == "__main__":
    main()
