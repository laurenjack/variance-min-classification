#!/usr/bin/env python3
"""Main entry point for Deep Double Descent training.

Reproduces Figure 1 from Nakkiran et al. (2019) "Deep Double Descent":
ResNet18 with varying width parameter k trained on CIFAR-10 with 15% label noise.

Requires exactly 8 GPUs. Trains 8 models in parallel using torch.multiprocessing.

Two modes:

1. DEFAULT MODE (no flags):
   - Trains 16 models across 2 batches (8 models per batch)
   - k values: 4, 8, 12, ..., 64 (increment by 4)
   - Batch 1: k=4,8,12,16,20,24,28,32
   - Batch 2: k=36,40,44,48,52,56,60,64

2. VARIANCE MODE (--variance):
   - Trains models for variance analysis: 16 k values x 4 disjoint splits
   - k values: 4, 8, 12, ..., 64 (increment by 4)
   - Each split is a disjoint 12.5K subset from the 50K training data
   - 8 batches total (2 k values x 4 splits per batch)

Usage:
    # Default mode (k=4-64 in 2 batches)
    python -m jl.double_descent.resnet18.resnet18_main --output-path ./output

    # Variance experiment
    python -m jl.double_descent.resnet18.resnet18_main --variance --output-path ./output
"""

import argparse
import logging
import os
import time
from functools import partial

import torch
import torch.multiprocessing as mp

from jl.double_descent.resnet18.resnet18_config import DDConfig
from jl.double_descent.resnet18.resnet18_data import download_cifar10
from jl.double_descent.resnet18.resnet18k import make_resnet18k
from jl.double_descent.resnet18.trainer import train_single_model

# Hardcoded experiment parameters
REQUIRED_GPUS = 6
K_INCREMENT = 1
NUM_BATCHES = 1  # Default mode runs 1 batch: k=1-6

# Variance experiment parameters
VARIANCE_K_VALUES = [2, 6, 10, 14]  # Temporary: fill in gaps for first/second descent
VARIANCE_NUM_SPLITS = 4  # 4 disjoint training sets of 12.5K each

# Configure logging with timestamps
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train ResNet18 models for Deep Double Descent reproduction"
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
        default="./data",
        help="Path to store CIFAR-10 data"
    )
    parser.add_argument(
        "--k-start",
        type=int,
        default=None,
        help="Starting width parameter k. Will train k, k+8, k+16, ..., k+8*(N-1) where N is GPU count. (default: 72)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs (overrides config default of 4000)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size (overrides config default of 128)"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Learning rate (overrides config default of 0.0001)"
    )
    parser.add_argument(
        "--label-noise",
        type=float,
        default=None,
        help="Label noise probability (overrides config default of 0.15)"
    )
    parser.add_argument(
        "--no-augmentation",
        action="store_true",
        help="Disable data augmentation"
    )
    parser.add_argument(
        "--cosine-decay-epoch",
        type=int,
        default=None,
        help="Epoch to start cosine LR decay to 0 (disabled by default)"
    )
    parser.add_argument(
        "--variance",
        action="store_true",
        help="Run variance experiment (16 k values x 4 disjoint splits) instead of default mode"
    )
    return parser.parse_args()


def run_default(args, config):
    """Run the default training mode (2 batches of 8 k values each, k=4-64)."""
    total_start = time.time()

    # Compute all k values across both batches
    all_k_values = [config.k_start + K_INCREMENT * i for i in range(REQUIRED_GPUS * NUM_BATCHES)]

    logger.info("Deep Double Descent Training - Default Mode")
    logger.info(f"Width values: k={all_k_values}")
    logger.info(f"Batches: {NUM_BATCHES} (8 models per batch)")
    logger.info(f"Epochs: {config.epochs}, Batch size: {config.batch_size}")
    logger.info(f"Learning rate: {config.learning_rate}")
    if config.cosine_decay_epoch is not None:
        logger.info(f"Cosine decay from epoch {config.cosine_decay_epoch}")
    logger.info(f"Label noise: {config.label_noise}")
    logger.info(f"Data augmentation: {config.data_augmentation}")

    # Set multiprocessing start method
    mp.set_start_method('spawn', force=True)

    # Run batches sequentially
    for batch_idx in range(NUM_BATCHES):
        batch_k_values = all_k_values[batch_idx * REQUIRED_GPUS:(batch_idx + 1) * REQUIRED_GPUS]
        batch_num = batch_idx + 1

        logger.info(f"\n{'='*60}")
        logger.info(f"Batch {batch_num}/{NUM_BATCHES}: k = {batch_k_values}")
        logger.info(f"{'='*60}")

        # Spawn training processes (one per GPU)
        logger.info(f"Spawning {REQUIRED_GPUS} training processes...")
        processes = []
        for gpu_id in range(REQUIRED_GPUS):
            k = batch_k_values[gpu_id]
            model_factory = partial(make_resnet18k, k=k, num_classes=10)
            model_label = f"k{k}"
            model_params = {"k": k}
            p = mp.Process(
                target=train_single_model,
                args=(gpu_id, model_factory, model_label, model_params, config, args.output_path, args.data_path)
            )
            p.start()
            processes.append(p)
            logger.info(f"Started process for k={k} on GPU {gpu_id}")

        # Wait for all processes to complete
        for i, p in enumerate(processes):
            p.join()
            logger.info(f"Process for k={batch_k_values[i]} completed")

        logger.info(f"Batch {batch_num}/{NUM_BATCHES} complete")

    total_time = time.time() - total_start
    logger.info("\n" + "=" * 60)
    logger.info(f"All training completed! Total time: {total_time:.2f}s ({total_time/3600:.2f}h)")
    logger.info(f"Metrics saved to {args.output_path}/metrics_k*.jsonl")


def run_variance(args, config):
    """Run the variance experiment (16 k values x 4 disjoint splits)."""
    total_start = time.time()

    # 2 k values per batch x 4 splits = 8 GPUs
    k_per_batch = REQUIRED_GPUS // VARIANCE_NUM_SPLITS  # 2

    logger.info("Deep Double Descent - Variance Experiment")
    logger.info(f"k values: {VARIANCE_K_VALUES}")
    logger.info(f"Disjoint splits: {VARIANCE_NUM_SPLITS} x 12500 samples each")
    logger.info(f"Total models: {len(VARIANCE_K_VALUES) * VARIANCE_NUM_SPLITS} = 64")
    logger.info(f"Batches: {len(VARIANCE_K_VALUES) // k_per_batch} (2 k per batch x 4 splits)")
    logger.info(f"Epochs: {config.epochs}, Batch size: {config.batch_size}")
    logger.info(f"Learning rate: {config.learning_rate}")
    if config.cosine_decay_epoch is not None:
        logger.info(f"Cosine decay from epoch {config.cosine_decay_epoch}")
    logger.info(f"Label noise: {config.label_noise} (applied independently per split)")
    logger.info(f"Data augmentation: {config.data_augmentation}")

    # Process k values in batches of 2
    num_batches = len(VARIANCE_K_VALUES) // k_per_batch
    for batch_idx in range(num_batches):
        batch_k_values = VARIANCE_K_VALUES[batch_idx * k_per_batch:(batch_idx + 1) * k_per_batch]
        batch_num = batch_idx + 1

        logger.info(f"\n{'='*60}")
        logger.info(f"Batch {batch_num}/{num_batches}: k = {batch_k_values}")
        logger.info(f"Training {len(batch_k_values) * VARIANCE_NUM_SPLITS} models")
        logger.info(f"{'='*60}")

        # Set multiprocessing start method
        mp.set_start_method('spawn', force=True)

        # Spawn processes: 2 k values x 4 splits = 8 processes
        processes = []
        gpu_id = 0
        for k in batch_k_values:
            model_factory = partial(make_resnet18k, k=k, num_classes=10)
            model_label = f"k{k}"
            model_params = {"k": k}
            for split_id in range(VARIANCE_NUM_SPLITS):
                p = mp.Process(
                    target=train_single_model,
                    args=(gpu_id, model_factory, model_label, model_params, config, args.output_path, args.data_path),
                    kwargs={'split_id': split_id, 'num_splits': VARIANCE_NUM_SPLITS}
                )
                p.start()
                processes.append((p, k, split_id))
                logger.info(f"Started k={k}, split={split_id} on GPU {gpu_id}")
                gpu_id += 1

        # Wait for batch to complete
        for p, k, split_id in processes:
            p.join()
            logger.info(f"Completed k={k}, split={split_id}")

        logger.info(f"Batch {batch_num}/{num_batches} complete")

    total_time = time.time() - total_start
    logger.info("\n" + "=" * 60)
    logger.info("Variance experiment complete!")
    logger.info(f"Total time: {total_time:.2f}s ({total_time/3600:.2f}h)")
    logger.info(f"Metrics files: {args.output_path}/metrics_k*_split*.jsonl")


def main():
    args = parse_args()
    config = DDConfig()

    # Set global seed for reproducibility
    torch.manual_seed(42)

    # Override config with command line arguments
    if args.epochs is not None:
        config.epochs = args.epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.learning_rate is not None:
        config.learning_rate = args.learning_rate
    if args.label_noise is not None:
        config.label_noise = args.label_noise
    if args.no_augmentation:
        config.data_augmentation = False
    if args.k_start is not None:
        config.k_start = args.k_start
    if args.cosine_decay_epoch is not None:
        config.cosine_decay_epoch = args.cosine_decay_epoch

    # Require exactly 8 GPUs
    num_gpus = torch.cuda.device_count()
    if num_gpus != REQUIRED_GPUS:
        raise RuntimeError(
            f"This experiment requires exactly {REQUIRED_GPUS} GPUs, "
            f"but found {num_gpus}. Please run on an 8-GPU instance."
        )

    # Create output directory
    os.makedirs(args.output_path, exist_ok=True)

    # Download data once before spawning processes
    logger.info("Downloading CIFAR-10 if needed...")
    download_cifar10(args.data_path)
    logger.info("Data ready.")

    # Run appropriate mode
    if args.variance:
        run_variance(args, config)
    else:
        run_default(args, config)


if __name__ == "__main__":
    main()
