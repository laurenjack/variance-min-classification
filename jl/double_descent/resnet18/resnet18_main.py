#!/usr/bin/env python3
"""Main entry point for Deep Double Descent training.

Reproduces Figure 1 from Nakkiran et al. (2019) "Deep Double Descent":
ResNet18 with varying width parameter k trained on CIFAR-10 with 15% label noise.

Requires exactly 10 GPUs in default mode. Trains 10 models in parallel
using torch.multiprocessing across 2 batches (20 k values total).

Two modes:

1. DEFAULT MODE (no flags):
   - Trains 20 models across 2 batches (10 models per batch)
   - Hardcoded k values: 2, 3, 4, 5, 6, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44,
     48, 52, 56, 60, 64
   - Batch 1: k = 2, 3, 4, 5, 6, 8, 12, 16, 20, 24
   - Batch 2: k = 28, 32, 36, 40, 44, 48, 52, 56, 60, 64

2. VARIANCE MODE (--variance):
   - Trains models for variance analysis: disjoint-split experiment.
   - See run_variance(). Unchanged from the previous behavior.

Optional flag:
  --val-split
     Hold out a deterministic 5K validation split from the noised 50K
     training set (seed=73132, see resnet18_data.save_val_set). Train on the
     remaining 45K. Per-epoch val_loss / val_error are logged to
     metrics_k*.jsonl, and the final evaluation.jsonl row gets
     val_loss / val_error / val_ece. The main process saves val.pt to the
     output folder before spawning workers. Cannot be combined with
     --variance.

Usage:
    # Default mode (k=2..64, 20 models in 2 batches of 10)
    python -m jl.double_descent.resnet18.resnet18_main --output-path ./output

    # Default mode with validation split (45K train / 5K val)
    python -m jl.double_descent.resnet18.resnet18_main --output-path ./output --val-split

    # Variance experiment (unchanged)
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
from jl.double_descent.resnet18.resnet18_data import download_cifar10, save_val_set
from jl.double_descent.resnet18.resnet18k import make_resnet18k
from jl.double_descent.resnet18.trainer import train_single_model

# Hardcoded experiment parameters (default mode)
REQUIRED_GPUS = 10
# The 20 target k values for the main sweep. First 10 go in batch 1, last 10
# in batch 2. 10 GPUs × 2 batches = 20 models.
K_VALUES = [2, 3, 4, 5, 6, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64]
NUM_BATCHES = 2
assert len(K_VALUES) == REQUIRED_GPUS * NUM_BATCHES, (
    f"K_VALUES length {len(K_VALUES)} must equal "
    f"REQUIRED_GPUS * NUM_BATCHES = {REQUIRED_GPUS * NUM_BATCHES}"
)

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
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs (overrides config default)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size (overrides config default)"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Learning rate (overrides config default)"
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
        help="Run variance experiment instead of default mode"
    )
    parser.add_argument(
        "--val-split",
        action="store_true",
        help="Hold out a deterministic 5K validation split from the noised "
             "50K training set (seed=73132). Train on 45K. Saves val.pt to "
             "the output folder and logs per-epoch val metrics. Not "
             "compatible with --variance.",
    )
    return parser.parse_args()


def run_default(args, config):
    """Run default training: 2 batches × 10 models, hardcoded K_VALUES."""
    total_start = time.time()

    logger.info("Deep Double Descent Training - Default Mode")
    logger.info(f"Width values: k={K_VALUES}")
    logger.info(f"Batches: {NUM_BATCHES} ({REQUIRED_GPUS} models per batch)")
    logger.info(
        f"Validation split: "
        f"{'enabled (45K train / 5K val, seed=73132)' if config.use_val_split else 'disabled (full 50K train)'}"
    )
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
        batch_k_values = K_VALUES[batch_idx * REQUIRED_GPUS:(batch_idx + 1) * REQUIRED_GPUS]
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
    """Run the variance experiment (disjoint-split mode). Unchanged."""
    total_start = time.time()

    k_per_batch = max(1, REQUIRED_GPUS // VARIANCE_NUM_SPLITS)

    logger.info("Deep Double Descent - Variance Experiment")
    logger.info(f"k values: {VARIANCE_K_VALUES}")
    logger.info(f"Disjoint splits: {VARIANCE_NUM_SPLITS} x 12500 samples each")
    logger.info(f"Total models: {len(VARIANCE_K_VALUES) * VARIANCE_NUM_SPLITS}")
    logger.info(f"Epochs: {config.epochs}, Batch size: {config.batch_size}")
    logger.info(f"Learning rate: {config.learning_rate}")
    if config.cosine_decay_epoch is not None:
        logger.info(f"Cosine decay from epoch {config.cosine_decay_epoch}")
    logger.info(f"Label noise: {config.label_noise} (applied independently per split)")
    logger.info(f"Data augmentation: {config.data_augmentation}")

    num_batches = max(1, len(VARIANCE_K_VALUES) // k_per_batch)
    for batch_idx in range(num_batches):
        batch_k_values = VARIANCE_K_VALUES[batch_idx * k_per_batch:(batch_idx + 1) * k_per_batch]
        batch_num = batch_idx + 1

        logger.info(f"\n{'='*60}")
        logger.info(f"Batch {batch_num}/{num_batches}: k = {batch_k_values}")
        logger.info(f"Training {len(batch_k_values) * VARIANCE_NUM_SPLITS} models")
        logger.info(f"{'='*60}")

        mp.set_start_method('spawn', force=True)

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
    if args.cosine_decay_epoch is not None:
        config.cosine_decay_epoch = args.cosine_decay_epoch
    if args.val_split:
        config.use_val_split = True

    # Reject incompatible combinations
    if args.val_split and args.variance:
        raise RuntimeError(
            "--val-split and --variance are mutually exclusive. "
            "Variance mode uses disjoint training splits; val-split mode "
            "uses a 45K/5K train/val cut from the full noised 50K."
        )

    # Require exactly REQUIRED_GPUS GPUs
    num_gpus = torch.cuda.device_count()
    if num_gpus != REQUIRED_GPUS:
        raise RuntimeError(
            f"This experiment requires exactly {REQUIRED_GPUS} GPUs, "
            f"but found {num_gpus}. Please run on a {REQUIRED_GPUS}-GPU instance."
        )

    # Create output directory
    os.makedirs(args.output_path, exist_ok=True)

    # Download data once before spawning processes
    logger.info("Downloading CIFAR-10 if needed...")
    download_cifar10(args.data_path)
    logger.info("Data ready.")

    # Save held-out validation set once in the parent process, so every
    # run folder has a reproducible val.pt artifact downstream TS can load.
    if config.use_val_split and not args.variance:
        logger.info("Saving held-out validation set (45K/5K split, seed=73132)...")
        val_path = save_val_set(
            output_path=args.output_path,
            data_dir=args.data_path,
            noise_prob=config.label_noise,
        )
        logger.info(f"Saved val set to {val_path}")

    # Run appropriate mode
    if args.variance:
        run_variance(args, config)
    else:
        run_default(args, config)


if __name__ == "__main__":
    main()
