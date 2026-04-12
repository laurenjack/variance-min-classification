#!/usr/bin/env python3
"""Main entry point for Deep Double Descent training.

Reproduces Figure 1 from Nakkiran et al. (2019) "Deep Double Descent":
ResNet18 with varying width parameter k trained on CIFAR-10 with 15% label noise.

Uses all available GPUs. Trains models in parallel (one per GPU), batching
sequentially when there are more k values than GPUs.

Two modes:
  Default: 20 k values (2..64) covering the double descent curve.
  --large-k: 8 larger k values (72..128) extending into the overparameterized regime.

Optional flag:
  --val-split
     Hold out a deterministic 5K validation split from the noised 50K
     training set (seed=73132, see resnet18_data.save_val_set). Train on the
     remaining 45K. Per-epoch val_loss / val_error are logged to
     metrics_k*.jsonl, and the final evaluation.jsonl row gets
     val_loss / val_error / val_ece plus temperature scaling fields.
     The main process saves val.pt to the output folder before spawning workers.

Usage:
    # Default mode (k=2..64, 20 models)
    python -m jl.double_descent.resnet18.resnet18_main --output-path ./output --val-split

    # Large-k mode (k=72..128, 8 models)
    python -m jl.double_descent.resnet18.resnet18_main --output-path ./output --large-k --val-split
"""

import argparse
import logging
import math
import os
import time
from functools import partial

import torch
import torch.multiprocessing as mp

from jl.double_descent.resnet18.resnet18_config import DDConfig
from jl.double_descent.resnet18.resnet18_data import download_cifar10, save_val_set
from jl.double_descent.resnet18.resnet18k import make_resnet18k
from jl.double_descent.resnet18.trainer import train_single_model

# The 20 target k values for the main double descent sweep.
K_VALUES = [2, 3, 4, 5, 6, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64]

# 8 larger k values extending into the overparameterized regime.
LARGE_K_VALUES = [72, 80, 88, 96, 104, 112, 120, 128]

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
        "--val-split",
        action="store_true",
        help="Hold out a deterministic 5K validation split from the noised "
             "50K training set (seed=73132). Train on 45K. Saves val.pt to "
             "the output folder and logs per-epoch val metrics.",
    )
    parser.add_argument(
        "--large-k",
        action="store_true",
        help="Train the 8 large-k models (72..128) instead of the default "
             "20-value sweep (2..64).",
    )
    return parser.parse_args()


def run_training(k_values, num_gpus, args, config):
    """Train models for the given k values using all available GPUs.

    Batches k values into groups of num_gpus, running each batch in parallel.
    """
    total_start = time.time()
    num_batches = math.ceil(len(k_values) / num_gpus)

    logger.info(f"Width values: k={k_values}")
    logger.info(f"GPUs: {num_gpus}, Batches: {num_batches}")
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

    mp.set_start_method('spawn', force=True)

    for batch_idx in range(num_batches):
        batch_k = k_values[batch_idx * num_gpus:(batch_idx + 1) * num_gpus]
        batch_num = batch_idx + 1

        logger.info(f"\n{'='*60}")
        logger.info(f"Batch {batch_num}/{num_batches}: k = {batch_k}")
        logger.info(f"{'='*60}")

        processes = []
        for gpu_id, k in enumerate(batch_k):
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

        for i, p in enumerate(processes):
            p.join()
            logger.info(f"Process for k={batch_k[i]} completed")

        logger.info(f"Batch {batch_num}/{num_batches} complete")

    total_time = time.time() - total_start
    logger.info("\n" + "=" * 60)
    logger.info(f"All training completed! Total time: {total_time:.2f}s ({total_time/3600:.2f}h)")
    logger.info(f"Metrics saved to {args.output_path}/metrics_k*.jsonl")


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

    # Determine k values and check GPUs
    k_values = LARGE_K_VALUES if args.large_k else K_VALUES
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("No GPUs found. This experiment requires at least 1 GPU.")

    mode = "Large-k" if args.large_k else "Default"
    logger.info(f"Deep Double Descent Training - {mode} Mode")

    # Create output directory
    os.makedirs(args.output_path, exist_ok=True)

    # Download data once before spawning processes
    logger.info("Downloading CIFAR-10 if needed...")
    download_cifar10(args.data_path)
    logger.info("Data ready.")

    # Save held-out validation set once in the parent process
    if config.use_val_split:
        logger.info("Saving held-out validation set (45K/5K split, seed=73132)...")
        val_path = save_val_set(
            output_path=args.output_path,
            data_dir=args.data_path,
            noise_prob=config.label_noise,
        )
        logger.info(f"Saved val set to {val_path}")

    run_training(k_values, num_gpus, args, config)


if __name__ == "__main__":
    main()
