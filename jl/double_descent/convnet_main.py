#!/usr/bin/env python3
"""Main entry point for Deep Double Descent training.

Reproduces Figure 1 from Nakkiran et al. (2019) "Deep Double Descent":
ResNet18 with varying width parameter k trained on CIFAR-10 with 15% label noise.

This script trains 8 models in parallel on 8 GPUs using torch.multiprocessing.
Each GPU trains one model with width k, k+1, ..., k+7.

Usage:
    # Train models with k=1 to k=8
    python -m jl.double_descent.convnet_main --output-path ./output --k-start 1

    # For quick smoke test:
    python -m jl.double_descent.convnet_main --output-path ./output --k-start 1 --epochs 10

    # Full run (64 widths) requires 8 separate runs:
    # --k-start 1, 9, 17, 25, 33, 41, 49, 57
"""

import argparse
import logging
import os
import time

import torch
import torch.multiprocessing as mp

from jl.double_descent.convnet_config import DDConfig
from jl.double_descent.convnet_data import download_cifar10
from jl.double_descent.trainer import train_single_model

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
        help="Starting width parameter k. Will train k, k+1, ..., k+7. (default: 9)"
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
    return parser.parse_args()


def main():
    args = parse_args()
    config = DDConfig()
    total_start = time.time()

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

    # Check GPU count - require exactly 8 GPUs
    num_gpus = torch.cuda.device_count()
    if num_gpus < 8:
        raise RuntimeError(
            f"This script requires 8 GPUs, but only found {num_gpus}. "
            f"Please run on an 8-GPU instance (e.g., 8x H100 or 8x A100)."
        )

    # Compute k values for this run
    k_values = [config.k_start + i for i in range(8)]

    logger.info("Deep Double Descent Training")
    logger.info(f"Width values: k={k_values}")
    logger.info(f"Epochs: {config.epochs}, Batch size: {config.batch_size}")
    logger.info(f"Learning rate: {config.learning_rate}")
    logger.info(f"Label noise: {config.label_noise}")
    logger.info(f"Data augmentation: {config.data_augmentation}")
    logger.info(f"GPUs available: {num_gpus}")

    # Create output directory
    os.makedirs(args.output_path, exist_ok=True)

    # Download data once before spawning processes
    logger.info("Downloading CIFAR-10 if needed...")
    download_cifar10(args.data_path)
    logger.info("Data ready.")

    # Set multiprocessing start method
    mp.set_start_method('spawn', force=True)

    # Spawn 8 training processes
    logger.info("Spawning 8 training processes...")
    processes = []
    for gpu_id in range(8):
        k = k_values[gpu_id]
        p = mp.Process(
            target=train_single_model,
            args=(gpu_id, k, config, args.output_path, args.data_path)
        )
        p.start()
        processes.append(p)
        logger.info(f"Started process for k={k} on GPU {gpu_id}")

    # Wait for all processes to complete
    for i, p in enumerate(processes):
        p.join()
        logger.info(f"Process for k={k_values[i]} completed")

    total_time = time.time() - total_start
    logger.info(f"All training completed! Total time: {total_time:.2f}s ({total_time/3600:.2f}h)")
    logger.info(f"Metrics saved to {args.output_path}/metrics_k*.jsonl")


if __name__ == "__main__":
    main()
