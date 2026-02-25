#!/usr/bin/env python3
"""Main entry point for Deep Double Descent training.

Reproduces Figure 1 from Nakkiran et al. (2019) "Deep Double Descent":
ResNet18 with varying width parameter k trained on CIFAR-10 with 15% label noise.

Usage:
    python -m jl.double_descent.convnet_main --output-path ./output

    # For quick smoke test:
    python -m jl.double_descent.convnet_main --output-path ./output --width-max 4 --epochs 10
"""

import argparse
import logging
import os
import time

import torch

from jl.double_descent.convnet_config import DDConfig
from jl.double_descent.convnet_data import load_cifar10_with_noise
from jl.double_descent.trainer import train

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
        "--width-min",
        type=int,
        default=None,
        help="Minimum width parameter k (overrides config default)"
    )
    parser.add_argument(
        "--width-max",
        type=int,
        default=None,
        help="Maximum width parameter k (overrides config default)"
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
        help="Label noise probability (overrides config default)"
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
    if args.width_min is not None:
        config.width_min = args.width_min
    if args.width_max is not None:
        config.width_max = args.width_max
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

    # Setup device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_path, exist_ok=True)

    logger.info(f"Deep Double Descent Training")
    logger.info(f"Width range: k={config.width_min}..{config.width_max}")
    logger.info(f"Epochs: {config.epochs}, Batch size: {config.batch_size}")
    logger.info(f"Learning rate: {config.learning_rate}")
    logger.info(f"Label noise: {config.label_noise}")
    logger.info(f"Data augmentation: {config.data_augmentation}")
    logger.info(f"Device: {device}")

    # Load data
    logger.info("Loading CIFAR-10 with label noise...")
    data_start = time.time()
    train_loader, test_loader = load_cifar10_with_noise(
        noise_prob=config.label_noise,
        batch_size=config.batch_size,
        data_augmentation=config.data_augmentation,
        data_dir=args.data_path,
    )
    data_time = time.time() - data_start
    logger.info(f"Data loading completed in {data_time:.2f}s")
    logger.info(f"Training samples: {len(train_loader.dataset)}")
    logger.info(f"Test samples: {len(test_loader.dataset)}")

    # Train
    logger.info("Starting training...")
    train_start = time.time()
    train(config, train_loader, test_loader, device, args.output_path)
    train_time = time.time() - train_start

    total_time = time.time() - total_start
    logger.info(
        f"Training completed! Total time: {total_time:.2f}s "
        f"(data: {data_time:.2f}s, train: {train_time:.2f}s)"
    )


if __name__ == "__main__":
    main()
