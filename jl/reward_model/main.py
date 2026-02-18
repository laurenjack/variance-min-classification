#!/usr/bin/env python3
"""Main entry point for reward model training.

Usage:
    python -m jl.reward_model.main --train-path ./data/tokenized --output-path ./output
"""

import argparse
import logging
import os
import time

import torch

from jl.reward_model.reward_config import RewardConfig
from jl.reward_model.load_data import load_data
from jl.reward_model.model import get_model
from jl.reward_model.trainer import train

# Configure logging with timestamps
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train a reward model")
    parser.add_argument(
        "--train-path",
        type=str,
        required=True,
        help="Path to the tokenized training dataset"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Path to save trained model"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Learning rate for training (overrides config default)"
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=None,
        help="Number of warmup steps for quadratic LR warmup (overrides config default)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = RewardConfig()
    total_start = time.time()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_path, exist_ok=True)
    logger.info(f"Training with batch size: {config.train_batch_size}")
    logger.info(f"Using device: {device}")

    # Load data
    logger.info("Loading and preparing dataset...")
    data_start = time.time()
    train_loader, val_loader = load_data(config, args.train_path)
    data_time = time.time() - data_start
    logger.info(f"Data loading completed in {data_time:.2f}s")

    # Load and prepare model
    logger.info("Loading model...")
    model_start = time.time()
    model = get_model(config, device)
    model_time = time.time() - model_start
    logger.info(f"Model loading completed in {model_time:.2f}s")

    # Train
    logger.info("Starting training...")
    train_start = time.time()
    train(model, train_loader, val_loader, config, device, args.output_path, args.learning_rate, args.warmup_steps)
    train_time = time.time() - train_start

    total_time = time.time() - total_start
    logger.info(f"Training completed! Total time: {total_time:.2f}s (data: {data_time:.2f}s, model: {model_time:.2f}s, train: {train_time:.2f}s)")


if __name__ == "__main__":
    main()
