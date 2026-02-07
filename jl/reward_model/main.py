#!/usr/bin/env python3
"""Main entry point for reward model training.

Launched via torchrun by SageMaker's torch_distributed configuration.

Usage on SageMaker:
    Automatic via launch_sagemaker.py (torch_distributed handles torchrun).

Local testing:
    torchrun --nproc_per_node=8 -m jl.reward_model.main --train-path ./data/tokenized --output-path ./output
"""

import argparse
import logging
import os
import time

import torch
import torch.distributed as dist

from jl.reward_model.reward_config import RewardConfig
from jl.reward_model.load_data import load_data
from jl.reward_model.model import get_model
from jl.reward_model.trainer import train

# Configure logging with timestamps for CloudWatch/SageMaker
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
    return parser.parse_args()


def main():
    args = parse_args()
    config = RewardConfig()
    total_start = time.time()

    # Initialize DDP (torchrun sets LOCAL_RANK, WORLD_SIZE, etc.)
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    is_main = local_rank == 0
    if not is_main:
        logging.getLogger().setLevel(logging.WARNING)

    # Create output directory on rank 0, then barrier so all ranks see it
    if is_main:
        os.makedirs(args.output_path, exist_ok=True)
    dist.barrier()

    if is_main:
        logger.info(f"DDP training: {world_size} GPUs, effective batch size: {config.train_batch_size * world_size}")
        logger.info(f"Using device: {device}")

    # Load data
    if is_main:
        logger.info("Loading and preparing dataset...")
    data_start = time.time()
    train_loader, val_loader = load_data(config, args.train_path, world_size, local_rank)
    data_time = time.time() - data_start
    if is_main:
        logger.info(f"Data loading completed in {data_time:.2f}s")

    # Load and prepare model (includes DDP wrapping)
    if is_main:
        logger.info("Loading model...")
    model_start = time.time()
    model = get_model(config, device)
    model_time = time.time() - model_start
    if is_main:
        logger.info(f"Model loading completed in {model_time:.2f}s")

    # Train
    if is_main:
        logger.info("Starting training...")
    train_start = time.time()
    train(model, train_loader, val_loader, config, device, args.output_path, is_main)
    train_time = time.time() - train_start

    total_time = time.time() - total_start
    if is_main:
        logger.info(f"Training completed! Total time: {total_time:.2f}s (data: {data_time:.2f}s, model: {model_time:.2f}s, train: {train_time:.2f}s)")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
