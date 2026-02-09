#!/usr/bin/env python3
"""Main entry point for reward model training.

Supports both single-GPU and multi-GPU (DDP) training via --is-multi flag.

Usage on SageMaker (multi-GPU via torchrun):
    Automatic via launch_sagemaker.py (torch_distributed handles torchrun).

Local single-GPU:
    python -m jl.reward_model.main --train-path ./data/tokenized --output-path ./output

Local multi-GPU:
    torchrun --nproc_per_node=8 -m jl.reward_model.main --is-multi --train-path ./data/tokenized --output-path ./output
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
        "--is-multi",
        action="store_true",
        default=False,
        help="Enable multi-GPU DDP training (requires torchrun)"
    )
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
        "--checkpoint-path",
        type=str,
        default="",
        help="Directory for training checkpoints (enables resume on spot interruption)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = RewardConfig(is_multi=args.is_multi, checkpoint_path=args.checkpoint_path)
    total_start = time.time()

    if config.is_multi:
        # Initialize DDP (torchrun sets LOCAL_RANK, WORLD_SIZE, etc.)
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")

        is_main = local_rank == 0
        if not is_main:
            logging.getLogger().setLevel(logging.WARNING)

        # Create output/checkpoint directories on rank 0, then barrier so all ranks see them
        if is_main:
            os.makedirs(args.output_path, exist_ok=True)
            if config.checkpoint_path:
                os.makedirs(config.checkpoint_path, exist_ok=True)
        dist.barrier()

        if is_main:
            logger.info(f"DDP training: {world_size} GPUs, effective batch size: {config.train_batch_size * world_size}")
            logger.info(f"Using device: {device}")
    else:
        # Single-GPU path
        local_rank = 0
        world_size = 1
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        is_main = True

        os.makedirs(args.output_path, exist_ok=True)
        if config.checkpoint_path:
            os.makedirs(config.checkpoint_path, exist_ok=True)
        logger.info(f"Single-GPU training, batch size: {config.train_batch_size}")
        logger.info(f"Using device: {device}")

    # Load data
    if is_main:
        logger.info("Loading and preparing dataset...")
    data_start = time.time()
    train_loader, val_loader = load_data(config, args.train_path, world_size, local_rank)
    data_time = time.time() - data_start
    if is_main:
        logger.info(f"Data loading completed in {data_time:.2f}s")

    # Load and prepare model (includes DDP wrapping when is_multi)
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

    if config.is_multi:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
