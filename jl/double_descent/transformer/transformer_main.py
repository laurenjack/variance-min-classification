#!/usr/bin/env python3
"""Main entry point for Transformer Double Descent training.

Reproduces Figure 3 from Nakkiran et al. (2019) "Deep Double Descent":
6-layer encoder-decoder Transformer with varying embedding dimension d_model
trained on IWSLT'14 German-to-English translation.

This script trains N models in parallel on N GPUs using torch.multiprocessing.
Each GPU trains one model with d_model, d_model+8, d_model+16, ..., d_model+8*(N-1).

Usage:
    # Train models starting at d_model=8 (one model per available GPU)
    python -m jl.double_descent.transformer.transformer_main --output-path ./output --d-model-start 8

    # For quick smoke test:
    python -m jl.double_descent.transformer.transformer_main --output-path ./output --d-model-start 64 --max-steps 100

    # On 8 GPUs with d-model-start=8, trains d_model=8,16,24,32,40,48,56,64
    # On 1 GPU with d-model-start=8, trains d_model=8
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
        "--d-model-start",
        type=int,
        default=None,
        help="Starting embedding dimension. Will train d_model, d_model+8, ..., d_model+8*(N-1) where N is GPU count. (default: 8)"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Maximum training steps (overrides config default of 80000)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Maximum tokens per batch (overrides config default of 4096)"
    )
    parser.add_argument(
        "--train-samples",
        type=int,
        default=None,
        help="Number of training samples (overrides config default of 4000)"
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=None,
        help="LR warmup steps (overrides config default of 4000)"
    )
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=None,
        help="Label smoothing value (overrides config default of 0.1, use 0 to disable)"
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=None,
        help="Log metrics every N steps (overrides config default of 100)"
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=None,
        help="Evaluate on validation set every N steps (overrides config default of 100)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = TDDConfig()
    total_start = time.time()

    # Override config with command line arguments
    if args.d_model_start is not None:
        config.d_model_start = args.d_model_start
    if args.max_steps is not None:
        config.max_steps = args.max_steps
    if args.max_tokens is not None:
        config.max_tokens = args.max_tokens
    if args.train_samples is not None:
        config.train_samples = args.train_samples
    if args.warmup_steps is not None:
        config.warmup_steps = args.warmup_steps
    if args.label_smoothing is not None:
        config.label_smoothing = args.label_smoothing if args.label_smoothing > 0 else None
    if args.log_interval is not None:
        config.log_interval = args.log_interval
    if args.eval_interval is not None:
        config.eval_interval = args.eval_interval

    # Check GPU count - require at least 1 GPU
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError(
            "This script requires at least 1 GPU, but none were found. "
            "Please run on a machine with CUDA-capable GPUs."
        )

    # Compute d_model values for this run (one d_model per GPU, incrementing by 8)
    d_model_values = [config.d_model_start + 8 * i for i in range(num_gpus)]

    logger.info("Transformer Double Descent Training")
    logger.info(f"d_model values: {d_model_values}")
    logger.info(f"Max steps: {config.max_steps}, Max tokens/batch: {config.max_tokens}")
    logger.info(f"Warmup steps: {config.warmup_steps}")
    logger.info(f"Label smoothing: {config.label_smoothing}")
    logger.info(f"Training samples: {config.train_samples}")
    logger.info(f"GPUs available: {num_gpus}")

    # Create output directory
    os.makedirs(args.output_path, exist_ok=True)

    # Verify data exists - check for required files
    required_files = ["train.de", "train.en", "valid.de", "valid.en", "test.de", "test.en", "code"]
    missing_files = [f for f in required_files if not os.path.isfile(os.path.join(args.data_path, f))]
    if missing_files:
        raise FileNotFoundError(
            f"Preprocessed IWSLT'14 data not found at {args.data_path}.\n"
            f"Missing files: {missing_files}\n\n"
            "Please run preprocessing locally first:\n"
            "  ./infra/prepare_iwslt14.sh\n\n"
            "Then upload the data directory to the remote machine before training."
        )

    # Set multiprocessing start method
    mp.set_start_method('spawn', force=True)

    # Spawn training processes (one per GPU)
    logger.info(f"Spawning {num_gpus} training process(es)...")
    processes = []
    for gpu_id in range(num_gpus):
        d_model = d_model_values[gpu_id]
        p = mp.Process(
            target=train_single_model,
            args=(gpu_id, d_model, config, args.output_path, args.data_path)
        )
        p.start()
        processes.append(p)
        logger.info(f"Started process for d_model={d_model} on GPU {gpu_id}")

    # Wait for all processes to complete
    for i, p in enumerate(processes):
        p.join()
        logger.info(f"Process for d_model={d_model_values[i]} completed")

    total_time = time.time() - total_start
    logger.info(f"All training completed! Total time: {total_time:.2f}s ({total_time/3600:.2f}h)")
    logger.info(f"Metrics saved to {args.output_path}/metrics_d*.jsonl")


if __name__ == "__main__":
    main()
