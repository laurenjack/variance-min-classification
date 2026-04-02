#!/usr/bin/env python3
"""Main entry point for CIFAR ResNet depth-varying double descent training.

Uses the CIFAR-10 ResNet architecture from He et al. (2015) with 3 stages
and variable depth (6n+2 layers). Sweeps n values to explore
underparameterized -> interpolation threshold -> overparameterized regimes.

Requires exactly 8 GPUs. Trains models in parallel using torch.multiprocessing.

Two modes:

1. DEFAULT MODE (no flags):
   - Trains 16 models in batches across available GPUs
   - n values: 4, 8, 12, ..., 64 (depth 26 to 386)

2. VARIANCE MODE (--variance):
   - Trains models for variance analysis: n values x 4 disjoint splits
   - Default n values: 4, 8, 12, ..., 64
   - Each split is a disjoint 12.5K subset from the 50K training data
   - Batches of 2 n values x 4 splits = 8 processes per batch

Usage:
    # Default mode (n=1-18 in 3 batches)
    python -m jl.double_descent.resnet_depth.resnet_depth_main --output-path ./output

    # Variance experiment
    python -m jl.double_descent.resnet_depth.resnet_depth_main --variance --output-path ./output
"""

import argparse
import logging
import os
import time
from functools import partial

import torch
import torch.multiprocessing as mp

from jl.double_descent.resnet_depth.cifar_resnet import make_cifar_resnet
from jl.double_descent.resnet_depth.depth_config import DepthDDConfig
from jl.double_descent.resnet18.resnet18_data import download_cifar10
from jl.double_descent.resnet18.trainer import train_single_model

# Variance experiment parameters
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
        description="Train CIFAR ResNet models with varying depth for double descent"
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
        "--n-start",
        type=int,
        default=None,
        help="Starting n value (blocks per stage, depth=6n+2). Default: 1"
    )
    parser.add_argument(
        "--n-step",
        type=int,
        default=None,
        help="Increment between n values. Default: 1"
    )
    parser.add_argument(
        "--n-count",
        type=int,
        default=None,
        help="Number of n values to sweep. Default: 16"
    )
    parser.add_argument(
        "--k",
        type=int,
        default=None,
        help="Width multiplier (widths = [16k, 32k, 64k]). Default: 1"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs (overrides config default of 800)"
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
        help="Learning rate (overrides config default of 0.001)"
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
        help="Run variance experiment (n values x 4 disjoint splits) instead of default mode"
    )
    return parser.parse_args()


def _make_model_label(n: int, k: int) -> str:
    """Create model label for filenames."""
    if k == 1:
        return f"n{n}"
    return f"n{n}_k{k}"


def _make_model_params(n: int, k: int) -> dict:
    """Create model params dict for metrics."""
    params = {"n": n}
    if k != 1:
        params["k"] = k
    return params


def run_default(args, config, num_gpus: int):
    """Run the default training mode (n=1..18, batched across available GPUs)."""
    total_start = time.time()

    # Compute all n values
    all_n_values = [config.n_start + config.n_step * i for i in range(config.n_count)]

    # Trim to valid range (n >= 1)
    all_n_values = [n for n in all_n_values if n >= 1]

    logger.info("CIFAR ResNet Depth Double Descent - Default Mode")
    logger.info(f"GPUs: {num_gpus}")
    logger.info(f"n values: {all_n_values} (depth = 6n+2)")
    logger.info(f"Width multiplier k={config.k} -> channels [16k, 32k, 64k] = [{16*config.k}, {32*config.k}, {64*config.k}]")
    logger.info(f"Epochs: {config.epochs}, Batch size: {config.batch_size}")
    logger.info(f"Learning rate: {config.learning_rate}")
    if config.cosine_decay_epoch is not None:
        logger.info(f"Cosine decay from epoch {config.cosine_decay_epoch}")
    logger.info(f"Label noise: {config.label_noise}")
    logger.info(f"Data augmentation: {config.data_augmentation}")

    # Set multiprocessing start method
    mp.set_start_method('spawn', force=True)

    # Run in batches sized to available GPUs
    num_batches = (len(all_n_values) + num_gpus - 1) // num_gpus
    for batch_idx in range(num_batches):
        start = batch_idx * num_gpus
        batch_n_values = all_n_values[start:start + num_gpus]
        batch_num = batch_idx + 1

        logger.info(f"\n{'='*60}")
        logger.info(f"Batch {batch_num}/{num_batches}: n = {batch_n_values}")
        logger.info(f"{'='*60}")

        # Spawn training processes (one per GPU)
        logger.info(f"Spawning {len(batch_n_values)} training processes...")
        processes = []
        for gpu_id, n in enumerate(batch_n_values):
            model_factory = partial(make_cifar_resnet, n=n, k=config.k, num_classes=10)
            model_label = _make_model_label(n, config.k)
            model_params = _make_model_params(n, config.k)
            p = mp.Process(
                target=train_single_model,
                args=(gpu_id, model_factory, model_label, model_params, config, args.output_path, args.data_path)
            )
            p.start()
            processes.append(p)
            logger.info(f"Started process for n={n} (depth={6*n+2}) on GPU {gpu_id}")

        # Wait for all processes to complete
        for i, p in enumerate(processes):
            p.join()
            logger.info(f"Process for n={batch_n_values[i]} completed")

        logger.info(f"Batch {batch_num}/{num_batches} complete")

    total_time = time.time() - total_start
    logger.info("\n" + "=" * 60)
    logger.info(f"All training completed! Total time: {total_time:.2f}s ({total_time/3600:.2f}h)")
    logger.info(f"Metrics saved to {args.output_path}/metrics_n*.jsonl")


def run_variance(args, config, num_gpus: int):
    """Run the variance experiment (n values x 4 disjoint splits)."""
    total_start = time.time()

    # Compute all n values
    all_n_values = [config.n_start + config.n_step * i for i in range(config.n_count)]
    all_n_values = [n for n in all_n_values if n >= 1]

    # How many n values per batch given available GPUs and splits
    n_per_batch = num_gpus // VARIANCE_NUM_SPLITS

    if n_per_batch < 1:
        raise RuntimeError(
            f"Variance mode requires at least {VARIANCE_NUM_SPLITS} GPUs "
            f"(one per split), but found {num_gpus}."
        )

    logger.info("CIFAR ResNet Depth Double Descent - Variance Experiment")
    logger.info(f"GPUs: {num_gpus}")
    logger.info(f"n values: {all_n_values} (depth = 6n+2)")
    logger.info(f"Width multiplier k={config.k}")
    logger.info(f"Disjoint splits: {VARIANCE_NUM_SPLITS} x 12500 samples each")
    logger.info(f"Total models: {len(all_n_values) * VARIANCE_NUM_SPLITS}")
    logger.info(f"Batches: {(len(all_n_values) + n_per_batch - 1) // n_per_batch} ({n_per_batch} n per batch x {VARIANCE_NUM_SPLITS} splits)")
    logger.info(f"Epochs: {config.epochs}, Batch size: {config.batch_size}")
    logger.info(f"Learning rate: {config.learning_rate}")
    if config.cosine_decay_epoch is not None:
        logger.info(f"Cosine decay from epoch {config.cosine_decay_epoch}")
    logger.info(f"Label noise: {config.label_noise} (applied independently per split)")
    logger.info(f"Data augmentation: {config.data_augmentation}")

    # Process n values in batches
    num_batches = (len(all_n_values) + n_per_batch - 1) // n_per_batch
    for batch_idx in range(num_batches):
        batch_n_values = all_n_values[batch_idx * n_per_batch:(batch_idx + 1) * n_per_batch]
        batch_num = batch_idx + 1

        logger.info(f"\n{'='*60}")
        logger.info(f"Batch {batch_num}/{num_batches}: n = {batch_n_values}")
        logger.info(f"Training {len(batch_n_values) * VARIANCE_NUM_SPLITS} models")
        logger.info(f"{'='*60}")

        # Set multiprocessing start method
        mp.set_start_method('spawn', force=True)

        # Spawn processes: 2 n values x 4 splits = 8 processes
        processes = []
        gpu_id = 0
        for n in batch_n_values:
            model_factory = partial(make_cifar_resnet, n=n, k=config.k, num_classes=10)
            model_label = _make_model_label(n, config.k)
            model_params = _make_model_params(n, config.k)
            for split_id in range(VARIANCE_NUM_SPLITS):
                p = mp.Process(
                    target=train_single_model,
                    args=(gpu_id, model_factory, model_label, model_params, config, args.output_path, args.data_path),
                    kwargs={'split_id': split_id, 'num_splits': VARIANCE_NUM_SPLITS}
                )
                p.start()
                processes.append((p, n, split_id))
                logger.info(f"Started n={n}, split={split_id} on GPU {gpu_id}")
                gpu_id += 1

        # Wait for batch to complete
        for p, n, split_id in processes:
            p.join()
            logger.info(f"Completed n={n}, split={split_id}")

        logger.info(f"Batch {batch_num}/{num_batches} complete")

    total_time = time.time() - total_start
    logger.info("\n" + "=" * 60)
    logger.info("Variance experiment complete!")
    logger.info(f"Total time: {total_time:.2f}s ({total_time/3600:.2f}h)")
    logger.info(f"Metrics files: {args.output_path}/metrics_n*_split*.jsonl")


def main():
    args = parse_args()
    config = DepthDDConfig()

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
    if args.n_start is not None:
        config.n_start = args.n_start
    if args.n_step is not None:
        config.n_step = args.n_step
    if args.n_count is not None:
        config.n_count = args.n_count
    if args.k is not None:
        config.k = args.k
    if args.cosine_decay_epoch is not None:
        config.cosine_decay_epoch = args.cosine_decay_epoch

    # Detect GPUs
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("No GPUs found. This experiment requires at least 1 GPU.")

    # Create output directory
    os.makedirs(args.output_path, exist_ok=True)

    # Download data once before spawning processes
    logger.info("Downloading CIFAR-10 if needed...")
    download_cifar10(args.data_path)
    logger.info("Data ready.")

    # Run appropriate mode
    if args.variance:
        run_variance(args, config, num_gpus)
    else:
        run_default(args, config, num_gpus)


if __name__ == "__main__":
    main()
