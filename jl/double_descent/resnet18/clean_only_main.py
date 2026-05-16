#!/usr/bin/env python3
"""Standalone entry point: train ResNet18 models on the clean-only subset.

Companion to resnet18_main.py --track-shadows. Trains the same widths
under the same recipe (DDConfig, augmentation, val_split), but on the
mislabel_mask==0 subset (~38K of the 45K train pool). Saves model +
ES checkpoint only — no shadow tracking.

Usage:
    python -m jl.double_descent.resnet18.clean_only_main \
        --output-path ./output/resnet18_clean/$(date +%m-%d-%H%M) \
        --data-path ./data \
        --k-values 18 32 64
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
from jl.double_descent.resnet18.clean_only_trainer import (
    train_single_model_clean_only,
)

K_VALUES = [2, 3, 4, 5, 6, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64]
LARGE_K_VALUES = [72, 80, 88, 96, 104, 112, 120, 128]

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="ResNet18 clean-only training (filter out mislabeled images)"
    )
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--data-path", type=str, default="./data")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--label-noise", type=float, default=None)
    parser.add_argument("--no-augmentation", action="store_true")
    parser.add_argument("--cosine-decay-epoch", type=int, default=None)
    parser.add_argument("--no-bf16", action="store_true")
    parser.add_argument(
        "--k-values", type=int, nargs="+", default=None,
        help="Subset of k values to train (default: full K_VALUES sweep).",
    )
    parser.add_argument(
        "--large-k", action="store_true",
        help="Use the large-k sweep (72..128) instead of the default 2..64.",
    )
    parser.add_argument(
        "--models-per-gpu", type=int, default=1,
        help="Number of training processes to colocate per GPU (needs MPS).",
    )
    return parser.parse_args()


def run_training(k_values, num_gpus, args, config):
    total_start = time.time()
    models_per_gpu = max(1, args.models_per_gpu)

    jobs = [(k, None) for k in k_values]
    gpu_to_jobs = [[] for _ in range(num_gpus)]
    for i, job in enumerate(jobs):
        gpu_to_jobs[i % num_gpus].append(job)

    logger.info(f"Clean-only sweep: k={k_values}")
    logger.info(f"GPUs: {num_gpus}, models/GPU: {models_per_gpu}")
    for gpu_id, jjs in enumerate(gpu_to_jobs):
        logger.info(f"  GPU {gpu_id} -> {jjs}")
    logger.info(f"Epochs: {config.epochs}, Batch size: {config.batch_size}, lr: {config.learning_rate}")
    logger.info(f"BF16 autocast: {config.use_bf16}")
    if config.cosine_decay_epoch is not None:
        logger.info(f"Cosine decay from epoch {config.cosine_decay_epoch}")
    logger.info(f"Label noise: {config.label_noise} (filtered to clean only)")

    mp.set_start_method('spawn', force=True)
    processes = []
    for gpu_id, jjs in enumerate(gpu_to_jobs):
        for k, _ in jjs:
            model_factory = partial(make_resnet18k, k=k, num_classes=10)
            model_label = f"k{k}"
            model_params = {"k": k}
            p = mp.Process(
                target=train_single_model_clean_only,
                args=(gpu_id, model_factory, model_label, model_params,
                      config, args.output_path, args.data_path),
            )
            p.start()
            processes.append((gpu_id, model_label, p))

    for gpu_id, label, p in processes:
        p.join()
        logger.info(f"Process {label} (GPU {gpu_id}) completed")

    total_time = time.time() - total_start
    logger.info(f"\nAll clean-only training done. Total: {total_time:.0f}s ({total_time/3600:.2f}h)")


def main():
    args = parse_args()
    config = DDConfig()
    torch.manual_seed(42)

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
    if args.no_bf16:
        config.use_bf16 = False
    # use_val_split is always on for clean-only (we need val for ES)
    config.use_val_split = True

    if args.k_values is not None:
        if args.large_k:
            raise RuntimeError("--k-values and --large-k are mutually exclusive.")
        k_values = list(args.k_values)
    else:
        k_values = LARGE_K_VALUES if args.large_k else K_VALUES

    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("No CUDA GPUs visible.")

    os.makedirs(args.output_path, exist_ok=True)
    download_cifar10(args.data_path)
    run_training(k_values, num_gpus, args, config)


if __name__ == "__main__":
    main()
