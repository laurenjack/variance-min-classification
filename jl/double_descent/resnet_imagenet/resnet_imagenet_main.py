#!/usr/bin/env python3
"""Entry point: train ResNet-50 on ImageNet-1K with the He et al. 2015 recipe.

Single GPU. SGD (momentum=0.9, wd=1e-4), base LR 0.1 with step decay
(÷10) at epochs 30/60/80, batch 256, 90 epochs, mixed precision.

Usage:
    python -m jl.double_descent.resnet_imagenet.resnet_imagenet_main \
        --output-path ./output/resnet_imagenet/$(date +%m-%d-%H%M)
"""

import argparse
import logging
import os

import torch

from jl.double_descent.resnet_imagenet.resnet50 import make_resnet50
from jl.double_descent.resnet_imagenet.resnet_imagenet_config import (
    ResNetImageNetConfig,
)
from jl.double_descent.resnet_imagenet.resnet_imagenet_data import (
    build_imagenet_loaders,
)
from jl.double_descent.resnet_imagenet.trainer import train

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train ResNet-50 on ImageNet-1K (He et al. 2015 recipe)"
    )
    parser.add_argument(
        "--output-path", type=str, required=True,
        help="Directory to save metrics.jsonl and final model.pt",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument(
        "--no-amp", action="store_true",
        help="Disable mixed-precision training",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = ResNetImageNetConfig()

    torch.manual_seed(42)

    if args.epochs is not None:
        config.epochs = args.epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.learning_rate is not None:
        config.learning_rate = args.learning_rate
    if args.num_workers is not None:
        config.num_workers = args.num_workers
    if args.no_amp:
        config.use_amp = False

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for ImageNet training.")
    device = torch.device("cuda:0")

    logger.info(f"Device: {device} ({torch.cuda.get_device_name(0)})")
    logger.info(
        f"Epochs: {config.epochs}, Batch size: {config.batch_size}, "
        f"LR: {config.learning_rate}, WD: {config.weight_decay}, "
        f"Momentum: {config.momentum}"
    )
    logger.info(
        f"LR decay epochs: {config.lr_decay_epochs} "
        f"(gamma={config.lr_decay_factor})"
    )
    logger.info(f"AMP: {config.use_amp}")

    os.makedirs(args.output_path, exist_ok=True)

    logger.info("Building ImageNet data loaders...")
    train_loader, val_loader = build_imagenet_loaders(
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        data_augmentation=config.data_augmentation,
    )

    logger.info("Building ResNet-50...")
    model = make_resnet50(num_classes=config.num_classes).to(device)
    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info(f"ResNet-50 params: {num_params:.2f}M")

    train(model, train_loader, val_loader, config, device, args.output_path)


if __name__ == "__main__":
    main()
