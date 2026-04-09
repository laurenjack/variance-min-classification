"""Calibration for RETFound ophthalmology models.

Loads a pre-trained RETFound checkpoint, extracts features, and runs
all calibration methods via the shared sweep module.

Usage:
    python -m jl.double_descent.calibration.calibrate_retfound \
        --checkpoint ./data/medical_calibration/checkpoint-best.pth \
        --data-path ./data/medical_calibration/APTOS2019 \
        --output-path ./output/medical_calibration
"""

import argparse
import logging
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from jl.double_descent.calibration.config import MedCalConfig
from jl.double_descent.calibration.sweep import run_calibration_sweep

logger = logging.getLogger(__name__)


# --- Model loading ---


def load_retfound_model(checkpoint_path: str, config: MedCalConfig, device: torch.device) -> nn.Module:
    """Load pre-trained RETFound checkpoint into a timm ViT-Large.

    Detects num_classes from checkpoint head.weight shape.
    """
    import timm

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    num_classes = checkpoint["model"]["head.weight"].shape[0]
    config.num_classes = num_classes

    model = timm.create_model(
        "vit_large_patch16_224",
        num_classes=num_classes,
        drop_path_rate=0.0,  # No drop path at inference
        global_pool="avg" if config.global_pool else "token",
    )

    model.load_state_dict(checkpoint["model"], strict=True)
    model = model.to(device)
    model.eval()

    logger.info(
        f"Loaded checkpoint (epoch {checkpoint.get('epoch', '?')}), "
        f"{num_classes} classes, "
        f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params"
    )
    return model


# --- Feature extraction ---


def extract_features(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract features from the frozen backbone (before the head layer).

    Uses timm's forward_features() + forward_head(pre_logits=True)
    which applies pool + fc_norm, returning [N, 1024].
    """
    model.eval()
    all_features = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            x = model.forward_features(images)
            features = model.forward_head(x, pre_logits=True)
            all_features.append(features.cpu())
            all_labels.append(labels)

    return torch.cat(all_features), torch.cat(all_labels)


# --- Data loading ---


def build_loader(
    data_dir: str, split: str, config: MedCalConfig
) -> DataLoader:
    """Build a DataLoader for a split using torchvision ImageFolder."""
    from torchvision import transforms
    from torchvision.datasets import ImageFolder

    transform = transforms.Compose([
        transforms.Resize((config.input_size, config.input_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    dataset = ImageFolder(
        root=str(Path(data_dir) / split),
        transform=transform,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    logger.info(f"Loaded {split}: {len(dataset)} images, {len(dataset.classes)} classes")
    logger.info(f"  Class mapping: {dataset.class_to_idx}")
    return loader


# --- Main ---


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Calibrate RETFound ophthalmology models"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to pre-trained RETFound checkpoint (.pth)",
    )
    parser.add_argument(
        "--data-path", type=str, required=True,
        help="Path to dataset directory with train/val/test subdirs",
    )
    parser.add_argument(
        "--output-path", type=str, required=True,
        help="Directory for output artifacts",
    )
    parser.add_argument(
        "--max-steps", type=int, default=30,
        help="L-BFGS steps for L2 calibration (default: 30)",
    )
    parser.add_argument(
        "--sweep-metric", type=str, default="ece", choices=["ece", "nll"],
        help="Metric to select best lambda on validation set (default: ece)",
    )
    args = parser.parse_args()

    config = MedCalConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Load model
    model = load_retfound_model(args.checkpoint, config, device)

    # Load data splits
    train_loader = build_loader(args.data_path, "train", config)
    val_loader = build_loader(args.data_path, "val", config)
    test_loader = build_loader(args.data_path, "test", config)

    # Extract features once
    logger.info("Extracting features...")
    train_features, train_labels = extract_features(model, train_loader, device)
    val_features, val_labels = extract_features(model, val_loader, device)
    test_features, test_labels = extract_features(model, test_loader, device)
    logger.info(
        f"Features: train={train_features.shape}, val={val_features.shape}, "
        f"test={test_features.shape}"
    )

    original_head_state = model.head.state_dict()
    feature_dim = train_features.shape[1]

    # Free model memory
    del model
    torch.cuda.empty_cache()

    # Run calibration sweep
    run_calibration_sweep(
        train_features=train_features,
        train_labels=train_labels,
        val_features=val_features,
        val_labels=val_labels,
        test_features=test_features,
        test_labels=test_labels,
        original_head_state=original_head_state,
        num_classes=config.num_classes,
        feature_dim=feature_dim,
        max_steps=args.max_steps,
        sweep_metric=args.sweep_metric,
        device=device,
        output_dir=Path(args.output_path),
    )


if __name__ == "__main__":
    main()
