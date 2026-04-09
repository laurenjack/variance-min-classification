"""Calibration for pre-trained ImageNet models (ResNet-152, ViT-B/16).

Loads pre-trained models from timm, extracts features, splits ImageNet
validation into val/test halves (Guo et al. protocol), and runs all
calibration methods via the shared sweep module.

Usage:
    python -m jl.double_descent.calibration.calibrate_imagenet \
        --model resnet152 --data-path ./data/imagenet \
        --output-path ./output/imagenet/resnet152

    python -m jl.double_descent.calibration.calibrate_imagenet \
        --model vit_base_patch16_224 --data-path ./data/imagenet \
        --output-path ./output/imagenet/vit_b16
"""

import argparse
import logging
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from jl.double_descent.calibration.sweep import run_calibration_sweep

logger = logging.getLogger(__name__)

SUPPORTED_MODELS = {
    "resnet152": {
        "timm_name": "resnet152",
        "feature_dim": 2048,
        "head_attr": "fc",
    },
    "vit_base_patch16_224": {
        "timm_name": "vit_base_patch16_224",
        "feature_dim": 768,
        "head_attr": "head",
    },
}


def load_model(model_name: str, device: torch.device) -> Tuple[nn.Module, dict]:
    """Load pre-trained model from timm.

    Returns:
        (model, model_info) where model_info has feature_dim and head_attr.
    """
    import timm

    if model_name not in SUPPORTED_MODELS:
        raise ValueError(
            f"Unsupported model: {model_name}. "
            f"Supported: {list(SUPPORTED_MODELS.keys())}"
        )

    info = SUPPORTED_MODELS[model_name]
    model = timm.create_model(info["timm_name"], pretrained=True)
    model = model.to(device)
    model.eval()

    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info(f"Loaded {model_name}: {num_params:.1f}M params, feature_dim={info['feature_dim']}")

    return model, info


def extract_features(
    model: nn.Module,
    loader: DataLoader,
    model_info: dict,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract features from the frozen backbone (before the classifier head).

    Uses timm's forward_features() + forward_head(pre_logits=True) for ViT,
    or a forward hook on the avgpool layer for ResNet.
    """
    model.eval()
    all_features = []
    all_labels = []

    head_attr = model_info["head_attr"]

    if head_attr == "head":
        # ViT-style: use timm's forward_head with pre_logits=True
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(device)
                x = model.forward_features(images)
                features = model.forward_head(x, pre_logits=True)
                all_features.append(features.cpu())
                all_labels.append(labels)
    else:
        # ResNet-style: hook before the fc layer
        # timm ResNet: forward_features returns pooled features
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(device)
                x = model.forward_features(images)
                # forward_features returns [N, C, 1, 1] for ResNet after global pool
                # or [N, C] depending on timm version — flatten to be safe
                features = x.flatten(1)
                all_features.append(features.cpu())
                all_labels.append(labels)

    return torch.cat(all_features), torch.cat(all_labels)


def build_imagenet_loaders(
    data_path: str,
    batch_size: int = 64,
    num_workers: int = 8,
    seed: int = 42,
    train_subset: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Build ImageNet train, val, and test loaders.

    Guo et al. protocol: 50K ImageNet validation set split into
    25K val / 25K test via random permutation with fixed seed.

    Args:
        data_path: Path to ImageNet root (with train/ and val/ subdirs).
        batch_size: Batch size for data loaders.
        num_workers: Number of data loading workers.
        seed: Random seed for val/test split.
        train_subset: If > 0, subsample training set to this many images.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    from torchvision import transforms
    from torchvision.datasets import ImageFolder

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    eval_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    # Training set (for L2 calibration features)
    train_dataset = ImageFolder(
        root=str(Path(data_path) / "train"),
        transform=eval_transform,  # No augmentation — just extracting features
    )

    if train_subset > 0 and train_subset < len(train_dataset):
        g = torch.Generator().manual_seed(seed)
        indices = torch.randperm(len(train_dataset), generator=g)[:train_subset].tolist()
        train_dataset = Subset(train_dataset, indices)
        logger.info(f"Subsampled training set to {train_subset} images")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    logger.info(f"Train: {len(train_dataset)} images")

    # Validation set — split into val/test halves
    full_val_dataset = ImageFolder(
        root=str(Path(data_path) / "val"),
        transform=eval_transform,
    )

    n_val = len(full_val_dataset)
    perm = torch.randperm(n_val, generator=torch.Generator().manual_seed(seed))
    val_indices = perm[: n_val // 2].tolist()
    test_indices = perm[n_val // 2:].tolist()

    val_dataset = Subset(full_val_dataset, val_indices)
    test_dataset = Subset(full_val_dataset, test_indices)

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    logger.info(f"Val: {len(val_dataset)} images, Test: {len(test_dataset)} images")
    logger.info(f"Total classes: {len(full_val_dataset.classes)}")

    return train_loader, val_loader, test_loader


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Calibrate pre-trained ImageNet models"
    )
    parser.add_argument(
        "--model", type=str, required=True,
        choices=list(SUPPORTED_MODELS.keys()),
        help="Model architecture to calibrate",
    )
    parser.add_argument(
        "--data-path", type=str, required=True,
        help="Path to ImageNet root directory (with train/ and val/ subdirs)",
    )
    parser.add_argument(
        "--output-path", type=str, required=True,
        help="Directory for output artifacts",
    )
    parser.add_argument(
        "--batch-size", type=int, default=64,
        help="Batch size for feature extraction (default: 64)",
    )
    parser.add_argument(
        "--num-workers", type=int, default=8,
        help="Number of data loading workers (default: 8)",
    )
    parser.add_argument(
        "--max-steps", type=int, default=30,
        help="L-BFGS steps for L2 calibration (default: 30)",
    )
    parser.add_argument(
        "--sweep-metric", type=str, default="ece", choices=["ece", "nll"],
        help="Metric to select best lambda on validation set (default: ece)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for val/test split (default: 42)",
    )
    parser.add_argument(
        "--train-subset", type=int, default=0,
        help="Subsample training set to N images for L2 calibration (default: 0 = use all)",
    )
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Load model
    model, model_info = load_model(args.model, device)

    # Build data loaders
    train_loader, val_loader, test_loader = build_imagenet_loaders(
        data_path=args.data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        train_subset=args.train_subset,
    )

    # Extract features
    logger.info("Extracting training features...")
    train_features, train_labels = extract_features(model, train_loader, model_info, device)
    logger.info(f"Train features: {train_features.shape}")

    logger.info("Extracting val features...")
    val_features, val_labels = extract_features(model, val_loader, model_info, device)
    logger.info(f"Val features: {val_features.shape}")

    logger.info("Extracting test features...")
    test_features, test_labels = extract_features(model, test_loader, model_info, device)
    logger.info(f"Test features: {test_features.shape}")

    # Get original head state
    head_attr = model_info["head_attr"]
    original_head = getattr(model, head_attr)
    original_head_state = original_head.state_dict()
    feature_dim = model_info["feature_dim"]
    num_classes = original_head.out_features

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
        num_classes=num_classes,
        feature_dim=feature_dim,
        max_steps=args.max_steps,
        sweep_metric=args.sweep_metric,
        device=device,
        output_dir=Path(args.output_path),
    )


if __name__ == "__main__":
    main()
