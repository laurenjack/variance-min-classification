"""Calibration for pre-trained ImageNet models (ResNet-152, ViT-B/16).

Loads pre-trained models from timm, downloads ImageNet from HuggingFace,
extracts features, splits validation into val/test halves (Guo et al.
protocol), and runs all calibration methods via the shared sweep module.

Usage:
    python -m jl.double_descent.calibration.calibrate_imagenet \
        --model resnet152 --output-path ./output/imagenet/resnet152

    python -m jl.double_descent.calibration.calibrate_imagenet \
        --model vit_base_patch16_224 --output-path ./output/imagenet/vit_b16
"""

import argparse
import logging
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from jl.double_descent.calibration.sweep import run_calibration_sweep

logger = logging.getLogger(__name__)

def info_to_cache_key(timm_name: str) -> str:
    """Sanitize a timm model name into a filesystem-safe cache directory name."""
    return timm_name.replace("/", "_").replace(".", "_")


SUPPORTED_MODELS = {
    "resnet152": {
        # Use the original He et al. 2015 recipe weights (~78.3% top-1, SGD+CE),
        # not the modern "ResNet Strikes Back" recipe (LAMB+BCE+heavy aug).
        # Matches Guo et al. 2017 calibration paper setup.
        "timm_name": "resnet152.tv_in1k",
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
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract features from the frozen backbone (before the classifier head).

    Uses timm's forward_features() + forward_head(pre_logits=True) which
    works uniformly for both ResNet and ViT models — applies global pool
    and any pre-classifier layers, returning the final feature vector.
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


class HFImageNetDataset(torch.utils.data.Dataset):
    """Wraps a HuggingFace ImageNet split with torchvision transforms."""

    def __init__(self, hf_dataset, transform):
        self.hf_dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        example = self.hf_dataset[idx]
        img = example["image"]
        if img.mode != "RGB":
            img = img.convert("RGB")
        img = self.transform(img)
        return img, example["label"]


def build_imagenet_loaders(
    batch_size: int = 64,
    num_workers: int = 8,
) -> Tuple[DataLoader, DataLoader]:
    """Build ImageNet train and full-validation loaders from HuggingFace.

    Downloads ImageNet-1K from HuggingFace (cached via HF_HOME). Returns the
    full 1.28M training set and full 50K validation set (no val/test split —
    the split is applied later on cached features, so changing the seed
    doesn't force re-extraction).

    Returns:
        (train_loader, val_loader)  # val_loader covers all 50K HF val images
    """
    from datasets import load_dataset
    from torchvision import transforms

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

    logger.info("Loading ImageNet-1K from HuggingFace (or cache)...")
    train_hf = load_dataset("ILSVRC/imagenet-1k", split="train")
    val_hf = load_dataset("ILSVRC/imagenet-1k", split="validation")
    logger.info(f"HuggingFace: train={len(train_hf)}, val={len(val_hf)}")

    train_dataset = HFImageNetDataset(train_hf, eval_transform)
    val_dataset = HFImageNetDataset(val_hf, eval_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    logger.info(f"Train: {len(train_dataset)} images, Val: {len(val_dataset)} images")

    return train_loader, val_loader


def load_or_extract_features(
    cache_dir: Path,
    split: str,
    model: nn.Module,
    loader: Optional[DataLoader],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load cached features if present, otherwise extract and save.

    Features are a deterministic function of (model weights, dataset), so we
    cache them under cache_dir/{split}_{features,labels}.pt. Re-running with
    a different --seed or different --output-path reuses the same cache.
    """
    feat_path = cache_dir / f"{split}_features.pt"
    label_path = cache_dir / f"{split}_labels.pt"

    if feat_path.exists() and label_path.exists():
        logger.info(f"Loading cached {split} features from {feat_path}")
        features = torch.load(feat_path, map_location="cpu")
        labels = torch.load(label_path, map_location="cpu")
        logger.info(f"{split.capitalize()} features: {tuple(features.shape)}")
        return features, labels

    if loader is None:
        raise RuntimeError(
            f"No cached {split} features at {cache_dir} and no loader provided."
        )

    logger.info(f"Extracting {split} features...")
    features, labels = extract_features(model, loader, device)
    logger.info(f"{split.capitalize()} features: {tuple(features.shape)}")

    cache_dir.mkdir(parents=True, exist_ok=True)
    torch.save(features, feat_path)
    torch.save(labels, label_path)
    logger.info(f"Saved {split} features to {feat_path}")

    return features, labels


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
        "--feature-cache-dir", type=str, default="./data/imagenet_features",
        help="Directory for cached extracted features (default: ./data/imagenet_features)",
    )
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Load model (needed for head state even when features are cached)
    model, model_info = load_model(args.model, device)
    head_attr = model_info["head_attr"]
    original_head = getattr(model, head_attr)
    original_head_state = {k: v.cpu() for k, v in original_head.state_dict().items()}
    feature_dim = model_info["feature_dim"]
    num_classes = original_head.out_features

    # Feature cache keyed by timm name (sanitized), independent of --seed and --output-path
    cache_key = info_to_cache_key(model_info["timm_name"])
    cache_dir = Path(args.feature_cache_dir) / cache_key
    logger.info(f"Feature cache: {cache_dir}")

    train_feat_path = cache_dir / "train_features.pt"
    val_feat_path = cache_dir / "val_features.pt"
    need_data = not (train_feat_path.exists() and val_feat_path.exists())

    if need_data:
        train_loader, val_loader = build_imagenet_loaders(
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
    else:
        train_loader = val_loader = None  # type: ignore[assignment]
        logger.info("All features cached — skipping ImageNet download and loader build")

    # Load or extract features (train + full 50K val; val/test split applied below)
    train_features, train_labels = load_or_extract_features(
        cache_dir, "train", model, train_loader, device
    )
    val_all_features, val_all_labels = load_or_extract_features(
        cache_dir, "val", model, val_loader, device
    )

    # Free model memory — features are cached on CPU, no need for the backbone
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Apply Guo et al. 2017 val/test split on cached 50K val features (seed-dependent)
    n_val = val_all_features.shape[0]
    perm = torch.randperm(n_val, generator=torch.Generator().manual_seed(args.seed))
    val_idx = perm[:5000]
    test_idx = perm[5000:]
    val_features = val_all_features[val_idx]
    val_labels = val_all_labels[val_idx]
    test_features = val_all_features[test_idx]
    test_labels = val_all_labels[test_idx]
    logger.info(
        f"Split (seed={args.seed}): val={val_features.shape[0]}, test={test_features.shape[0]}"
    )

    # Lambda sweep — cheap now that features are cached (SGD only, ~1 min each)
    lambdas = [1e-5, 1e-4, 1e-3, 1e-2]

    # Run calibration sweep (SGD with momentum for ImageNet-scale data)
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
        lambdas=lambdas,
        max_steps=args.max_steps,
        sweep_metric=args.sweep_metric,
        device=device,
        output_dir=Path(args.output_path),
        use_sgd=True,
        sgd_lr=0.3,
        sgd_epochs=150,
        sgd_warmup_epochs=10,
    )


if __name__ == "__main__":
    main()
