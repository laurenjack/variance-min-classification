"""Calibration for ResNet-110 on CIFAR-10 or CIFAR-100.

Loads a trained ResNet-110, extracts features, and runs all calibration
methods via the shared sweep module.

Usage:
    python -m jl.double_descent.calibration.calibrate_cifar \
        --dataset cifar10 \
        --model-path ./output \
        --data-path ./data \
        --output-path ./output/calibration_cifar10
"""

import argparse
import logging
from pathlib import Path

import torch

from jl.double_descent.calibration.resnet110 import extract_features, make_resnet110
from jl.double_descent.calibration.sweep import run_calibration_sweep
from jl.double_descent.calibration.train_resnet110 import get_cifar_splits

logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Calibrate ResNet-110 on CIFAR")
    parser.add_argument("--dataset", type=str, required=True, choices=["cifar10", "cifar100"])
    parser.add_argument("--model-path", type=str, required=True,
                        help="Directory containing resnet110_<dataset>.pt")
    parser.add_argument("--data-path", type=str, default="./data")
    parser.add_argument("--output-path", type=str, default=None,
                        help="Output directory (default: <model-path>/calibration_<dataset>)")
    parser.add_argument("--max-steps", type=int, default=100,
                        help="L-BFGS steps per lambda (default: 100)")
    parser.add_argument("--sweep-metric", type=str, default="ece", choices=["ece", "nll"],
                        help="Metric to select best lambda (default: ece)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed for val/test split (must match training)")
    args = parser.parse_args()

    num_classes = 10 if args.dataset == "cifar10" else 100
    feature_dim = 64
    checkpoint_path = Path(args.model_path) / f"resnet110_{args.dataset}.pt"

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {checkpoint_path}. "
            f"Run train_resnet110 first."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Calibrating ResNet-110 on {args.dataset}, device={device}")

    # Load model
    model = make_resnet110(num_classes).to(device)
    model.load_state_dict(
        torch.load(str(checkpoint_path), map_location=device, weights_only=True)
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    original_head_state = model.fc.state_dict()

    # Feature cache: invalidated when checkpoint is newer than cached features
    cache_dir = checkpoint_path.parent / f"features_{args.dataset}_seed{args.seed}"
    feat_files = ["train_features.pt", "train_labels.pt",
                  "val_features.pt", "val_labels.pt",
                  "test_features.pt", "test_labels.pt"]
    cache_valid = (
        cache_dir.exists()
        and all((cache_dir / f).exists() for f in feat_files)
        and min((cache_dir / f).stat().st_mtime for f in feat_files) > checkpoint_path.stat().st_mtime
    )

    if cache_valid:
        logger.info(f"Loading cached features from {cache_dir}")
        train_features = torch.load(cache_dir / "train_features.pt", map_location="cpu")
        train_labels = torch.load(cache_dir / "train_labels.pt", map_location="cpu")
        val_features = torch.load(cache_dir / "val_features.pt", map_location="cpu")
        val_labels = torch.load(cache_dir / "val_labels.pt", map_location="cpu")
        test_features = torch.load(cache_dir / "test_features.pt", map_location="cpu")
        test_labels = torch.load(cache_dir / "test_labels.pt", map_location="cpu")
        logger.info(f"Train: {train_features.shape}, Val: {val_features.shape}, Test: {test_features.shape}")
    else:
        # Load data with the same split as training
        train_dataset, val_dataset, test_dataset = get_cifar_splits(
            args.dataset, args.data_path, args.seed
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=128, shuffle=False, num_workers=2, pin_memory=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=128, shuffle=False, num_workers=2, pin_memory=True
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=128, shuffle=False, num_workers=2, pin_memory=True
        )

        def extract_all(loader, name):
            all_features = []
            all_labels = []
            with torch.no_grad():
                for images, labels in loader:
                    feats = extract_features(model, images.to(device))
                    all_features.append(feats.cpu())
                    all_labels.append(labels)
            features = torch.cat(all_features)
            labels = torch.cat(all_labels)
            logger.info(f"{name} features: {features.shape}")
            return features, labels

        logger.info("Extracting features...")
        train_features, train_labels = extract_all(train_loader, "Train")
        val_features, val_labels = extract_all(val_loader, "Val")
        test_features, test_labels = extract_all(test_loader, "Test")

        cache_dir.mkdir(parents=True, exist_ok=True)
        torch.save(train_features, cache_dir / "train_features.pt")
        torch.save(train_labels, cache_dir / "train_labels.pt")
        torch.save(val_features, cache_dir / "val_features.pt")
        torch.save(val_labels, cache_dir / "val_labels.pt")
        torch.save(test_features, cache_dir / "test_features.pt")
        torch.save(test_labels, cache_dir / "test_labels.pt")
        logger.info(f"Saved features to {cache_dir}")

    # Free model memory
    del model
    torch.cuda.empty_cache()

    # Determine output directory
    output_dir = args.output_path
    if output_dir is None:
        output_dir = str(Path(args.model_path) / f"calibration_{args.dataset}")

    # Lambda range shifted downward — best λ from default range was 1e-3
    # (the second-smallest), suggesting the sweet spot is below the old range.
    lambdas = [1e-7, 1e-6, 1e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2]

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
        output_dir=Path(output_dir),
    )


if __name__ == "__main__":
    main()
