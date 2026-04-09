"""Calibration for ResNet18k models on CIFAR-10.

Loads a single ResNet18k model (by k), extracts features, splits the
CIFAR-10 test set into val/test halves, and runs all calibration methods
via the shared sweep module.

Usage:
    python -m jl.double_descent.calibration.calibrate_resnet \
        --model-path ./data/resnet18/long_double_descent \
        --data-path ./data \
        --k 64
"""

import argparse
import logging
from pathlib import Path

import torch

from jl.double_descent.calibration.sweep import run_calibration_sweep
from jl.double_descent.resnet18.evaluation import discover_models
from jl.double_descent.resnet18.l2_calibrate import extract_features
from jl.double_descent.resnet18.resnet18_config import DDConfig
from jl.double_descent.resnet18.resnet18_data import load_cifar10_with_noise
from jl.double_descent.resnet18.resnet18k import make_resnet18k

logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Calibrate ResNet18k on CIFAR-10"
    )
    parser.add_argument(
        "--model-path", type=str, required=True,
        help="Directory containing model_k*.pt files",
    )
    parser.add_argument(
        "--data-path", type=str, default="./data",
        help="Directory containing CIFAR-10 data",
    )
    parser.add_argument(
        "--k", type=int, required=True,
        help="ResNet18 width parameter k",
    )
    parser.add_argument(
        "--max-steps", type=int, default=100,
        help="Number of L-BFGS steps per lambda (default: 100)",
    )
    parser.add_argument(
        "--sweep-metric", type=str, default="ece", choices=["ece", "nll"],
        help="Metric to select best lambda (default: ece)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Seed for test set val/test split (default: 42)",
    )
    parser.add_argument(
        "--output-path", type=str, default=None,
        help="Directory for output artifacts (default: <model-path>/calibration_k<k>)",
    )
    args = parser.parse_args()

    # Discover and validate model
    models = discover_models(args.model_path)
    if args.k not in models:
        available = sorted(models.keys())
        raise ValueError(f"No model found for k={args.k}. Available: {available}")

    model_path = str(models[args.k])
    logger.info(f"Using model: {model_path}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load model and freeze
    model = make_resnet18k(k=args.k, num_classes=10).to(device)
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    original_layer_state = model.linear.state_dict()
    feature_dim = 8 * args.k

    # Extract training features
    config = DDConfig()
    train_loader, _ = load_cifar10_with_noise(
        noise_prob=config.label_noise,
        batch_size=config.batch_size,
        data_augmentation=False,
        data_dir=args.data_path,
    )

    logger.info("Extracting training features...")
    all_train_features = []
    all_train_labels = []
    with torch.no_grad():
        for images, labels in train_loader:
            feats = extract_features(model, images.to(device))
            all_train_features.append(feats.cpu())
            all_train_labels.append(labels)

    train_features = torch.cat(all_train_features)
    train_labels = torch.cat(all_train_labels)
    logger.info(f"Training features: {train_features.shape}")

    # Extract test features and split into val/test
    _, test_loader = load_cifar10_with_noise(
        noise_prob=0.0,
        batch_size=config.batch_size,
        data_augmentation=False,
        data_dir=args.data_path,
    )

    logger.info("Extracting test features...")
    all_test_features = []
    all_test_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            feats = extract_features(model, images.to(device))
            all_test_features.append(feats.cpu())
            all_test_labels.append(labels)

    full_test_features = torch.cat(all_test_features)
    full_test_labels = torch.cat(all_test_labels)

    # Split test set into val/test halves
    n_test = len(full_test_labels)
    perm = torch.randperm(n_test, generator=torch.Generator().manual_seed(args.seed))
    val_idx = perm[: n_test // 2]
    test_idx = perm[n_test // 2 :]

    val_features = full_test_features[val_idx]
    val_labels = full_test_labels[val_idx]
    test_features = full_test_features[test_idx]
    test_labels = full_test_labels[test_idx]
    logger.info(f"Val: {val_features.shape[0]} samples, Test: {test_features.shape[0]} samples")

    # Free model memory
    del model
    torch.cuda.empty_cache()

    # Determine output directory
    output_dir = args.output_path
    if output_dir is None:
        output_dir = str(Path(args.model_path) / f"calibration_k{args.k}")

    # Run calibration sweep
    lambdas = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2]
    run_calibration_sweep(
        train_features=train_features,
        train_labels=train_labels,
        val_features=val_features,
        val_labels=val_labels,
        test_features=test_features,
        test_labels=test_labels,
        original_head_state=original_layer_state,
        num_classes=10,
        feature_dim=feature_dim,
        lambdas=lambdas,
        max_steps=args.max_steps,
        sweep_metric=args.sweep_metric,
        device=device,
        output_dir=Path(output_dir),
    )


if __name__ == "__main__":
    main()
