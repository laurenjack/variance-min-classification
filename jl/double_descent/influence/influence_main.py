#!/usr/bin/env python3
"""Training point influence decomposition for ResNet18 double descent.

For each saved ResNet18k model, this script:
1. Extracts penultimate features (frozen backbone)
2. Fine-tunes the final linear layer with L2 regularization (L-BFGS)
3. Validates the Yeh & Kim (2018) analytic decomposition
4. Computes per-training-point influence scores
5. Saves results and generates Figure X (original vs fine-tuned loss)

Usage:
    python -m jl.double_descent.influence.influence_main \
        --model-dir ./data/resnet18/04-11-1602 \
        --data-path ./data \
        --lambda-l2 1e-4 \
        --output-dir ./data/resnet18/04-11-1602/influence
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from jl.double_descent.resnet18.evaluation import (
    _metrics_pass,
    discover_models,
)
from jl.double_descent.resnet18.resnet18_data import (
    NoisyCIFAR10,
    compute_val_split_indices,
)
from jl.double_descent.resnet18.resnet18k import make_resnet18k

from .decompose import (
    build_mislabel_mask,
    compute_influence_scores,
    compute_residuals,
    compute_summary_stats,
    save_influence_jsonl,
)
from .l2_finetune import extract_features, l2_finetune
from .validate import plot_figure_x, validate_decomposition

logger = logging.getLogger(__name__)


def process_k(
    k: int,
    model_path: Path,
    train_loader: DataLoader,
    test_loader: DataLoader,
    mislabel_mask: np.ndarray,
    train_indices: np.ndarray,
    noisy_labels: list,
    original_labels: list,
    lambda_l2: float,
    output_dir: Path,
    device: torch.device,
) -> dict:
    """Run the full influence pipeline for one k value.

    Returns a summary dict with influence stats and fine-tuned metrics.
    """
    logger.info(f"--- k={k} ---")

    # Load model
    model = make_resnet18k(k=k, num_classes=10).to(device)
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )
    model.eval()

    # Extract features
    logger.info(f"Extracting features (dim={8*k})...")
    phi_train, train_labels = extract_features(model, train_loader, device)
    phi_test, _ = extract_features(model, test_loader, device)
    logger.info(
        f"Features: train {phi_train.shape}, test {phi_test.shape}"
    )

    # L2 fine-tune
    logger.info("Fine-tuning final layer with L-BFGS...")
    grad_norm = l2_finetune(
        model, phi_train, train_labels, lambda_l2=lambda_l2
    )

    # Validate decomposition identity
    val_record = validate_decomposition(
        phi_train, train_labels, model.linear, lambda_l2
    )
    val_record["grad_norm"] = grad_norm

    # Save validation
    val_path = output_dir / f"validation_k{k}.json"
    with open(val_path, "w") as f:
        json.dump(val_record, f, indent=2)

    # Compute fine-tuned test metrics for Figure X
    ft_test_loss, ft_test_error, _ = _metrics_pass(
        model, test_loader, device
    )
    ft_metrics = {"test_loss": ft_test_loss}
    logger.info(f"Fine-tuned test_loss={ft_test_loss:.4f}")

    # Compute influence scores
    logger.info("Computing influence scores...")
    residuals = compute_residuals(phi_train, train_labels, model.linear)
    influence = compute_influence_scores(phi_train, phi_test, residuals)

    # Save per-point influence
    save_influence_jsonl(
        influence, mislabel_mask, train_indices,
        noisy_labels, original_labels, output_dir, k,
    )

    # Save fine-tuned linear layer
    ft_path = output_dir / f"finetune_k{k}.pt"
    torch.save(model.linear.state_dict(), ft_path)

    # Summary stats
    stats = compute_summary_stats(influence, mislabel_mask)
    logger.info(
        f"Influence ratio (mislabeled/clean): {stats['influence_ratio']:.3f}"
    )

    # Clean up GPU memory
    del phi_train, phi_test, residuals, influence, model
    torch.cuda.empty_cache()

    return {"metrics": ft_metrics, **stats}


def main():
    parser = argparse.ArgumentParser(
        description="Training point influence decomposition"
    )
    parser.add_argument(
        "--model-dir", required=True,
        help="Directory containing model_k*.pt and evaluation.jsonl",
    )
    parser.add_argument(
        "--data-path", default="./data",
        help="Root directory for CIFAR-10 data",
    )
    parser.add_argument(
        "--lambda-l2", type=float, default=1e-4,
        help="L2 regularization strength for fine-tuning",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Output directory (default: <model-dir>/influence)",
    )
    parser.add_argument(
        "--ts-eval-path", default=None,
        help="Path to temperature_scaled_evaluation.jsonl for Figure X",
    )
    parser.add_argument(
        "--device", default=None,
        help="Device (default: cuda if available, else cpu)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir) if args.output_dir else model_dir / "influence"
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logger.info(f"Using device: {device}")

    # Discover models
    models = discover_models(str(model_dir))
    logger.info(f"Found {len(models)} models: k={list(models.keys())}")

    # Load data with test-time transforms (no augmentation)
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
    )
    test_transform = transforms.Compose([transforms.ToTensor(), normalize])

    # Training set: full 50K with noise, then subset to 45K train indices
    noisy_train_full = NoisyCIFAR10(
        root=args.data_path, train=True, noise_prob=0.15,
        transform=test_transform, seed=42,
    )
    train_indices, _ = compute_val_split_indices()
    train_subset = Subset(noisy_train_full, train_indices.tolist())

    train_loader = DataLoader(
        train_subset, batch_size=256, shuffle=False, num_workers=2,
    )

    # Test set: 10K CIFAR-10 test (no noise)
    test_dataset = NoisyCIFAR10(
        root=args.data_path, train=False, noise_prob=0.0,
        transform=test_transform, seed=42,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=256, shuffle=False, num_workers=2,
    )

    # Build mislabel mask (same for all k)
    mislabel_mask = build_mislabel_mask(
        noisy_train_full.labels,
        noisy_train_full.cifar.targets,
        train_indices,
    )
    n_mislabeled = mislabel_mask.sum()
    logger.info(
        f"Mislabel mask: {n_mislabeled}/{len(mislabel_mask)} "
        f"({100*n_mislabeled/len(mislabel_mask):.1f}%) corrupted"
    )

    # Process each k
    finetuned_metrics = {}
    summary_records = []

    for k, model_path in sorted(models.items()):
        result = process_k(
            k=k,
            model_path=model_path,
            train_loader=train_loader,
            test_loader=test_loader,
            mislabel_mask=mislabel_mask,
            train_indices=train_indices,
            noisy_labels=noisy_train_full.labels,
            original_labels=noisy_train_full.cifar.targets,
            lambda_l2=args.lambda_l2,
            output_dir=output_dir,
            device=device,
        )
        finetuned_metrics[k] = result.pop("metrics")
        summary_records.append({"k": k, **result})

    # Write summary
    summary_path = output_dir / "summary.jsonl"
    with open(summary_path, "w") as f:
        for rec in summary_records:
            f.write(json.dumps(rec) + "\n")
    logger.info(f"Saved summary to {summary_path}")

    # Plot Figure X (temperature-scaled vs L2 fine-tuned test loss)
    ts_eval_path = (
        Path(args.ts_eval_path) if args.ts_eval_path
        else model_dir / "temperature_scaled" / "temperature_scaled_evaluation.jsonl"
    )
    if ts_eval_path.exists():
        plot_figure_x(
            ts_eval_path, finetuned_metrics,
            output_dir / "figure_x.png",
        )
    else:
        logger.warning(
            f"No temperature_scaled_evaluation.jsonl found at {ts_eval_path}, "
            "skipping Figure X"
        )

    logger.info("Done.")


if __name__ == "__main__":
    main()
