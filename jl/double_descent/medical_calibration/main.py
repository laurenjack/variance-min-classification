"""Entry point for RETFound + APTOS-2019 calibration experiment.

Phase 1: Download data, create splits
Phase 2: Fine-tune RETFound on APTOS-2019
Phase 3: Save logits for val and test sets

Usage:
    python -m jl.double_descent.medical_calibration.main \
        --data-path ./data/aptos2019 \
        --output-path ./output/medical_calibration/04-03-1200
"""

import argparse
import json
import logging
from pathlib import Path

import torch
import torch.nn.functional as F

from jl.double_descent.medical_calibration.config import MedCalConfig
from jl.double_descent.medical_calibration.data import (
    build_dataloaders,
    create_split_manifest,
    download_aptos,
)
from jl.double_descent.medical_calibration.model import build_retfound_model
from jl.double_descent.medical_calibration.train import train

logger = logging.getLogger(__name__)


def save_logits(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    output_path: str,
    split_name: str,
):
    """Save logits, probabilities, and labels for a data split.

    Writes a single .pt file with:
        logits: [N, num_classes]
        probs: [N, num_classes]
        labels: [N]
        predictions: [N]
    """
    model.eval()
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            all_logits.append(logits.cpu())
            all_labels.append(labels)

    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    probs = F.softmax(logits, dim=-1)
    predictions = logits.argmax(dim=-1)

    out_file = Path(output_path) / f"{split_name}_logits.pt"
    torch.save(
        {
            "logits": logits,
            "probs": probs,
            "labels": labels,
            "predictions": predictions,
        },
        out_file,
    )
    logger.info(f"Saved {split_name} logits: {out_file} ({len(labels)} samples)")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="RETFound + APTOS-2019 calibration experiment"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="./data/aptos2019",
        help="Directory for APTOS-2019 data",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Directory for output (checkpoints, metrics, logits)",
    )
    args = parser.parse_args()

    config = MedCalConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Phase 1: Data setup
    logger.info("=== Phase 1: Data setup ===")
    data_dir = download_aptos(args.data_path)
    train_manifest, val_manifest, test_manifest = create_split_manifest(
        str(data_dir), args.output_path, config
    )

    # Build dataloaders
    train_loader, val_loader, test_loader = build_dataloaders(
        str(data_dir), train_manifest, val_manifest, test_manifest, config
    )
    logger.info(
        f"Data: train={len(train_loader.dataset)}, "
        f"val={len(val_loader.dataset)}, test={len(test_loader.dataset)}"
    )

    # Phase 2: Fine-tune
    logger.info("=== Phase 2: Fine-tuning ===")
    model = build_retfound_model(config, device)
    best_path = train(model, train_loader, val_loader, config, args.output_path, device)

    # Reload best checkpoint
    model.load_state_dict(
        torch.load(best_path, map_location=device, weights_only=True)
    )
    logger.info(f"Loaded best checkpoint: {best_path}")

    # Phase 3: Save logits
    logger.info("=== Phase 3: Saving logits ===")
    save_logits(model, val_loader, device, args.output_path, "val")
    save_logits(model, test_loader, device, args.output_path, "test")

    logger.info("=== Done ===")
    logger.info(f"Output: {args.output_path}")


if __name__ == "__main__":
    main()
