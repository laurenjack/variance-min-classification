"""Evaluate variance across training splits for ResNet18 models.

Loads all variance-mode models (model_k*_split*.pt), runs them on the test set,
and computes:
  - Mean test loss across splits
  - Jensen Gap: E[log(q_bar[y] / q_j[y])] - the variance term in bias-variance decomposition

Output: evaluation.jsonl written alongside the model files.

Usage:
    python -m jl.double_descent.resnet18.variance_evaluation \
        --model-path ./output/resnet18_variance/03-01-1010 \
        --data-path ./data
"""

import argparse
import json
import logging
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from jl.double_descent.resnet18.resnet18_data import NoisyCIFAR10
from jl.double_descent.resnet18.resnet18k import make_resnet18k
import torchvision.transforms as transforms

logger = logging.getLogger(__name__)

EVAL_BATCH_SIZE = 256


def discover_models(model_dir: str) -> Dict[int, List[Path]]:
    """Discover variance model files grouped by k.

    Returns:
        Dict mapping k -> sorted list of model file Paths.
    """
    path = Path(model_dir)
    model_files = sorted(path.glob("model_k*_split*.pt"))

    grouped: Dict[int, List[Path]] = defaultdict(list)
    for f in model_files:
        match = re.match(r"model_k(\d+)_split(\d+)\.pt", f.name)
        if match:
            k = int(match.group(1))
            grouped[k].append(f)

    return dict(sorted(grouped.items()))


def load_model(
    model_path: Path,
    k: int,
    device: torch.device,
) -> torch.nn.Module:
    """Instantiate model architecture and load saved weights."""
    model = make_resnet18k(k=k, num_classes=10).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    return model


def evaluate_k(
    model_paths: List[Path],
    k: int,
    test_loader: DataLoader,
    device: torch.device,
) -> Dict:
    """Compute mean test loss and Jensen Gap for one k value.

    For each test batch:
      1. Forward pass all models, collect softmax distributions
      2. Compute q_bar = mean distribution across models
      3. Compute Jensen Gap: log(q_bar[y] / q_j[y]) for each model

    """
    num_models = len(model_paths)
    logger.info(f"k={k}: loading {num_models} models")

    models = [load_model(p, k, device) for p in model_paths]

    total_loss_per_model = [0.0] * num_models
    total_jensen_gap = 0.0
    total_samples = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            batch_size = images.size(0)

            all_log_probs = []
            for j, model in enumerate(models):
                logits = model(images)  # [B, 10]

                loss = F.cross_entropy(logits, labels, reduction='sum')
                total_loss_per_model[j] += loss.item()

                log_probs = F.log_softmax(logits, dim=-1)  # [B, 10]
                all_log_probs.append(log_probs)

            # Compute Jensen Gap in log space: log(q_bar[y]) - log(q_j[y])
            all_log_probs_t = torch.stack(all_log_probs, dim=0)  # [M, B, 10]
            log_q_bar = torch.logsumexp(all_log_probs_t, dim=0) - math.log(num_models)  # [B, 10]

            log_q_bar_y = log_q_bar.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)  # [B]

            batch_jensen = 0.0
            for j in range(num_models):
                log_q_j_y = all_log_probs_t[j].gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)  # [B]
                jensen_per_sample = log_q_bar_y - log_q_j_y  # [B]
                batch_jensen += jensen_per_sample.sum().item()

            total_jensen_gap += batch_jensen / num_models
            total_samples += batch_size

    del models
    torch.cuda.empty_cache()

    mean_test_loss = (
        sum(total_loss_per_model) / (num_models * total_samples)
        if total_samples > 0
        else 0.0
    )
    mean_jensen_gap = total_jensen_gap / total_samples if total_samples > 0 else 0.0

    logger.info(
        f"k={k}: mean_test_loss={mean_test_loss:.4f}, "
        f"mean_jensen_gap={mean_jensen_gap:.6f}"
    )

    return {
        "k": k,
        "mean_test_loss": round(mean_test_loss, 6),
        "mean_jensen_gap": round(mean_jensen_gap, 6),
        "num_models": num_models,
        "total_samples": total_samples,
    }


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Evaluate variance across training splits for ResNet18 models"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Directory containing model_k*_split*.pt files",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="./data",
        help="Directory containing CIFAR-10 data",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    grouped = discover_models(args.model_path)
    if not grouped:
        raise FileNotFoundError(
            f"No model_k*_split*.pt files found in {args.model_path}"
        )
    logger.info(
        f"Found models for k values: {list(grouped.keys())} "
        f"({sum(len(v) for v in grouped.values())} total)"
    )

    # Load test set
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
    )
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    test_dataset = NoisyCIFAR10(
        root=args.data_path,
        train=False,
        noise_prob=0.0,
        transform=test_transform,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    logger.info(f"Test set: {len(test_dataset)} samples")

    output_path = Path(args.model_path) / "evaluation.jsonl"
    if output_path.exists():
        output_path.unlink()

    for k, model_paths in grouped.items():
        result = evaluate_k(model_paths, k, test_loader, device)
        with open(output_path, "a") as fh:
            fh.write(json.dumps(result) + "\n")

    logger.info(f"Results written to {output_path}")


if __name__ == "__main__":
    main()
