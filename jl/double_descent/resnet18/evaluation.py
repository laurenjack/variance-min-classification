"""Evaluation for ResNet18 main (non-variance) training runs.

Computes test metrics and ECE on saved models, consolidates with train metrics
from existing metrics_k*.jsonl files, and writes to evaluation.jsonl.

Dual-use:
  - Called from trainer.py at end of training
  - Standalone: discovers model_k*.pt files and evaluates all

Usage:
    python -m jl.double_descent.resnet18.evaluation \
        --model-path ./output/resnet18/03-01-1010 \
        --data-path ./data
"""

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from jl.double_descent.resnet18.resnet18_data import NoisyCIFAR10
from jl.double_descent.resnet18.resnet18k import make_resnet18k
from jl.double_descent.temperature_scaling import fit_temperature, metrics_with_temperature

logger = logging.getLogger(__name__)

EVAL_BATCH_SIZE = 256
ECE_NUM_BINS = 20


def compute_ece(confidences: torch.Tensor, correct: torch.Tensor, num_bins: int = ECE_NUM_BINS) -> float:
    """Compute Expected Calibration Error with equal-width bins.

    Args:
        confidences: Tensor of max softmax probabilities per prediction.
        correct: Boolean tensor of whether each prediction was correct.
        num_bins: Number of equal-width bins in [0, 1].

    Returns:
        ECE as a float.
    """
    bin_boundaries = torch.linspace(0, 1, num_bins + 1)
    n = len(confidences)
    ece = 0.0
    for i in range(num_bins):
        lo, hi = bin_boundaries[i].item(), bin_boundaries[i + 1].item()
        mask = (confidences > lo) & (confidences <= hi)
        if i == 0:
            mask = mask | (confidences == lo)
        n_bin = mask.sum().item()
        if n_bin > 0:
            avg_confidence = confidences[mask].mean().item()
            avg_accuracy = correct[mask].float().mean().item()
            ece += (n_bin / n) * abs(avg_accuracy - avg_confidence)
    return ece


def read_final_train_metrics(metrics_path: Path) -> Dict[str, float]:
    """Read final train metrics from metrics_k*.jsonl file.

    Args:
        metrics_path: Path to the metrics JSONL file.

    Returns:
        Dict with train_loss and train_error from the last epoch.

    Raises:
        FileNotFoundError: If metrics file doesn't exist.
        ValueError: If metrics file is empty or missing required fields.
    """
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")

    last_line = None
    with open(metrics_path, 'r') as f:
        for line in f:
            if line.strip():
                last_line = line

    if last_line is None:
        raise ValueError(f"Metrics file is empty: {metrics_path}")

    metrics = json.loads(last_line)

    if 'train_loss' not in metrics or 'train_error' not in metrics:
        raise ValueError(f"Metrics file missing train_loss or train_error: {metrics_path}")

    return {
        'train_loss': metrics['train_loss'],
        'train_error': metrics['train_error'],
    }


def _metrics_pass(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float, float]:
    """Forward a loader and return (avg_loss, error_rate, ECE)."""
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    all_confidences = []
    all_correct = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            batch_size = images.size(0)

            logits = model(images)
            loss = F.cross_entropy(logits, labels, reduction='sum')
            total_loss += loss.item()

            probs = F.softmax(logits, dim=-1)
            max_probs, predictions = probs.max(dim=-1)
            correct = predictions == labels

            total_correct += correct.sum().item()
            total_samples += batch_size
            all_confidences.append(max_probs.cpu())
            all_correct.append(correct.cpu())

    avg_loss = total_loss / total_samples
    error = 1.0 - (total_correct / total_samples)
    ece = compute_ece(torch.cat(all_confidences), torch.cat(all_correct))
    return avg_loss, error, ece


def _collect_logits(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Forward pass collecting all logits and labels as tensors."""
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            all_logits.append(logits.cpu())
            all_labels.append(labels)
    return torch.cat(all_logits), torch.cat(all_labels)


def compute_final_metrics(
    model: torch.nn.Module,
    test_loader: DataLoader,
    metrics_path: Path,
    output_path: Path,
    model_label: str,
    model_params: Dict,
    device: torch.device,
    val_loader: "Optional[DataLoader]" = None,
) -> Dict:
    """Compute final metrics for a trained model.

    1. Runs forward pass on test set to compute test_loss, test_error, ECE
    2. (Optional) Runs forward pass on val set to compute val_loss, val_error,
       val_ece — only when val_loader is provided (val-split mode).
    3. Reads final train_loss, train_error from metrics_path
    4. Appends one JSON line to output_path/evaluation.jsonl

    Args:
        model: Trained model (already on device, in eval mode).
        test_loader: DataLoader for test set.
        metrics_path: Path to metrics file.
        output_path: Directory to write evaluation.jsonl.
        model_label: Label for logging, e.g. "k4" or "n3".
        model_params: Dict of model params to include in result, e.g. {"k": 4}.
        device: Device model is on.
        val_loader: Optional DataLoader for the held-out val split. When
            provided, val_loss / val_error / val_ece are added to the
            evaluation row.

    Returns:
        Dict of computed metrics.
    """
    model.eval()

    test_loss, test_error, ece = _metrics_pass(model, test_loader, device)

    # Read train metrics from metrics file
    train_metrics = read_final_train_metrics(metrics_path)

    result = {
        **model_params,
        'test_loss': round(test_loss, 6),
        'test_error': round(test_error, 6),
        'train_loss': round(train_metrics['train_loss'], 6),
        'train_error': round(train_metrics['train_error'], 6),
        'ece': round(ece, 6),
    }

    if val_loader is not None:
        val_loss, val_error, val_ece = _metrics_pass(model, val_loader, device)
        result['val_loss'] = round(val_loss, 6)
        result['val_error'] = round(val_error, 6)
        result['val_ece'] = round(val_ece, 6)

        # Temperature scaling: fit T on val logits, evaluate on test logits
        val_logits, val_labels = _collect_logits(model, val_loader, device)
        test_logits, test_labels = _collect_logits(model, test_loader, device)
        temperature = fit_temperature(val_logits, val_labels)
        ts_metrics = metrics_with_temperature(
            test_logits, test_labels, temperature, compute_ece_fn=compute_ece,
        )
        result['temperature'] = round(temperature, 6)
        result['ts_loss'] = round(ts_metrics['ts_loss'], 6)
        result['ts_error'] = round(ts_metrics['ts_error'], 6)
        result['ts_ece'] = round(ts_metrics['ts_ece'], 6)

    # Append to evaluation.jsonl
    eval_file = output_path / 'evaluation.jsonl'
    with open(eval_file, 'a') as f:
        f.write(json.dumps(result) + '\n')

    val_str = ""
    if val_loader is not None:
        val_str = (
            f", val_loss={result['val_loss']:.4f}, "
            f"val_error={result['val_error']:.4f}, val_ece={result['val_ece']:.6f}, "
            f"T={result['temperature']:.4f}, ts_loss={result['ts_loss']:.4f}"
        )
    logger.info(
        f"{model_label}: test_loss={test_loss:.4f}, test_error={test_error:.4f}, "
        f"ece={ece:.6f}{val_str}"
    )

    return result


def discover_models(model_dir: str) -> Dict[int, Path]:
    """Discover main (non-variance) model files.

    Returns:
        Dict mapping k -> model file Path.
    """
    path = Path(model_dir)
    # Match model_k*.pt but NOT model_k*_split*.pt
    model_files = sorted(path.glob("model_k*.pt"))

    models: Dict[int, Path] = {}
    for f in model_files:
        # Skip variance models (have _split in name)
        if '_split' in f.name:
            continue
        match = re.match(r"model_k(\d+)\.pt", f.name)
        if match:
            k = int(match.group(1))
            models[k] = f

    return dict(sorted(models.items()))


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Evaluate ResNet18 main training runs"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Directory containing model_k*.pt and metrics_k*.jsonl files",
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

    # Discover models
    models = discover_models(args.model_path)
    if not models:
        raise FileNotFoundError(
            f"No model_k*.pt files found in {args.model_path}"
        )
    logger.info(f"Found models for k values: {list(models.keys())}")

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

    # Load val set from val.pt if present (enables temperature scaling)
    val_loader = None
    val_path = Path(args.model_path) / "val.pt"
    if val_path.exists():
        val_data = torch.load(val_path, weights_only=True)
        val_dataset = torch.utils.data.TensorDataset(
            val_data["images"], val_data["labels"],
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=EVAL_BATCH_SIZE,
            shuffle=False,
        )
        logger.info(f"Val set loaded from {val_path}: {len(val_dataset)} samples")
    else:
        logger.info("No val.pt found — skipping temperature scaling")

    # Clear existing evaluation file (overwrite mode)
    output_path = Path(args.model_path)
    eval_file = output_path / 'evaluation.jsonl'
    if eval_file.exists():
        eval_file.unlink()

    # Evaluate each model
    for k, model_path in models.items():
        logger.info(f"Evaluating k={k}...")

        # Load model
        model = make_resnet18k(k=k, num_classes=10).to(device)
        model.load_state_dict(
            torch.load(model_path, map_location=device, weights_only=True)
        )
        model.eval()

        # Get metrics file path
        metrics_path = output_path / f"metrics_k{k}.jsonl"

        # Compute and save metrics
        compute_final_metrics(
            model, test_loader, metrics_path, output_path,
            model_label=f"k{k}", model_params={"k": k}, device=device,
            val_loader=val_loader,
        )

        # Clean up
        del model
        torch.cuda.empty_cache()

    logger.info(f"Evaluation complete. Results written to {eval_file}")


if __name__ == "__main__":
    main()
