#!/usr/bin/env python3
"""Plot original vs fine-tuned test loss for ResNet18 and Transformer.

Produces a side-by-side figure showing that fine-tuning the final layer
with L2 regularization preserves the double descent curve shape.

Usage:
    python -m jl.double_descent.plot_fine_tune \
        --resnet-path ./data/resnet18/03-01-1010 \
        --transformer-path ./data/transformer/03-01-1010 \
        --output-dir ./data

Requires evaluation.jsonl (original metrics) and fine_tuned/ directory
in each model path. If evaluation.jsonl is missing, computes test loss
from saved models directly.
"""

import argparse
import json
import logging
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def _load_evaluation_jsonl(eval_path: Path) -> Dict:
    """Load evaluation.jsonl into a dict keyed by model width."""
    results = {}
    if not eval_path.exists():
        return results
    with open(eval_path) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                results[r.get("k") or r.get("d_model")] = r
    return results


# --- ResNet18 helpers ---


def _resnet_test_loss(model, test_loader, device) -> float:
    """Compute cross-entropy test loss for a ResNet18 model."""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = F.cross_entropy(logits, labels, reduction="sum")
            total_loss += loss.item()
            total_samples += labels.size(0)
    return total_loss / total_samples


def compute_resnet_losses(
    model_dir: str, data_path: str, device: torch.device
) -> Tuple[List[int], List[float], List[float]]:
    """Compute original and fine-tuned test losses for all ResNet18 models.

    Returns:
        (k_values, original_losses, fine_tuned_losses)
    """
    from jl.double_descent.resnet18.evaluation import discover_models
    from jl.double_descent.resnet18.resnet18_config import DDConfig
    from jl.double_descent.resnet18.resnet18_data import NoisyCIFAR10
    from jl.double_descent.resnet18.resnet18k import make_resnet18k
    import torchvision.transforms as transforms

    model_path = Path(model_dir)
    fine_tuned_dir = model_path / "fine_tuned"

    models = discover_models(model_dir)
    if not models:
        raise FileNotFoundError(f"No model_k*.pt files in {model_dir}")

    # Check fine-tuned layers exist
    for k in models:
        layer_path = fine_tuned_dir / f"layer_k{k}.pt"
        if not layer_path.exists():
            raise FileNotFoundError(f"Missing fine-tuned layer: {layer_path}")

    # Try reading original losses from evaluation.jsonl
    eval_results = _load_evaluation_jsonl(model_path / "evaluation.jsonl")

    # Load test data
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]
    )
    test_dataset = NoisyCIFAR10(
        root=data_path,
        train=False,
        noise_prob=0.0,
        transform=transforms.Compose([transforms.ToTensor(), normalize]),
    )
    config = DDConfig()
    test_loader = DataLoader(
        test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4
    )

    k_values = []
    original_losses = []
    fine_tuned_losses = []

    for k, orig_model_path in sorted(models.items()):
        logger.info(f"ResNet k={k}...")

        # Original test loss
        if k in eval_results and "test_loss" in eval_results[k]:
            orig_loss = eval_results[k]["test_loss"]
        else:
            model = make_resnet18k(k=k, num_classes=10).to(device)
            model.load_state_dict(
                torch.load(orig_model_path, map_location=device, weights_only=True)
            )
            orig_loss = _resnet_test_loss(model, test_loader, device)
            del model

        # Fine-tuned test loss
        model = make_resnet18k(k=k, num_classes=10).to(device)
        model.load_state_dict(
            torch.load(orig_model_path, map_location=device, weights_only=True)
        )
        layer_state = torch.load(
            fine_tuned_dir / f"layer_k{k}.pt", map_location=device, weights_only=True
        )
        model.linear.load_state_dict(layer_state)
        ft_loss = _resnet_test_loss(model, test_loader, device)
        del model

        k_values.append(k)
        original_losses.append(orig_loss)
        fine_tuned_losses.append(ft_loss)
        logger.info(f"  k={k}: original={orig_loss:.4f}, fine_tuned={ft_loss:.4f}")

    torch.cuda.empty_cache()
    return k_values, original_losses, fine_tuned_losses


# --- Transformer helpers ---


def _transformer_test_loss(model, test_loader, pad_idx, device) -> float:
    """Compute cross-entropy test loss for a Transformer model."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for src, tgt in test_loader:
            src, tgt = src.to(device), tgt.to(device)
            logits = model(src, tgt[:, :-1])
            target = tgt[:, 1:].contiguous()
            mask = target != pad_idx
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                target.view(-1),
                ignore_index=pad_idx,
                reduction="sum",
            )
            total_loss += loss.item()
            total_tokens += mask.sum().item()
    return total_loss / total_tokens


def compute_transformer_losses(
    model_dir: str, data_path: str, device: torch.device
) -> Tuple[List[int], List[float], List[float]]:
    """Compute original and fine-tuned test losses for all Transformer models.

    Returns:
        (d_model_values, original_losses, fine_tuned_losses)
    """
    from jl.double_descent.transformer.evaluation import discover_models
    from jl.double_descent.transformer.transformer_config import TDDConfig
    from jl.double_descent.transformer.transformer_data import (
        build_vocab,
        collate_fn,
        load_split,
    )
    from jl.double_descent.transformer.transformer_data import TranslationDataset
    from jl.double_descent.transformer.transformer_model import TransformerModel

    model_path = Path(model_dir)
    fine_tuned_dir = model_path / "fine_tuned"

    models = discover_models(model_dir)
    if not models:
        raise FileNotFoundError(f"No model_d*_*k.pt files in {model_dir}")

    # Check fine-tuned layers exist
    for (d_model, train_samples) in models:
        samples_k = train_samples // 1000
        layer_path = fine_tuned_dir / f"layer_d{d_model}_{samples_k}k.pt"
        if not layer_path.exists():
            raise FileNotFoundError(f"Missing fine-tuned layer: {layer_path}")

    # Try reading original losses from evaluation.jsonl
    eval_results = _load_evaluation_jsonl(model_path / "evaluation.jsonl")

    # Load test data
    config = TDDConfig()
    vocab = build_vocab(data_path)
    test_src, test_tgt = load_split(data_path, "test")
    test_dataset = TranslationDataset(test_src, test_tgt, vocab)
    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
        collate_fn=partial(collate_fn, pad_idx=vocab.pad_idx),
    )

    d_model_values = []
    original_losses = []
    fine_tuned_losses = []

    for (d_model, train_samples), orig_model_path in sorted(models.items()):
        samples_k = train_samples // 1000
        logger.info(f"Transformer d_model={d_model}...")

        def _make_model():
            return TransformerModel(
                vocab_size=len(vocab),
                d_model=d_model,
                n_layers=config.n_layers,
                n_heads=config.n_heads,
                d_ff_multiplier=config.d_ff_multiplier,
                pad_idx=vocab.pad_idx,
            ).to(device)

        # Original test loss
        if d_model in eval_results and "test_loss" in eval_results[d_model]:
            orig_loss = eval_results[d_model]["test_loss"]
        else:
            model = _make_model()
            model.load_state_dict(
                torch.load(orig_model_path, map_location=device, weights_only=True)
            )
            orig_loss = _transformer_test_loss(model, test_loader, vocab.pad_idx, device)
            del model

        # Fine-tuned test loss: load original, replace output_proj with untied fine-tuned layer
        model = _make_model()
        model.load_state_dict(
            torch.load(orig_model_path, map_location=device, weights_only=True)
        )
        new_linear = nn.Linear(d_model, len(vocab), bias=False).to(device)
        layer_state = torch.load(
            fine_tuned_dir / f"layer_d{d_model}_{samples_k}k.pt",
            map_location=device,
            weights_only=True,
        )
        new_linear.load_state_dict(layer_state)
        model.output_proj = new_linear  # Untie from embedding, use fine-tuned weights
        ft_loss = _transformer_test_loss(model, test_loader, vocab.pad_idx, device)
        del model

        d_model_values.append(d_model)
        original_losses.append(orig_loss)
        fine_tuned_losses.append(ft_loss)
        logger.info(f"  d={d_model}: original={orig_loss:.4f}, fine_tuned={ft_loss:.4f}")

    torch.cuda.empty_cache()
    return d_model_values, original_losses, fine_tuned_losses


# --- Plotting ---


def plot_fine_tune_comparison(
    resnet_data: Optional[Tuple[List[int], List[float], List[float]]],
    transformer_data: Optional[Tuple[List[int], List[float], List[float]]],
    output_dir: str,
) -> None:
    """Plot side-by-side original vs fine-tuned test loss.

    Args:
        resnet_data: (k_values, original_losses, fine_tuned_losses) or None.
        transformer_data: (d_model_values, original_losses, fine_tuned_losses) or None.
        output_dir: Directory to save plot.
    """
    num_panels = sum(1 for d in [resnet_data, transformer_data] if d is not None)
    if num_panels == 0:
        raise ValueError("No data to plot")

    fig, axes = plt.subplots(1, num_panels, figsize=(7 * num_panels, 5), dpi=150)
    if num_panels == 1:
        axes = [axes]

    panel_idx = 0

    if resnet_data is not None:
        ax = axes[panel_idx]
        k_vals, orig, ft = resnet_data
        ax.plot(k_vals, orig, "-o", color="blue", lw=2, label="Original")
        ax.plot(k_vals, ft, "--s", color="red", lw=2, label="Fine-tuned")
        ax.set_xlabel("ResNet18 width parameter k")
        ax.set_ylabel("Test Cross-Entropy Loss")
        ax.set_title("ResNet18 on CIFAR-10 (15% label noise)")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)
        panel_idx += 1

    if transformer_data is not None:
        ax = axes[panel_idx]
        d_vals, orig, ft = transformer_data
        ax.plot(d_vals, orig, "-o", color="blue", lw=2, label="Original")
        ax.plot(d_vals, ft, "--s", color="red", lw=2, label="Fine-tuned")
        ax.set_xlabel("Transformer embedding dimension d_model")
        ax.set_ylabel("Test Cross-Entropy Loss")
        ax.set_title("Transformer on IWSLT14 de-en")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)

    plt.tight_layout()
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_path = Path(output_dir) / "fine_tune_comparison.png"
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to {output_path}")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Plot original vs fine-tuned test loss for ResNet18 and Transformer"
    )
    parser.add_argument(
        "--resnet-path",
        type=str,
        default=None,
        help="Directory containing ResNet18 models and fine_tuned/ subdirectory",
    )
    parser.add_argument(
        "--transformer-path",
        type=str,
        default=None,
        help="Directory containing Transformer models and fine_tuned/ subdirectory",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="./data",
        help="Directory containing CIFAR-10 data (for ResNet evaluation)",
    )
    parser.add_argument(
        "--transformer-data-path",
        type=str,
        default=None,
        help="Directory containing preprocessed IWSLT data (for Transformer evaluation)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data",
        help="Directory to save plot",
    )
    args = parser.parse_args()

    if args.resnet_path is None and args.transformer_path is None:
        parser.error("At least one of --resnet-path or --transformer-path is required")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    resnet_data = None
    transformer_data = None

    if args.resnet_path:
        logger.info("Computing ResNet18 losses...")
        resnet_data = compute_resnet_losses(args.resnet_path, args.data_path, device)

    if args.transformer_path:
        t_data_path = args.transformer_data_path
        if t_data_path is None:
            parser.error("--transformer-data-path is required when --transformer-path is set")
        logger.info("Computing Transformer losses...")
        transformer_data = compute_transformer_losses(
            args.transformer_path, t_data_path, device
        )

    plot_fine_tune_comparison(resnet_data, transformer_data, args.output_dir)


if __name__ == "__main__":
    main()
