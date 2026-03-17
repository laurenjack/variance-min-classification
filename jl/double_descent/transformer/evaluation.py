"""Evaluation for Transformer main (non-variance) training runs.

Computes test metrics, BLEU score, and per-token ECE on saved models,
consolidates with train metrics from existing metrics_d*_*k.jsonl files,
and writes to evaluation.jsonl.

Dual-use:
  - Called from trainer.py at end of training
  - Standalone: discovers model_d*_*k.pt files and evaluates all

Usage:
    python -m jl.double_descent.transformer.evaluation \
        --model-path ./output/transformer/03-01-1010 \
        --data-path ./data/iwslt14.tokenized.de-en
"""

import argparse
import json
import logging
import re
from functools import partial
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from jl.double_descent.transformer.bleu import compute_bleu
from jl.double_descent.transformer.transformer_config import TDDConfig
from jl.double_descent.transformer.transformer_data import (
    TranslationDataset,
    Vocab,
    build_vocab,
    collate_fn,
    load_split,
)
from jl.double_descent.transformer.transformer_model import TransformerModel

logger = logging.getLogger(__name__)

EVAL_BATCH_SIZE = 32
ECE_NUM_BINS = 20


def compute_ece(confidences: torch.Tensor, correct: torch.Tensor, num_bins: int = ECE_NUM_BINS) -> float:
    """Compute per-token Expected Calibration Error with equal-width bins.

    Args:
        confidences: Tensor of max softmax probabilities per token.
        correct: Boolean tensor of whether each token prediction was correct.
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


def _prepare_batch(tgt: torch.Tensor, pad_idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Split target tensor into decoder input/output and compute padding mask."""
    tgt_input = tgt[:, :-1]
    target = tgt[:, 1:].contiguous()
    mask = target != pad_idx
    num_tokens = mask.sum().item()
    return tgt_input, target, mask, num_tokens


def read_final_train_metrics(metrics_path: Path) -> Dict[str, float]:
    """Read final train metrics from metrics_d*_*k.jsonl file.

    For transformer, we look for the last entry with test_bleu (the final evaluation entry).
    We extract train_loss and train_acc from that entry.

    Args:
        metrics_path: Path to the metrics JSONL file.

    Returns:
        Dict with train_loss and train_acc from the final evaluation.

    Raises:
        FileNotFoundError: If metrics file doesn't exist.
        ValueError: If metrics file is empty or missing required fields.
    """
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")

    final_metrics = None
    with open(metrics_path, 'r') as f:
        for line in f:
            if line.strip():
                entry = json.loads(line)
                # Look for entries with test_bleu (indicates final evaluation)
                if 'test_bleu' in entry:
                    final_metrics = entry

    if final_metrics is None:
        raise ValueError(f"No final evaluation entry found in: {metrics_path}")

    if 'train_loss' not in final_metrics or 'train_acc' not in final_metrics:
        raise ValueError(f"Final entry missing train_loss or train_acc: {metrics_path}")

    return {
        'train_loss': final_metrics['train_loss'],
        'train_acc': final_metrics['train_acc'],
    }


def compute_final_metrics(
    model: TransformerModel,
    test_dataset: TranslationDataset,
    vocab: Vocab,
    metrics_path: Path,
    output_path: Path,
    d_model: int,
    train_samples: int,
    device: torch.device,
) -> Dict:
    """Compute final metrics for a trained transformer model.

    1. Runs forward pass on test set to compute test_loss, test_acc, ECE
    2. Computes test BLEU score
    3. Reads final train_loss, train_acc from metrics_path
    4. Appends one JSON line to output_path/evaluation.jsonl

    Args:
        model: Trained model (already on device, in eval mode).
        test_dataset: Test dataset.
        vocab: Vocabulary.
        metrics_path: Path to metrics_d{d_model}_{samples}k.jsonl file.
        output_path: Directory to write evaluation.jsonl.
        d_model: Model embedding dimension.
        train_samples: Number of training samples used.
        device: Device model is on.

    Returns:
        Dict of computed metrics.
    """
    model.eval()
    config = TDDConfig()

    criterion = nn.CrossEntropyLoss(
        ignore_index=vocab.pad_idx,
        label_smoothing=config.label_smoothing or 0.0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        collate_fn=partial(collate_fn, pad_idx=vocab.pad_idx),
    )

    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    all_confidences = []
    all_correct = []

    with torch.no_grad():
        for src, tgt in test_loader:
            src = src.to(device)
            tgt = tgt.to(device)
            tgt_input, target, mask, num_tokens = _prepare_batch(tgt, vocab.pad_idx)

            logits = model(src, tgt_input)

            # Compute loss
            loss = criterion(
                logits.view(-1, logits.size(-1)),
                target.view(-1),
            )
            total_loss += loss.item() * num_tokens

            # Compute accuracy (excluding padding)
            probs = F.softmax(logits, dim=-1)
            max_probs, predictions = probs.max(dim=-1)
            correct = (predictions == target) & mask

            total_correct += correct.sum().item()
            total_tokens += num_tokens

            # Collect ECE inputs (only non-pad tokens)
            all_confidences.append(max_probs[mask].cpu())
            all_correct.append(correct[mask].cpu())

    test_loss = total_loss / total_tokens if total_tokens > 0 else 0.0
    test_acc = total_correct / total_tokens if total_tokens > 0 else 0.0

    # Compute ECE
    confidences = torch.cat(all_confidences)
    correct = torch.cat(all_correct)
    ece = compute_ece(confidences, correct)

    # Compute BLEU
    logger.info(f"d_model={d_model}: Computing BLEU score...")
    test_bleu = compute_bleu(model, test_dataset, vocab, device, max_len=128)

    # Read train metrics from metrics file
    train_metrics = read_final_train_metrics(metrics_path)

    result = {
        'd_model': d_model,
        'train_samples': train_samples,
        'test_loss': round(test_loss, 6),
        'test_acc': round(test_acc, 6),
        'test_bleu': round(test_bleu, 2),
        'train_loss': round(train_metrics['train_loss'], 6),
        'train_acc': round(train_metrics['train_acc'], 6),
        'ece': round(ece, 6),
    }

    # Append to evaluation.jsonl
    eval_file = output_path / 'evaluation.jsonl'
    with open(eval_file, 'a') as f:
        f.write(json.dumps(result) + '\n')

    logger.info(
        f"d_model={d_model}: test_loss={test_loss:.4f}, test_acc={test_acc:.4f}, "
        f"test_bleu={test_bleu:.2f}, ece={ece:.6f}"
    )

    return result


def discover_models(model_dir: str) -> Dict[Tuple[int, int], Path]:
    """Discover main (non-variance) model files.

    Returns:
        Dict mapping (d_model, train_samples) -> model file Path.
    """
    path = Path(model_dir)
    # Match model_d*_*k.pt but NOT model_d*_split*.pt
    model_files = sorted(path.glob("model_d*_*k.pt"))

    models: Dict[Tuple[int, int], Path] = {}
    for f in model_files:
        # Skip variance models (have _split in name)
        if '_split' in f.name:
            continue
        match = re.match(r"model_d(\d+)_(\d+)k\.pt", f.name)
        if match:
            d_model = int(match.group(1))
            train_samples = int(match.group(2)) * 1000
            models[(d_model, train_samples)] = f

    return dict(sorted(models.items()))


def _load_test_data(data_path: str) -> Tuple[TranslationDataset, Vocab]:
    """Build vocab and test dataset from preprocessed IWSLT data."""
    vocab = build_vocab(data_path)
    test_src, test_tgt = load_split(data_path, "test")
    test_dataset = TranslationDataset(test_src, test_tgt, vocab)
    return test_dataset, vocab


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Evaluate Transformer main training runs"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Directory containing model_d*_*k.pt and metrics_d*_*k.jsonl files",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Directory containing preprocessed IWSLT data",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Discover models
    models = discover_models(args.model_path)
    if not models:
        raise FileNotFoundError(
            f"No model_d*_*k.pt files found in {args.model_path}"
        )
    logger.info(f"Found models: {list(models.keys())}")

    # Load test data
    test_dataset, vocab = _load_test_data(args.data_path)
    logger.info(f"Test set: {len(test_dataset)} examples, vocab: {len(vocab)} tokens")

    config = TDDConfig()

    # Clear existing evaluation file (overwrite mode)
    output_path = Path(args.model_path)
    eval_file = output_path / 'evaluation.jsonl'
    if eval_file.exists():
        eval_file.unlink()

    # Evaluate each model
    for (d_model, train_samples), model_path in models.items():
        samples_k = train_samples // 1000
        logger.info(f"Evaluating d_model={d_model}, {samples_k}K samples...")

        # Load model
        model = TransformerModel(
            vocab_size=len(vocab),
            d_model=d_model,
            n_layers=config.n_layers,
            n_heads=config.n_heads,
            d_ff_multiplier=config.d_ff_multiplier,
            pad_idx=vocab.pad_idx,
        ).to(device)
        model.load_state_dict(
            torch.load(model_path, map_location=device, weights_only=True)
        )
        model.eval()

        # Get metrics file path
        metrics_path = output_path / f"metrics_d{d_model}_{samples_k}k.jsonl"

        # Compute and save metrics
        compute_final_metrics(
            model, test_dataset, vocab, metrics_path, output_path,
            d_model, train_samples, device
        )

        # Clean up
        del model
        torch.cuda.empty_cache()

    logger.info(f"Evaluation complete. Results written to {eval_file}")


if __name__ == "__main__":
    main()
