"""Evaluate variance across training splits for Transformer models.

Loads all variance-mode models (model_d*_split*.pt), runs them on the test set,
and computes:
  - Mean test loss across splits
  - Jensen Gap: E[log(q_bar[y] / q_j[y])] - the variance term in bias-variance decomposition

Output: evaluation.jsonl written alongside the model files.

Usage:
    python -m jl.double_descent.transformer.evaluate \
        --model-path ./output/transformer_variance/03-01-1010 \
        --data-path ./data/iwslt14.tokenized.de-en
"""

import argparse
import json
import logging
import re
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

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


def discover_models(model_dir: str) -> Dict[int, List[Path]]:
    """Discover variance model files grouped by d_model.

    Returns:
        Dict mapping d_model -> sorted list of model file Paths.
    """
    path = Path(model_dir)
    model_files = sorted(path.glob("model_d*_split*.pt"))

    grouped: Dict[int, List[Path]] = defaultdict(list)
    for f in model_files:
        match = re.match(r"model_d(\d+)_split(\d+)\.pt", f.name)
        if match:
            d_model = int(match.group(1))
            grouped[d_model].append(f)

    return dict(sorted(grouped.items()))


def load_model(
    model_path: Path,
    d_model: int,
    vocab_size: int,
    pad_idx: int,
    config: TDDConfig,
    device: torch.device,
) -> TransformerModel:
    """Instantiate model architecture and load saved weights."""
    model = TransformerModel(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        d_ff_multiplier=config.d_ff_multiplier,
        pad_idx=pad_idx,
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    return model


def evaluate_d_model(
    model_paths: List[Path],
    d_model: int,
    test_dataset: TranslationDataset,
    vocab: Vocab,
    config: TDDConfig,
    device: torch.device,
) -> Dict:
    """Compute mean test loss and Jensen Gap for one d_model.

    For each test batch:
      1. Forward pass all models, collect softmax distributions
      2. Compute q_bar = mean distribution across models
      3. Compute Jensen Gap: log(q_bar[y] / q_j[y]) for each model
    """
    num_models = len(model_paths)
    logger.info(f"d_model={d_model}: loading {num_models} models")

    models = [
        load_model(p, d_model, len(vocab), vocab.pad_idx, config, device)
        for p in model_paths
    ]

    loader = DataLoader(
        test_dataset,
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        collate_fn=partial(collate_fn, pad_idx=vocab.pad_idx),
    )

    criterion = nn.CrossEntropyLoss(
        ignore_index=vocab.pad_idx,
        label_smoothing=config.label_smoothing or 0.0,
    )

    total_loss_per_model = [0.0] * num_models
    total_jensen_gap = 0.0
    total_tokens = 0

    with torch.no_grad():
        for src, tgt in loader:
            src = src.to(device)
            tgt = tgt.to(device)

            tgt_input = tgt[:, :-1]
            target = tgt[:, 1:].contiguous()
            mask = target != vocab.pad_idx  # [B, T]
            num_tokens = mask.sum().item()

            all_probs = []
            for j, model in enumerate(models):
                logits = model(src, tgt_input)  # [B, T, V]

                loss = criterion(
                    logits.view(-1, logits.size(-1)),
                    target.view(-1),
                )
                total_loss_per_model[j] += loss.item() * num_tokens

                probs = F.softmax(logits, dim=-1)  # [B, T, V]
                all_probs.append(probs)

            # Compute Jensen Gap: log(q_bar[y] / q_j[y])
            all_probs_t = torch.stack(all_probs, dim=0)  # [M, B, T, V]
            q_bar = all_probs_t.mean(dim=0)  # [B, T, V]
            log_q_bar = torch.log(q_bar + 1e-10)

            target_idx = target.unsqueeze(-1)  # [B, T, 1]
            log_q_bar_y = log_q_bar.gather(dim=-1, index=target_idx).squeeze(-1)  # [B, T]

            batch_jensen = 0.0
            for j in range(num_models):
                log_q_j = torch.log(all_probs_t[j] + 1e-10)
                log_q_j_y = log_q_j.gather(dim=-1, index=target_idx).squeeze(-1)  # [B, T]
                jensen_per_token = log_q_bar_y - log_q_j_y  # [B, T]
                batch_jensen += (jensen_per_token * mask).sum().item()

            total_jensen_gap += batch_jensen / num_models
            total_tokens += num_tokens

    del models
    torch.cuda.empty_cache()

    mean_test_loss = (
        sum(total_loss_per_model) / (num_models * total_tokens)
        if total_tokens > 0
        else 0.0
    )
    mean_jensen_gap = total_jensen_gap / total_tokens if total_tokens > 0 else 0.0

    logger.info(
        f"d_model={d_model}: mean_test_loss={mean_test_loss:.4f}, "
        f"mean_jensen_gap={mean_jensen_gap:.6f}"
    )

    return {
        "d_model": d_model,
        "mean_test_loss": round(mean_test_loss, 6),
        "mean_jensen_gap": round(mean_jensen_gap, 6),
        "num_models": num_models,
        "total_tokens": total_tokens,
    }


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Evaluate variance across training splits for Transformer models"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Directory containing model_d*_split*.pt files",
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

    grouped = discover_models(args.model_path)
    if not grouped:
        raise FileNotFoundError(
            f"No model_d*_split*.pt files found in {args.model_path}"
        )
    logger.info(
        f"Found models for d_model values: {list(grouped.keys())} "
        f"({sum(len(v) for v in grouped.values())} total)"
    )

    config = TDDConfig()

    # Build vocab and load test set (shared across all d_model values)
    vocab = build_vocab(args.data_path)
    test_src, test_tgt = load_split(args.data_path, "test")
    test_dataset = TranslationDataset(test_src, test_tgt, vocab)
    logger.info(f"Test set: {len(test_dataset)} examples, vocab: {len(vocab)} tokens")

    output_path = Path(args.model_path) / "evaluation.jsonl"
    if output_path.exists():
        output_path.unlink()

    for d_model, model_paths in grouped.items():
        result = evaluate_d_model(
            model_paths, d_model, test_dataset, vocab, config, device
        )
        with open(output_path, "a") as f:
            f.write(json.dumps(result) + "\n")

    logger.info(f"Results written to {output_path}")


if __name__ == "__main__":
    main()
