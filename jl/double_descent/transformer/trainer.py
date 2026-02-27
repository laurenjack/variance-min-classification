"""Single model training for Transformer Double Descent."""

import json
import logging
import os
import time
from functools import partial
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from jl.double_descent.transformer.bleu import compute_bleu
from jl.double_descent.transformer.transformer_config import TDDConfig
from jl.double_descent.transformer.transformer_data import (
    MaxTokensBatchSampler,
    TranslationDataset,
    Vocab,
    collate_fn,
    load_iwslt14,
)
from jl.double_descent.transformer.transformer_model import TransformerModel, count_parameters

logger = logging.getLogger(__name__)


def evaluate(
    model: TransformerModel,
    dataset: TranslationDataset,
    vocab: Vocab,
    device: torch.device,
    criterion: nn.Module,
    batch_size: int = 64,
) -> Tuple[float, float]:
    """Evaluate model on a dataset.

    Args:
        model: Transformer model.
        dataset: Dataset to evaluate on.
        vocab: Vocabulary.
        device: Device to run on.
        criterion: Loss function.
        batch_size: Batch size for evaluation.

    Returns:
        (loss, accuracy) tuple.
    """
    model.eval()

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=partial(collate_fn, pad_idx=vocab.pad_idx),
    )

    total_loss = 0.0
    total_correct = 0
    total_tokens = 0

    with torch.no_grad():
        for src, tgt in loader:
            src = src.to(device)
            tgt = tgt.to(device)

            # Forward pass (tgt[:, :-1] as input, tgt[:, 1:] as target)
            logits = model(src, tgt[:, :-1])
            target = tgt[:, 1:].contiguous()

            # Compute loss
            loss = criterion(
                logits.view(-1, logits.size(-1)),
                target.view(-1),
            )

            # Compute accuracy (excluding padding)
            predictions = logits.argmax(dim=-1)
            mask = target != vocab.pad_idx
            correct = ((predictions == target) & mask).sum().item()

            total_loss += loss.item() * mask.sum().item()
            total_correct += correct
            total_tokens += mask.sum().item()

    model.train()

    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0.0
    accuracy = total_correct / total_tokens if total_tokens > 0 else 0.0

    return avg_loss, accuracy


def log_metrics(
    output_path: str,
    d_model: int,
    metrics: Dict,
) -> None:
    """Log metrics to JSONL file.

    Args:
        output_path: Directory to save metrics.
        d_model: Model embedding dimension.
        metrics: Dictionary of metrics to log.
    """
    metrics_file = Path(output_path) / f"metrics_d{d_model}.jsonl"
    with open(metrics_file, "a") as f:
        f.write(json.dumps(metrics) + "\n")


def train_single_model(
    gpu_id: int,
    d_model: int,
    config: TDDConfig,
    output_path: str,
    data_path: str,
) -> None:
    """Train a single Transformer with embedding dimension d_model.

    Args:
        gpu_id: GPU device ID.
        d_model: Embedding dimension.
        config: Training configuration.
        output_path: Directory to save metrics.
        data_path: Directory containing preprocessed IWSLT data.
    """
    device = torch.device(f"cuda:{gpu_id}")

    # Setup logging for this process
    process_logger = logging.getLogger(f"trainer_d{d_model}")
    process_logger.setLevel(logging.INFO)

    process_logger.info(f"Starting training for d_model={d_model} on GPU {gpu_id}")

    # Load data
    train_dataset, valid_dataset, test_dataset, vocab = load_iwslt14(
        data_path, config.train_samples, config.subsample_seed
    )

    process_logger.info(f"Loaded data: {len(train_dataset)} train, {len(valid_dataset)} valid, {len(test_dataset)} test")
    process_logger.info(f"Vocabulary size: {len(vocab)}")

    # Create model
    model = TransformerModel(
        vocab_size=len(vocab),
        d_model=d_model,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        d_ff_multiplier=config.d_ff_multiplier,
        pad_idx=vocab.pad_idx,
    ).to(device)

    num_params = count_parameters(model)
    process_logger.info(f"Model parameters: {num_params:,}")

    # Optimizer with Vaswani hyperparameters (hardcoded)
    if config.warmup_steps is not None:
        # Vaswani LR schedule: lr scaled by scheduler
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=1.0,  # Scaled by scheduler
            betas=(0.9, 0.98),
            eps=1e-9,
        )

        def lr_lambda(step: int) -> float:
            step = max(step, 1)
            return (d_model ** -0.5) * min(step ** -0.5, step * config.warmup_steps ** -1.5)

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        # Constant learning rate (no warmup)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            betas=(0.9, 0.98),
            eps=1e-9,
        )
        scheduler = None

    # Loss function with optional label smoothing
    criterion = nn.CrossEntropyLoss(
        ignore_index=vocab.pad_idx,
        label_smoothing=config.label_smoothing or 0.0,
    )

    # Create dataloader with max-tokens batching
    train_sampler = MaxTokensBatchSampler(
        train_dataset,
        max_tokens=config.max_tokens,
        shuffle=True,
        seed=config.subsample_seed,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        collate_fn=partial(collate_fn, pad_idx=vocab.pad_idx),
    )

    # Clear metrics file
    metrics_file = Path(output_path) / f"metrics_d{d_model}.jsonl"
    if metrics_file.exists():
        metrics_file.unlink()

    # Training loop
    model.train()
    step = 0
    epoch = 0
    train_start = time.time()

    while step < config.max_steps:
        train_sampler.set_epoch(epoch)
        epoch += 1

        for src, tgt in train_loader:
            src = src.to(device)
            tgt = tgt.to(device)

            # Forward pass
            logits = model(src, tgt[:, :-1])
            target = tgt[:, 1:].contiguous()

            loss = criterion(
                logits.view(-1, logits.size(-1)),
                target.view(-1),
            )

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            step += 1

            # Log training metrics
            if step % config.log_interval == 0:
                # Compute training accuracy for this batch
                predictions = logits.argmax(dim=-1)
                mask = target != vocab.pad_idx
                train_acc = ((predictions == target) & mask).sum().item() / mask.sum().item()

                current_lr = scheduler.get_last_lr()[0] if scheduler else config.learning_rate

                metrics = {
                    "step": step,
                    "d_model": d_model,
                    "train_loss": loss.item(),
                    "train_acc": train_acc,
                    "lr": current_lr,
                }

                # Evaluate on validation set
                if step % config.eval_interval == 0:
                    valid_loss, valid_acc = evaluate(
                        model, valid_dataset, vocab, device, criterion
                    )
                    metrics["valid_loss"] = valid_loss
                    metrics["valid_acc"] = valid_acc

                log_metrics(output_path, d_model, metrics)

                elapsed = time.time() - train_start
                process_logger.info(
                    f"[d_model={d_model}] Step {step}/{config.max_steps} | "
                    f"Loss: {loss.item():.4f} | LR: {current_lr:.6f} | "
                    f"Time: {elapsed:.1f}s"
                )

            if step >= config.max_steps:
                break

    # Final evaluation
    process_logger.info(f"[d_model={d_model}] Running final evaluation...")

    valid_loss, valid_acc = evaluate(model, valid_dataset, vocab, device, criterion)
    test_loss, test_acc = evaluate(model, test_dataset, vocab, device, criterion)

    process_logger.info(f"[d_model={d_model}] Computing BLEU scores...")

    # Compute BLEU (only on subsets for speed)
    train_bleu = compute_bleu(model, train_dataset, vocab, device, max_len=128)
    test_bleu = compute_bleu(model, test_dataset, vocab, device, max_len=128)

    # Log final metrics
    final_metrics = {
        "step": step,
        "d_model": d_model,
        "train_loss": 0.0,  # Not meaningful at end
        "train_acc": 0.0,  # Not meaningful at end
        "valid_loss": valid_loss,
        "valid_acc": valid_acc,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "train_bleu": train_bleu,
        "test_bleu": test_bleu,
        "lr": scheduler.get_last_lr()[0] if scheduler else config.learning_rate,
    }
    log_metrics(output_path, d_model, final_metrics)

    total_time = time.time() - train_start
    process_logger.info(
        f"[d_model={d_model}] Training complete! "
        f"Test BLEU: {test_bleu:.2f} | Test Loss: {test_loss:.4f} | "
        f"Total time: {total_time:.1f}s ({total_time/3600:.2f}h)"
    )
