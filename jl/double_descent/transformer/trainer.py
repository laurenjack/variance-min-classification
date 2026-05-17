"""Single model training for Transformer Double Descent."""

import json
import logging
import math
import os
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.amp import autocast

from jl.double_descent.transformer.bleu import compute_bleu
from jl.double_descent.transformer.transformer_config import TDDConfig
from jl.double_descent.transformer.transformer_data import (
    GPUBatchedLoader,
    TranslationDataset,
    Vocab,
    load_iwslt14,
    load_iwslt14_variance_split,
    load_m2m100_iwslt14,
    load_m2m100_iwslt14_variance_split,
)
from jl.double_descent.transformer.transformer_model import TransformerModel, count_parameters

logger = logging.getLogger(__name__)


def evaluate(
    model: TransformerModel,
    loader: GPUBatchedLoader,
    vocab: Vocab,
    criterion: nn.Module,
) -> Tuple[float, float]:
    """Evaluate model on a pre-built GPU loader. Returns (loss, accuracy).

    The loader yields padded (src, tgt) batches already on the model's
    device, so we skip the per-batch .to() copy. Evaluation runs in FP32
    (no autocast wrap) to match the resnet18 evaluator's convention.
    """
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_tokens = 0

    with torch.no_grad():
        for src, tgt in loader:
            # Forward pass (tgt[:, :-1] as input, tgt[:, 1:] as target)
            logits = model(src, tgt[:, :-1])
            target = tgt[:, 1:].contiguous()

            loss = criterion(
                logits.view(-1, logits.size(-1)),
                target.view(-1),
            )

            predictions = logits.argmax(dim=-1)
            mask = target != vocab.pad_idx
            n_tok = mask.sum().item()
            correct = ((predictions == target) & mask).sum().item()

            total_loss += loss.item() * n_tok
            total_correct += correct
            total_tokens += n_tok

    model.train()

    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0.0
    accuracy = total_correct / total_tokens if total_tokens > 0 else 0.0

    return avg_loss, accuracy


def log_metrics(
    output_path: str,
    d_model: int,
    output_suffix: str,
    metrics: Dict,
) -> None:
    """Log metrics to JSONL file.

    Args:
        output_path: Directory to save metrics.
        d_model: Model embedding dimension.
        output_suffix: Suffix for output files (e.g., "18k" or "split0").
        metrics: Dictionary of metrics to log.
    """
    metrics_file = Path(output_path) / f"metrics_d{d_model}_{output_suffix}.jsonl"
    with open(metrics_file, "a") as f:
        f.write(json.dumps(metrics) + "\n")


def train_single_model(
    gpu_id: int,
    d_model: int,
    train_samples: int,
    config: TDDConfig,
    output_path: str,
    data_path: str,
    m2m100: bool = False,
    split_id: Optional[int] = None,
) -> None:
    """Train a single Transformer with embedding dimension d_model.

    Args:
        gpu_id: GPU device ID (0-7).
        d_model: Embedding dimension.
        train_samples: Number of training samples (default mode only).
        config: Training configuration.
        output_path: Directory to save metrics.
        data_path: Directory containing preprocessed IWSLT data.
        m2m100: If True, load M2M100-tokenized data.
        split_id: When not None, run in variance mode and use the disjoint
            training chunk `split_id`. The held-out chunk (index num_splits)
            is used as the in-distribution test set.
    """
    device = torch.device(f"cuda:{gpu_id}")

    is_variance = split_id is not None
    if is_variance:
        output_suffix = f"split{split_id}"
        log_name = f"trainer_d{d_model}_split{split_id}"
    else:
        samples_k = train_samples // 1000
        output_suffix = f"{samples_k}k"
        log_name = f"trainer_d{d_model}_{samples_k}k"

    # Setup logging for this process
    process_logger = logging.getLogger(log_name)
    process_logger.setLevel(logging.INFO)

    if is_variance:
        process_logger.info(
            f"Starting variance training for d_model={d_model}, split={split_id} "
            f"on GPU {gpu_id} (num_splits={config.num_splits}, "
            f"samples_per_split={config.samples_per_split})"
        )
        loader_fn = (
            load_m2m100_iwslt14_variance_split if m2m100
            else load_iwslt14_variance_split
        )
        train_dataset, valid_dataset, test_dataset, vocab = loader_fn(
            data_dir=data_path,
            split_id=split_id,
            num_splits=config.num_splits,
            samples_per_split=config.samples_per_split,
            subsample_seed=config.subsample_seed,
        )
    else:
        samples_k = train_samples // 1000
        process_logger.info(f"Starting training for d_model={d_model}, {samples_k}K samples on GPU {gpu_id}")
        if m2m100:
            train_dataset, valid_dataset, test_dataset, vocab = load_m2m100_iwslt14(
                data_path, train_samples, config.subsample_seed
            )
        else:
            train_dataset, valid_dataset, test_dataset, vocab = load_iwslt14(
                data_path, train_samples, config.subsample_seed
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

    # Optimizer with warmup + cosine decay schedule
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(0.9, 0.98),
        eps=1e-9,
    )

    def lr_lambda(step: int) -> float:
        """Linear warmup for warmup_steps, then cosine decay to 0."""
        if step < config.warmup_steps:
            # Linear warmup: 0 -> 1
            return step / config.warmup_steps
        else:
            # Cosine decay from 1 -> 0
            progress = (step - config.warmup_steps) / (config.max_steps - config.warmup_steps)
            return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Loss function with optional label smoothing
    criterion = nn.CrossEntropyLoss(
        ignore_index=vocab.pad_idx,
        label_smoothing=config.label_smoothing or 0.0,
    )

    # GPU-resident data: pre-build all (train/valid/test) batches as padded
    # tensors on this worker's device. Removes per-step CPU collation +
    # .to(device) copy entirely. IWSLT-14 train fits in <50 MB on-device
    # even with padding, so the memory cost is trivial.
    train_loader = GPUBatchedLoader(
        train_dataset, vocab.pad_idx, max_tokens=config.max_tokens,
        device=device, shuffle=True, seed=config.subsample_seed,
    )
    valid_loader = GPUBatchedLoader(
        valid_dataset, vocab.pad_idx, max_tokens=config.max_tokens,
        device=device, shuffle=False, seed=config.subsample_seed,
    )
    test_loader = GPUBatchedLoader(
        test_dataset, vocab.pad_idx, max_tokens=config.max_tokens,
        device=device, shuffle=False, seed=config.subsample_seed,
    )

    use_bf16 = getattr(config, "use_bf16", True)
    if use_bf16:
        process_logger.info(f"[d_model={d_model}, {output_suffix}] BF16 autocast enabled")

    # Clear metrics file
    metrics_file = Path(output_path) / f"metrics_d{d_model}_{output_suffix}.jsonl"
    if metrics_file.exists():
        metrics_file.unlink()

    # Early stopping checkpoint tracking
    es_dir = Path(output_path) / "early_stop"
    es_dir.mkdir(parents=True, exist_ok=True)
    es_model_path = es_dir / f"model_d{d_model}_{output_suffix}.pt"
    best_valid_loss = float("inf")
    best_valid_step = 0

    # Training loop
    model.train()
    step = 0
    epoch = 0
    train_start = time.time()

    while step < config.max_steps:
        train_loader.set_epoch(epoch)
        epoch += 1

        for src, tgt in train_loader:
            # src/tgt are already on `device` (GPUBatchedLoader).
            target = tgt[:, 1:].contiguous()

            optimizer.zero_grad()
            with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
                logits = model(src, tgt[:, :-1])
                loss = criterion(
                    logits.view(-1, logits.size(-1)),
                    target.view(-1),
                )
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            step += 1

            # Log training metrics
            if step % config.log_interval == 0:
                # Compute training accuracy for this batch.
                predictions = logits.argmax(dim=-1)
                mask = target != vocab.pad_idx
                train_acc = ((predictions == target) & mask).sum().item() / mask.sum().item()

                current_lr = scheduler.get_last_lr()[0] if scheduler else config.learning_rate

                metrics = {
                    "step": step,
                    "d_model": d_model,
                    "train_samples": train_samples,
                    "train_loss": loss.item(),
                    "train_acc": train_acc,
                    "lr": current_lr,
                }

                # Evaluate on validation set
                if step % config.eval_interval == 0:
                    valid_loss, valid_acc = evaluate(
                        model, valid_loader, vocab, criterion
                    )
                    metrics["valid_loss"] = valid_loss
                    metrics["valid_acc"] = valid_acc

                    # Save early-stop checkpoint if validation loss improved
                    if valid_loss < best_valid_loss:
                        best_valid_loss = valid_loss
                        best_valid_step = step
                        torch.save(model.state_dict(), es_model_path)

                log_metrics(output_path, d_model, output_suffix, metrics)

                elapsed = time.time() - train_start
                process_logger.info(
                    f"[d_model={d_model}, {output_suffix}] Step {step}/{config.max_steps} | "
                    f"Loss: {loss.item():.4f} | LR: {current_lr:.6f} | "
                    f"Time: {elapsed:.1f}s"
                )

            if step >= config.max_steps:
                break

    # Final evaluation
    process_logger.info(f"[d_model={d_model}, {output_suffix}] Running final evaluation...")

    # Reuse train_loader for eval purposes (shuffle off-by-iteration; we just
    # want a deterministic pass to compute final loss/acc on the train set).
    eval_train_loader = GPUBatchedLoader(
        train_dataset, vocab.pad_idx, max_tokens=config.max_tokens,
        device=device, shuffle=False, seed=config.subsample_seed,
    )
    train_loss, train_acc = evaluate(model, eval_train_loader, vocab, criterion)
    valid_loss, valid_acc = evaluate(model, valid_loader, vocab, criterion)
    test_loss, test_acc = evaluate(model, test_loader, vocab, criterion)
    del eval_train_loader

    process_logger.info(f"[d_model={d_model}, {output_suffix}] Computing BLEU score (test only)...")

    # Compute BLEU on the test set only. train_bleu was previously also
    # computed here but on 32K sentences it was the memory hog that OOMed
    # the post-training phase when many processes shared a GPU under MPS.
    test_bleu = compute_bleu(model, test_dataset, vocab, device, max_len=128)

    # Log final metrics
    final_metrics = {
        "step": step,
        "d_model": d_model,
        "train_samples": train_samples,
        "train_loss": train_loss,
        "train_acc": train_acc,
        "valid_loss": valid_loss,
        "valid_acc": valid_acc,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "test_bleu": test_bleu,
        "lr": scheduler.get_last_lr()[0],
    }
    log_metrics(output_path, d_model, output_suffix, final_metrics)

    # Save final model
    model_path = Path(output_path) / f"model_d{d_model}_{output_suffix}.pt"
    torch.save(model.state_dict(), model_path)
    process_logger.info(f"[d_model={d_model}, {output_suffix}] Model saved to {model_path}")

    # Compute and save final evaluation metrics
    from jl.double_descent.transformer.evaluation import compute_final_metrics
    metrics_path = Path(output_path) / f"metrics_d{d_model}_{output_suffix}.jsonl"
    eval_output = Path(output_path)
    compute_final_metrics(
        model, test_dataset, vocab, metrics_path, eval_output,
        d_model, train_samples, device, val_dataset=valid_dataset,
    )

    # Evaluate early-stop checkpoint
    process_logger.info(
        f"[d_model={d_model}, {output_suffix}] Evaluating early-stop model "
        f"(best valid_loss={best_valid_loss:.4f} at step {best_valid_step})..."
    )
    es_model = TransformerModel(
        vocab_size=len(vocab),
        d_model=d_model,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        d_ff_multiplier=config.d_ff_multiplier,
        pad_idx=vocab.pad_idx,
    ).to(device)
    es_model.load_state_dict(
        torch.load(es_model_path, map_location=device, weights_only=True)
    )
    es_model.eval()
    compute_final_metrics(
        es_model, test_dataset, vocab, metrics_path, es_dir,
        d_model, train_samples, device, val_dataset=valid_dataset,
    )
    del es_model
    torch.cuda.empty_cache()

    total_time = time.time() - train_start
    process_logger.info(
        f"[d_model={d_model}, {output_suffix}] Training complete! "
        f"Test BLEU: {test_bleu:.2f} | Test Loss: {test_loss:.4f} | "
        f"Total time: {total_time:.1f}s ({total_time/3600:.2f}h)"
    )
