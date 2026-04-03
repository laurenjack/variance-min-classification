"""Downstream fine-tuning for RETFound on APTOS-2019.

Follows the paper recipe: AdamW with layer-wise LR decay, cosine schedule,
warmup, label smoothing. Saves checkpoint with best validation AUROC.
"""

import json
import logging
import math
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from jl.double_descent.medical_calibration.config import MedCalConfig

logger = logging.getLogger(__name__)


def cosine_lr(optimizer, epoch: int, config: MedCalConfig, lr: float):
    """Adjust learning rate with warmup + cosine decay."""
    if epoch < config.warmup_epochs:
        scale = epoch / config.warmup_epochs
    else:
        progress = (epoch - config.warmup_epochs) / (config.epochs - config.warmup_epochs)
        scale = 0.5 * (1.0 + math.cos(math.pi * progress))

    for param_group in optimizer.param_groups:
        base_lr = param_group.get("_base_lr", param_group["lr"])
        if "_base_lr" not in param_group:
            param_group["_base_lr"] = param_group["lr"]
        param_group["lr"] = base_lr * scale


def evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> Dict[str, float]:
    """Compute validation metrics: loss, accuracy, AUROC.

    Returns:
        Dict with val_loss, val_acc, val_auroc.
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            with torch.amp.autocast("cuda"):
                logits = model(images)
            loss = F.cross_entropy(logits.float(), labels, reduction="sum")
            total_loss += loss.item()
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_samples += labels.size(0)

            probs = F.softmax(logits, dim=-1)
            all_probs.append(probs.cpu())
            all_labels.append(labels.cpu())

    val_loss = total_loss / total_samples
    val_acc = total_correct / total_samples

    all_probs = torch.cat(all_probs).numpy()
    all_labels = torch.cat(all_labels).numpy()

    try:
        val_auroc = roc_auc_score(all_labels, all_probs, multi_class="ovr", average="macro")
    except ValueError:
        val_auroc = 0.0

    return {
        "val_loss": val_loss,
        "val_acc": val_acc,
        "val_auroc": val_auroc,
    }


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: MedCalConfig,
    output_path: str,
    device: torch.device,
) -> Path:
    """Fine-tune model on APTOS-2019 and save best checkpoint.

    Args:
        model: Initialized RETFound model.
        train_loader: Training data.
        val_loader: Validation data.
        config: Experiment config.
        output_path: Directory to save checkpoints and metrics.
        device: Training device.

    Returns:
        Path to best checkpoint.
    """
    from jl.double_descent.medical_calibration.model import build_layer_decay_param_groups

    out_dir = Path(output_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Compute effective LR from base LR
    # Paper: effective_lr = blr * batch_size / 256
    lr = config.blr * config.batch_size / 256

    param_groups = build_layer_decay_param_groups(model, config, lr)
    optimizer = torch.optim.AdamW(param_groups, lr=lr, betas=(0.9, 0.999))

    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    scaler = torch.amp.GradScaler("cuda")

    best_auroc = 0.0
    best_path = out_dir / "best_model.pt"
    metrics_path = out_dir / "metrics.jsonl"

    # Clear metrics file
    if metrics_path.exists():
        metrics_path.unlink()

    logger.info(
        f"Starting fine-tuning: {config.epochs} epochs, "
        f"lr={lr:.6f}, layer_decay={config.layer_decay}"
    )

    for epoch in range(config.epochs):
        # Adjust learning rate
        cosine_lr(optimizer, epoch, config, lr)

        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_samples = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda"):
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * labels.size(0)
            train_correct += (logits.argmax(dim=1) == labels).sum().item()
            train_samples += labels.size(0)

        train_loss /= train_samples
        train_acc = train_correct / train_samples

        # Validate
        val_metrics = evaluate(model, val_loader, device)

        current_lr = optimizer.param_groups[-1]["lr"]
        metrics = {
            "epoch": epoch + 1,
            "train_loss": round(train_loss, 6),
            "train_acc": round(train_acc, 6),
            "val_loss": round(val_metrics["val_loss"], 6),
            "val_acc": round(val_metrics["val_acc"], 6),
            "val_auroc": round(val_metrics["val_auroc"], 6),
            "lr": round(current_lr, 8),
        }

        with open(metrics_path, "a") as f:
            f.write(json.dumps(metrics) + "\n")

        # Save best model by AUROC
        if val_metrics["val_auroc"] > best_auroc:
            best_auroc = val_metrics["val_auroc"]
            torch.save(model.state_dict(), best_path)
            logger.info(
                f"Epoch {epoch + 1}/{config.epochs}: "
                f"train_loss={train_loss:.4f}, val_auroc={val_metrics['val_auroc']:.4f} "
                f"(NEW BEST)"
            )
        elif (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info(
                f"Epoch {epoch + 1}/{config.epochs}: "
                f"train_loss={train_loss:.4f}, val_auroc={val_metrics['val_auroc']:.4f}"
            )

    logger.info(f"Training complete. Best val AUROC: {best_auroc:.4f}")
    return best_path
