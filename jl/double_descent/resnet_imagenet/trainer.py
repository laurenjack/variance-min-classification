"""Single-GPU training loop for ResNet-50 on ImageNet-1K.

Per-epoch train accuracy/loss are accumulated during the training pass
(no extra forward). Test accuracy/loss come from a separate pass over
the 50K HuggingFace validation split. Metrics are appended to
metrics.jsonl for later plotting.
"""

import json
import logging
import time
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    use_amp: bool,
) -> Tuple[float, float]:
    """Return (error_rate, average_loss) over data_loader."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.amp.autocast(
                device_type="cuda", dtype=torch.float16, enabled=use_amp
            ):
                logits = model(images)
                loss = F.cross_entropy(logits, labels, reduction="sum")

            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += images.shape[0]

    model.train()
    return 1.0 - (total_correct / total_samples), total_loss / total_samples


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config,
    device: torch.device,
    output_path: str,
) -> None:
    """Run the He et al. training recipe on a single GPU."""
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config.learning_rate,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=config.lr_decay_epochs,
        gamma=config.lr_decay_factor,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=config.use_amp)

    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"
    with open(metrics_path, "w"):
        pass

    model.train()
    for epoch in range(config.epochs):
        epoch_start = time.time()

        train_loss_sum = 0.0
        train_correct = 0
        train_samples = 0
        current_lr = optimizer.param_groups[0]["lr"]

        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            batch_size = images.shape[0]

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                device_type="cuda", dtype=torch.float16, enabled=config.use_amp
            ):
                logits = model(images)
                loss = F.cross_entropy(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            with torch.no_grad():
                train_loss_sum += loss.item() * batch_size
                preds = logits.argmax(dim=1)
                train_correct += (preds == labels).sum().item()
                train_samples += batch_size

        train_loss = train_loss_sum / train_samples
        train_error = 1.0 - (train_correct / train_samples)

        test_error, test_loss = evaluate(
            model, val_loader, device, config.use_amp
        )

        scheduler.step()

        epoch_time = time.time() - epoch_start
        metrics = {
            "epoch": epoch + 1,
            "train_error": train_error,
            "test_error": test_error,
            "train_loss": train_loss,
            "test_loss": test_loss,
            "lr": current_lr,
            "epoch_time_s": epoch_time,
        }
        with open(metrics_path, "a") as f:
            f.write(json.dumps(metrics) + "\n")

        if (epoch + 1) % config.log_interval == 0 or epoch == 0:
            logger.info(
                f"Epoch {epoch + 1:3d}/{config.epochs} | "
                f"train_err={train_error:.4f} test_err={test_error:.4f} | "
                f"train_loss={train_loss:.4f} test_loss={test_loss:.4f} | "
                f"lr={current_lr:.4f} | {epoch_time:.1f}s"
            )

    model_path = output_dir / "model.pt"
    torch.save(model.state_dict(), model_path)
    logger.info(f"Training complete. Model saved to {model_path}")
