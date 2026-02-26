"""Single model training for double descent experiments."""

import json
import os
import time
from pathlib import Path
from typing import Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from jl.double_descent.convnet_config import DDConfig
from jl.double_descent.convnet_data import load_cifar10_with_noise
from jl.double_descent.resnet18k import make_resnet18k


def evaluate(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
    """Evaluate model on a dataset.

    Args:
        model: The model to evaluate.
        data_loader: Data loader for evaluation.
        device: Device to run on.

    Returns:
        Tuple of (error_rate, average_loss).
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            batch_size = images.shape[0]

            logits = model(images)
            loss = F.cross_entropy(logits, labels, reduction='sum')

            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += batch_size

    avg_loss = total_loss / total_samples
    error_rate = 1.0 - (total_correct / total_samples)

    model.train()
    return error_rate, avg_loss


def train_single_model(
    gpu_id: int,
    k: int,
    config: DDConfig,
    output_path: str,
    data_path: str,
) -> None:
    """Train a single ResNet18 with width k on the specified GPU.

    This function is designed to be called from a multiprocessing worker.

    Args:
        gpu_id: GPU device ID (0-7).
        k: Width parameter for ResNet18.
        config: Training configuration.
        output_path: Directory to save metrics.
        data_path: Path to CIFAR-10 data.
    """
    device = torch.device(f"cuda:{gpu_id}")

    print(f"[GPU {gpu_id}] Training k={k} on {device}")

    # Load data (each process loads independently)
    train_loader, test_loader = load_cifar10_with_noise(
        noise_prob=config.label_noise,
        batch_size=config.batch_size,
        data_augmentation=config.data_augmentation,
        data_dir=data_path,
    )

    # Create model
    model = make_resnet18k(k=k, num_classes=10).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"[GPU {gpu_id}] k={k} has {num_params:,} parameters")

    # Optimizer
    if config.optimizer == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    # Metrics file for this k value
    os.makedirs(output_path, exist_ok=True)
    metrics_path = Path(output_path) / f"metrics_k{k}.jsonl"

    # Clear metrics file
    with open(metrics_path, 'w') as f:
        pass

    # Training loop
    model.train()
    for epoch in range(config.epochs):
        epoch_start = time.time()

        # Accumulate training metrics during training
        train_loss_sum = 0.0
        train_correct = 0
        train_samples = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            batch_size = images.shape[0]

            optimizer.zero_grad()
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()

            # Accumulate metrics (no extra forward pass needed)
            with torch.no_grad():
                train_loss_sum += loss.item() * batch_size
                preds = logits.argmax(dim=1)
                train_correct += (preds == labels).sum().item()
                train_samples += batch_size

        # Compute training metrics from accumulated values
        train_loss = train_loss_sum / train_samples
        train_error = 1.0 - (train_correct / train_samples)

        # Only evaluate on test set (separate pass required)
        test_error, test_loss = evaluate(model, test_loader, device)

        # Log metrics
        epoch_time = time.time() - epoch_start
        metrics = {
            "epoch": epoch + 1,
            "k": k,
            "train_error": train_error,
            "test_error": test_error,
            "train_loss": train_loss,
            "test_loss": test_loss,
        }
        with open(metrics_path, 'a') as f:
            f.write(json.dumps(metrics) + "\n")

        # Print progress
        if (epoch + 1) % config.log_interval == 0 or epoch == 0:
            print(
                f"[GPU {gpu_id}] k={k} Epoch {epoch + 1:4d}/{config.epochs} | "
                f"train_err={train_error:.4f} test_err={test_error:.4f} | "
                f"train_loss={train_loss:.4f} test_loss={test_loss:.4f} | "
                f"{epoch_time:.1f}s"
            )

    print(f"[GPU {gpu_id}] k={k} training complete! Metrics saved to {metrics_path}")
