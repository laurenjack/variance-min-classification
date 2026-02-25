"""Parallel training with vmap for double descent experiments."""

import json
import os
import time
from pathlib import Path
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.func import vmap, stack_module_state, functional_call, grad_and_value
from torch.utils.data import DataLoader

from jl.double_descent.convnet_config import DDConfig
from jl.double_descent.resnet18k import make_resnet18k, create_width_masks


def _build_models(
    k_max: int,
    num_widths: int,
    num_classes: int,
    device: torch.device,
) -> List[nn.Module]:
    """Create num_widths copies of ResNet18 with width k_max."""
    models = []
    for _ in range(num_widths):
        model = make_resnet18k(k=k_max, num_classes=num_classes).to(device)
        models.append(model)
    return models


def _create_all_width_masks(
    width_range: List[int],
    k_max: int,
    device: torch.device,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Create stacked width masks for all width configurations.

    Returns:
        Tuple of 4 tensors, each [num_widths, channels]:
        (mask1, mask2, mask4, mask8) for the 4 ResNet stages.
    """
    num_widths = len(width_range)

    mask1 = torch.zeros(num_widths, k_max, device=device)
    mask2 = torch.zeros(num_widths, 2 * k_max, device=device)
    mask4 = torch.zeros(num_widths, 4 * k_max, device=device)
    mask8 = torch.zeros(num_widths, 8 * k_max, device=device)

    for i, k in enumerate(width_range):
        mask1[i, :k] = 1.0
        mask2[i, :2 * k] = 1.0
        mask4[i, :4 * k] = 1.0
        mask8[i, :8 * k] = 1.0

    return mask1, mask2, mask4, mask8


def train(
    config: DDConfig,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    output_path: str,
) -> None:
    """Train ResNet18 models in parallel across different widths.

    Args:
        config: Training configuration.
        train_loader: Training data loader (CIFAR-10 with label noise).
        test_loader: Test data loader (clean CIFAR-10).
        device: Device to train on.
        output_path: Directory to save metrics.
    """
    width_range = list(range(config.width_min, config.width_max + 1))
    num_widths = len(width_range)
    k_max = config.width_max
    num_classes = 10

    print(f"Training {num_widths} models with k={config.width_min}..{config.width_max}")
    print(f"Epochs: {config.epochs}, Batch size: {config.batch_size}, LR: {config.learning_rate}")

    # Build models and stack parameters
    models = _build_models(k_max, num_widths, num_classes, device)
    template = models[0]
    params, buffers = stack_module_state(models)
    params = {name: nn.Parameter(p) for name, p in params.items()}

    # Create width masks for all configurations
    mask1, mask2, mask4, mask8 = _create_all_width_masks(width_range, k_max, device)

    # Define loss function for a single model
    def single_loss(params, buffers, x, y, m1, m2, m4, m8):
        width_masks = [m1, m2, m4, m8]
        logits = functional_call(template, (params, buffers), (x, width_masks))
        loss = F.cross_entropy(logits, y)
        return loss

    # Vectorize across width dimension
    loss_and_grad = grad_and_value(single_loss)
    vectorized_loss_grad = vmap(
        loss_and_grad,
        in_dims=(0, 0, None, None, 0, 0, 0, 0),
        randomness='different',
    )

    # Vectorized forward for evaluation
    def single_forward(params, buffers, x, m1, m2, m4, m8):
        width_masks = [m1, m2, m4, m8]
        return functional_call(template, (params, buffers), (x, width_masks))

    vectorized_forward = vmap(
        single_forward,
        in_dims=(0, 0, None, 0, 0, 0, 0),
        randomness='different',
    )

    # Optimizer
    optimizer = torch.optim.Adam(params.values(), lr=config.learning_rate)

    # Metrics file
    metrics_path = Path(output_path) / "metrics.jsonl"
    os.makedirs(output_path, exist_ok=True)

    print(f"Saving metrics to {metrics_path}")

    with open(metrics_path, 'w') as f:
        pass  # Create empty file

    # Training loop
    template.train()
    for epoch in range(config.epochs):
        epoch_start = time.time()
        epoch_loss = torch.zeros(num_widths, device=device)
        epoch_correct = torch.zeros(num_widths, device=device)
        epoch_samples = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            batch_size = images.shape[0]

            # Forward and backward for all widths
            grads, losses = vectorized_loss_grad(
                params, buffers, images, labels,
                mask1, mask2, mask4, mask8,
            )

            # Update parameters
            for name, param in params.items():
                param.grad = grads[name]
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            # Accumulate metrics
            epoch_loss += losses
            epoch_samples += batch_size

            # Compute training accuracy for this batch
            with torch.no_grad():
                logits = vectorized_forward(params, buffers, images, mask1, mask2, mask4, mask8)
                preds = logits.argmax(dim=-1)  # [num_widths, batch_size]
                correct = (preds == labels.unsqueeze(0)).float().sum(dim=1)
                epoch_correct += correct

        # Compute epoch train metrics
        num_batches = batch_idx + 1
        train_loss = (epoch_loss / num_batches).cpu()
        train_acc = (epoch_correct / epoch_samples).cpu()
        train_error = 1.0 - train_acc

        # Evaluate on test set
        template.eval()
        test_loss = torch.zeros(num_widths, device=device)
        test_correct = torch.zeros(num_widths, device=device)
        test_samples = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                batch_size = images.shape[0]

                logits = vectorized_forward(params, buffers, images, mask1, mask2, mask4, mask8)

                # Compute loss for each width
                for w_idx in range(num_widths):
                    loss = F.cross_entropy(logits[w_idx], labels)
                    test_loss[w_idx] += loss * batch_size

                # Compute accuracy
                preds = logits.argmax(dim=-1)  # [num_widths, batch_size]
                correct = (preds == labels.unsqueeze(0)).float().sum(dim=1)
                test_correct += correct
                test_samples += batch_size

        test_loss = (test_loss / test_samples).cpu()
        test_acc = (test_correct / test_samples).cpu()
        test_error = 1.0 - test_acc

        template.train()

        # Log metrics
        epoch_time = time.time() - epoch_start
        with open(metrics_path, 'a') as f:
            for w_idx, k in enumerate(width_range):
                metrics = {
                    "epoch": epoch + 1,
                    "k": k,
                    "train_error": train_error[w_idx].item(),
                    "test_error": test_error[w_idx].item(),
                    "train_loss": train_loss[w_idx].item(),
                    "test_loss": test_loss[w_idx].item(),
                }
                f.write(json.dumps(metrics) + "\n")

        # Print progress
        if (epoch + 1) % config.log_interval == 0 or epoch == 0:
            # Print summary for a few key widths
            if num_widths <= 4:
                for w_idx, k in enumerate(width_range):
                    print(
                        f"Epoch {epoch + 1:4d}/{config.epochs} | k={k:2d} | "
                        f"train_err={train_error[w_idx]:.4f} test_err={test_error[w_idx]:.4f} | "
                        f"train_loss={train_loss[w_idx]:.4f} test_loss={test_loss[w_idx]:.4f} | "
                        f"{epoch_time:.1f}s"
                    )
            else:
                # Print summary for min, interpolation threshold, and max
                k_min_idx = 0
                k_max_idx = num_widths - 1
                k_mid_idx = num_widths // 2
                print(
                    f"Epoch {epoch + 1:4d}/{config.epochs} | "
                    f"k={width_range[k_min_idx]}: err={test_error[k_min_idx]:.3f} | "
                    f"k={width_range[k_mid_idx]}: err={test_error[k_mid_idx]:.3f} | "
                    f"k={width_range[k_max_idx]}: err={test_error[k_max_idx]:.3f} | "
                    f"{epoch_time:.1f}s"
                )

    print(f"\nTraining complete! Metrics saved to {metrics_path}")
