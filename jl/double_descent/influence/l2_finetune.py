"""Feature extraction and L-BFGS fine-tuning of the final linear layer.

Fine-tunes only the final nn.Linear with L2 regularization to reach a
stationary point, enabling the Yeh & Kim et al. (2018) decomposition.
"""

import logging
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def extract_features(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract penultimate features by hooking model.linear.

    Args:
        model: PreActResNet with a .linear final layer.
        loader: DataLoader with shuffle=False, test-time transforms.
        device: CUDA or CPU device.

    Returns:
        features: (N, feature_dim) float tensor on device.
        labels: (N,) long tensor on device.
    """
    model.eval()
    features_list = []
    labels_list = []

    def hook_fn(module, inp, out):
        features_list.append(inp[0].detach())

    handle = model.linear.register_forward_hook(hook_fn)

    with torch.no_grad():
        for images, batch_labels in loader:
            images = images.to(device)
            model(images)
            labels_list.append(batch_labels.to(device))

    handle.remove()

    features = torch.cat(features_list, dim=0)
    labels = torch.cat(labels_list, dim=0)
    return features, labels


def l2_finetune(
    model: nn.Module,
    features: torch.Tensor,
    labels: torch.Tensor,
    lambda_l2: float = 1e-4,
    max_iter: int = 200,
    tolerance_grad: float = 1e-7,
) -> float:
    """Fine-tune model.linear via L-BFGS on pre-extracted features.

    Optimizes: (1/n) CE(Phi @ W.T + b, labels) + lambda * (||W||^2 + ||b||^2)

    Mutates model.linear in place.

    Args:
        model: Model whose .linear layer will be updated.
        features: (N, d) pre-extracted penultimate features.
        labels: (N,) integer labels.
        lambda_l2: L2 regularization strength.
        max_iter: Maximum L-BFGS iterations.
        tolerance_grad: Convergence threshold on gradient norm.

    Returns:
        Final gradient norm (float).
    """
    n = features.size(0)
    num_classes = model.linear.out_features

    W = model.linear.weight.detach().clone().requires_grad_(True)
    b = model.linear.bias.detach().clone().requires_grad_(True)

    optimizer = torch.optim.LBFGS(
        [W, b],
        max_iter=max_iter,
        tolerance_grad=tolerance_grad,
        tolerance_change=0,
        line_search_fn='strong_wolfe',
    )

    step_count = [0]

    def closure():
        optimizer.zero_grad()
        logits = features @ W.t() + b
        ce_loss = F.cross_entropy(logits, labels)
        l2_term = lambda_l2 * (W.pow(2).sum() + b.pow(2).sum())
        loss = ce_loss + l2_term
        loss.backward()
        step_count[0] += 1
        return loss

    final_loss = optimizer.step(closure)

    grad_norm = torch.cat([W.grad.flatten(), b.grad.flatten()]).norm().item()
    logger.info(
        f"L-BFGS converged in {step_count[0]} closure evals: "
        f"loss={final_loss.item():.6f}, grad_norm={grad_norm:.2e}"
    )

    with torch.no_grad():
        model.linear.weight.copy_(W)
        model.linear.bias.copy_(b)

    return grad_norm
