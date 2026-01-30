"""
Optimizer module for variance-min-classification.

Provides:
1. SignSGD - Signed gradient descent optimizer
2. create_optimizer - Factory function for creating optimizers from Config
"""

import torch
from torch.optim import Optimizer
from typing import Iterable, Union

from jl.config import Config


class SignSGD(Optimizer):
    """
    Signed Stochastic Gradient Descent optimizer.

    Updates parameters using only the sign of the gradient:
        w = w - lr * sign(grad)

    This makes all parameter updates have the same magnitude (lr),
    regardless of gradient magnitude.

    Args:
        params: Iterable of parameters to optimize
        lr: Learning rate (default: 0.01)
        weight_decay: Weight decay coefficient for L2 regularization (default: 0.0)
    """

    def __init__(self, params, lr: float = 0.01, weight_decay: float = 0.0):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """
        Performs a single optimization step.

        Args:
            closure: A closure that reevaluates the model and returns the loss (optional)
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                # Apply weight decay (L2 regularization) to gradient
                if weight_decay != 0:
                    grad = grad.add(p, alpha=weight_decay)

                # Update using sign of gradient
                p.add_(grad.sign(), alpha=-lr)

        return loss


def create_optimizer(
    params: Union[Iterable, dict],
    c: Config,
    lr: float,
) -> Optimizer:
    """
    Factory function to create an optimizer based on Config.

    Args:
        params: Model parameters to optimize. Can be an iterable (e.g., model.parameters())
                or a dict (e.g., params.values() from stacked parameters).
        c: Config object containing optimizer settings
        lr: Learning rate (may be adjusted for warmup)

    Returns:
        Configured optimizer instance

    Raises:
        ValueError: If optimizer type is unknown
    """
    # Handle dict input (from multi_runner's stacked parameters)
    if isinstance(params, dict):
        params = params.values()

    if c.optimizer == "adam_w":
        return torch.optim.AdamW(
            params,
            lr=lr,
            weight_decay=c.weight_decay,
            eps=c.adam_eps,
            betas=c.adam_betas,
        )
    elif c.optimizer == "sgd":
        return torch.optim.SGD(
            params,
            lr=lr,
            momentum=c.sgd_momentum,
            weight_decay=c.weight_decay,
        )
    elif c.optimizer == "sign_sgd":
        return SignSGD(
            params,
            lr=lr,
            weight_decay=c.weight_decay,
        )
    else:
        raise ValueError(f"Unknown optimizer: {c.optimizer}")
