import math
from typing import Optional


class LRScheduler:
    """Learning rate scheduler supporting stable+decay and warmup+stable+decay modes"""

    def __init__(self, optimizer, training_steps, lr, mode):
        self.optimizer = optimizer
        self.training_steps = training_steps
        self.lr = lr
        self.mode = mode

        if mode == 'sd':
            # Stable + Decay: 85% stable, 15% decay
            self.stable_steps = round(0.85 * training_steps)
            self.decay_steps = training_steps - self.stable_steps
            self.warmup_steps = 0
        elif mode == 'wsd':
            # Warmup + Stable + Decay: 5% warmup, 80% stable, 15% decay
            self.warmup_steps = round(0.05 * training_steps)
            self.stable_steps = round(0.80 * training_steps)
            self.decay_steps = training_steps - self.warmup_steps - self.stable_steps
        else:
            raise ValueError(f"Unknown scheduler mode: {mode}")

        self.stable_start = self.warmup_steps
        self.decay_start = self.warmup_steps + self.stable_steps
        self.current_step = 0

    def step(self):
        self.current_step += 1

        if self.current_step <= self.warmup_steps:
            # Warmup phase: linear increase from 0 to lr
            progress = (self.current_step - 1) / self.warmup_steps  # -1 because step is called after optimizer.step
            new_lr = self.lr * progress
        elif self.current_step <= self.decay_start:
            # Stable phase: constant lr
            new_lr = self.lr
        else:
            # Decay phase: cosine annealing from lr to 0
            decay_progress = (self.current_step - self.decay_start) / self.decay_steps
            new_lr = self.lr * (0.5 * (1 + math.cos(math.pi * decay_progress)))

        # Update all parameter groups
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr


def create_lr_scheduler(optimizer, training_steps: int, lr: float, lr_scheduler: Optional[str]):
    """Factory method to create learning rate scheduler or return None"""
    if lr_scheduler is None:
        return None

    if lr_scheduler not in ['sd', 'wsd']:
        raise ValueError(f"Unsupported lr_scheduler: {lr_scheduler}")

    return LRScheduler(optimizer, training_steps, lr, lr_scheduler)
