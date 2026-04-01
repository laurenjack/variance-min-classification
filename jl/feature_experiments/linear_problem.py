import math
from typing import Optional, Tuple

import torch

from jl.problem import Problem


class LinearBinary(Problem):
    """
    Binary classification with a single linear feature.

    A direction w* in R^true_d is drawn randomly. Each sample x lies along +w*
    (class 1) or -w* (class 0), with class-balanced generation. The label is
    flipped with probability (1 - percent_correct), so percent_correct=0.8 means
    20% of labels are wrong.

    If noisy_d > 0, each sample gets noisy_d extra dimensions drawn from
    N(0, 1/true_d), giving total dimensionality true_d + noisy_d.

    If random_basis=True, a random orthonormal rotation is applied across all
    d = true_d + noisy_d dimensions, spreading the true feature across all axes.
    """

    def __init__(
        self,
        noisy_d: int = 0,
        percent_correct: float = 0.8,
        random_basis: bool = True,
        device: Optional[torch.device] = None,
        generator: Optional[torch.Generator] = None,
    ) -> None:
        self._true_d = 1
        if noisy_d < 0:
            raise ValueError("noisy_d must be non-negative")
        self.noisy_d = int(noisy_d)
        if not (0.5 <= percent_correct <= 1.0):
            raise ValueError(f"percent_correct must be in [0.5, 1.0], got {percent_correct}")
        self.percent_correct = percent_correct
        self.device = device if device is not None else torch.device("cpu")
        self.generator = generator

        # Feature direction: a single unit vector in R^1 (just [1.0])
        self.w_star = torch.ones(1, device=self.device, dtype=torch.float32)

        # Random basis rotation across all d dimensions
        self.random_basis = random_basis
        if self.random_basis:
            total_d = self._true_d + self.noisy_d
            A = torch.randn(total_d, total_d, generator=self.generator, device=self.device)
            Q, _ = torch.linalg.qr(A)
            self.basis: Optional[torch.Tensor] = Q
        else:
            self.basis = None

    @property
    def d(self) -> int:
        return self._true_d + self.noisy_d

    def num_classes(self) -> int:
        return 2

    def generate_dataset(
        self,
        n: int,
        clean_mode: bool = False,
        shuffle: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if n <= 0:
            raise ValueError("n must be positive")

        # Class-balanced: first half class 0, second half class 1
        half = n // 2
        y = torch.cat([
            torch.zeros(half, device=self.device, dtype=torch.int64),
            torch.ones(n - half, device=self.device, dtype=torch.int64),
        ])

        # x in true feature space: class 0 -> -w*, class 1 -> +w*
        signs = 2.0 * y.float() - 1.0  # -1 for class 0, +1 for class 1
        x = signs.unsqueeze(1) * self.w_star.unsqueeze(0)  # (n, 1)

        # Add noisy dimensions
        if self.noisy_d > 0:
            noise = torch.randn(
                n, self.noisy_d, generator=self.generator, device=self.device
            )
            scale = 1.0 / math.sqrt(self._true_d)
            x = torch.cat([x, noise * scale], dim=1)  # (n, d)

        center_indices = y.clone()

        # Apply label noise
        if not clean_mode and self.percent_correct < 1.0:
            num_to_flip = round(n * (1.0 - self.percent_correct))
            if num_to_flip > 0:
                perm = torch.randperm(n, generator=self.generator, device=self.device)
                flip_indices = perm[:num_to_flip]
                y[flip_indices] = 1 - y[flip_indices]

        # Random basis rotation
        if self.basis is not None:
            x = x @ self.basis

        if shuffle:
            perm = torch.randperm(n, generator=self.generator, device=self.device)
            x = x[perm]
            y = y[perm]
            center_indices = center_indices[perm]

        return x, y, center_indices
