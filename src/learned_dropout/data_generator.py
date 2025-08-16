from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple, List, Optional

import math
import torch


class Problem(ABC):
    """
    Abstract base class for data generation problems.

    Implementations must return features and labels as tensors.
    """

    @abstractmethod
    def generate_dataset(
        self,
        n: int,
        percent_correct: float = 1.0,
        shuffle: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate a dataset of size n.

        Args:
            n: Number of samples to generate.
            percent_correct: Fraction of samples with correct labels (1.0 = all correct).
            shuffle: If True, randomly permute the resulting dataset.

        Returns:
            (x, y):
              - x: shape (n, d) float32 tensor of features
              - y: shape (n,) int64 tensor of class labels
        """
        raise NotImplementedError


class SubDirections(Problem):
    """
    Subsection-based center generator in R^d with center-balanced sampling.

    - d is partitioned into S = d // sub_d contiguous subsections of length sub_d.
      subsection s covers indices [s*sub_d, (s+1)*sub_d).
    - There are 2^sub_d possible sign patterns per subsection.
    - perms centers are assigned round-robin to subsections (0,1,...,S-1, 0,1,...),
      choosing a unique sign pattern (no duplicates) within each subsection from
      its 2^sub_d possibilities.
    - Each center is assigned a class label in [0, num_class) such that within
      each subsection, class labels are as balanced as possible across classes
      (order randomized), to avoid coupling a class to a single subsection.
    - To generate a sample: uses center-balanced sampling to ensure equal representation
      across all centers. Each center gets approximately n/perms samples.
      The chosen center's subsection block uses the center's fixed sign pattern. 
      All other subsection blocks use independently sampled random sign patterns, 
      drawn uniformly with replacement from the subsection's patterns EXCLUDING 
      that subsection's center patterns. If `sigma` was provided at construction 
      (and > 0), Gaussian noise N(0, sigma^2 I) is added over R^d. The sample's 
      label is the center's class. Label noise can be introduced via percent_correct,
      with incorrect samples distributed proportionally across centers.

    Constraints:
    - d >= sub_d > 0
    - d % sub_d == 0
    - ceil(perms / (d // sub_d)) < 2^sub_d  (leave at least one non-center pattern)
    - sigma: optional; if provided must be >= 0
    """

    def __init__(
        self,
        d: int,
        sub_d: int,
        perms: int,
        num_class: int,
        sigma: Optional[float] = None,
        device: torch.device | None = None,
        generator: Optional[torch.Generator] = None,
    ) -> None:
        if sub_d <= 0:
            raise ValueError("sub_d must be positive")
        if d < sub_d:
            raise ValueError("d must be >= sub_d")
        if d % sub_d != 0:
            raise ValueError("d % sub_d must be zero")
        if perms <= 0:
            raise ValueError("perms must be positive")
        if num_class <= 0:
            raise ValueError("num_class must be positive")
        if sigma is not None and sigma < 0:
            raise ValueError("sigma must be >= 0 if provided")

        self.d: int = int(d)
        self.sub_d: int = int(sub_d)
        self.perms: int = int(perms)
        self.num_class: int = int(num_class)
        self.sigma: Optional[float] = float(sigma) if sigma is not None else None
        self.device: torch.device = device if device is not None else torch.device("cpu")
        self.generator: Optional[torch.Generator] = generator

        # Subsection structure
        self.num_subsections: int = self.d // self.sub_d
        self.subsection_ranges: List[tuple[int, int]] = [
            (s * self.sub_d, (s + 1) * self.sub_d) for s in range(self.num_subsections)
        ]

        # Capacity constraint per subsection
        num_patterns = 1 << self.sub_d
        max_centers_per_subsection = (self.perms + self.num_subsections - 1) // self.num_subsections
        if max_centers_per_subsection >= num_patterns:
            raise ValueError(
                "Requested perms leaves no non-center patterns for sampling in some subsection: "
                f"ceil(perms/S)={max_centers_per_subsection} >= 2^sub_d={num_patterns}"
            )

        # Precompute all sign patterns of length sub_d as {-1,+1}
        patterns = self._enumerate_sign_patterns(self.sub_d).to(dtype=torch.float32, device=self.device)
        self._all_patterns: torch.Tensor = patterns  # keep for dataset generation

        # For each subsection, create a random order over pattern indices to pick unique centers
        pattern_order_per_subsection: List[torch.Tensor] = []
        for _ in range(self.num_subsections):
            order = torch.randperm(num_patterns, generator=self.generator, device=self.device)
            pattern_order_per_subsection.append(order)

        # Assign centers round-robin to subsections with unique patterns within each subsection
        centers_block_signs = torch.zeros(self.perms, self.sub_d, dtype=torch.float32, device=self.device)
        center_subsection = torch.empty(self.perms, dtype=torch.int64, device=self.device)

        # Track how many centers already assigned per subsection to take next unique pattern
        taken_count = [0 for _ in range(self.num_subsections)]

        for i in range(self.perms):
            s = i % self.num_subsections
            idx_in_s = taken_count[s]
            if idx_in_s >= num_patterns:
                # Should not happen due to earlier validation
                raise ValueError("Ran out of unique patterns in subsection assignment")
            pat_idx = int(pattern_order_per_subsection[s][idx_in_s].item())
            taken_count[s] += 1

            block = patterns[pat_idx]
            centers_block_signs[i] = block
            center_subsection[i] = s

        self.center_subsection: torch.Tensor = center_subsection  # (perms,)
        self.centers_block_signs: torch.Tensor = centers_block_signs  # (perms, sub_d)

        # Assign classes balanced within each subsection to avoid coupling a class to a subsection
        center_to_class = torch.empty(self.perms, dtype=torch.int64, device=self.device)
        # Collect center indices per subsection
        indices_per_subsection: List[torch.Tensor] = [
            torch.nonzero(center_subsection == s, as_tuple=False).view(-1) for s in range(self.num_subsections)
        ]
        for s, idxs in enumerate(indices_per_subsection):
            k = int(idxs.numel())
            if k == 0:
                continue
            base = k // self.num_class
            rem = k % self.num_class
            labels = []
            for c in range(self.num_class):
                labels.extend([c] * base)
            if rem > 0:
                class_order = torch.randperm(self.num_class, generator=self.generator, device=self.device)[:rem]
                labels.extend([int(c.item()) for c in class_order])
            labels_tensor = torch.tensor(labels, dtype=torch.int64, device=self.device)
            perm = torch.randperm(k, generator=self.generator, device=self.device)
            labels_tensor = labels_tensor[perm]
            center_to_class[idxs] = labels_tensor

        self.center_to_class: torch.Tensor = center_to_class  # (perms,)

        # Precompute allowed non-center pattern indices per subsection (for sampling with replacement)
        self._allowed_noncenter_pattern_indices: List[torch.Tensor] = []
        for s in range(self.num_subsections):
            used_count = taken_count[s]
            used_idx = pattern_order_per_subsection[s][:used_count]  # 1D tensor of used pattern indices
            all_idx = torch.arange(num_patterns, device=self.device)
            mask = torch.ones(num_patterns, dtype=torch.bool, device=self.device)
            mask[used_idx] = False
            allowed = all_idx[mask]
            if allowed.numel() == 0:
                raise ValueError(f"No non-center patterns left to sample in subsection {s}")
            self._allowed_noncenter_pattern_indices.append(allowed)

    def _enumerate_sign_patterns(self, k: int) -> torch.Tensor:
        """Return tensor of shape (2^k, k) with all +/-1 sign patterns."""
        total = 1 << k
        ints = torch.arange(total, dtype=torch.int64)
        bits = ((ints.view(-1, 1) >> torch.arange(k, dtype=torch.int64)) & 1).to(torch.int8)
        bits = torch.flip(bits, dims=[1])
        signs = bits.to(torch.float32) * 2.0 - 1.0
        return signs

    def generate_dataset(
        self,
        n: int,
        percent_correct: float = 1.0,
        shuffle: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if n <= 0:
            raise ValueError("n must be positive")
        if not (0.0 <= percent_correct <= 1.0):
            raise ValueError("percent_correct must be between 0.0 and 1.0")

        # Balance samples across centers (not classes)
        base_per_center = n // self.perms
        remainder = n % self.perms
        
        # Each center gets base_per_center samples
        counts_per_center = torch.full((self.perms,), base_per_center, dtype=torch.int64, device=self.device)
        
        # Randomly assign the remainder samples to centers
        center_order = torch.randperm(self.perms, generator=self.generator, device=self.device)
        counts_per_center[center_order[:remainder]] += 1

        # Build the per-sample center index list
        center_indices = torch.repeat_interleave(
            torch.arange(self.perms, device=self.device), counts_per_center
        )
        # Safety: ensure length matches n (could differ due to integer math bugs)
        if int(center_indices.numel()) != n:
            raise RuntimeError("Internal error: class-balanced center assignment did not sum to n")
        center_subsections = self.center_subsection[center_indices]  # (n,)

        # For each subsection, sample random non-center patterns with replacement,
        # then override samples whose chosen center belongs to this subsection with the center's pattern
        blocks: List[torch.Tensor] = []
        for s in range(self.num_subsections):
            allowed = self._allowed_noncenter_pattern_indices[s]  # (m_s,)
            idx_in_allowed = torch.randint(
                low=0,
                high=allowed.numel(),
                size=(n,),
                generator=self.generator,
                device=self.device,
            )
            rand_idx = allowed[idx_in_allowed]
            block_patterns = self._all_patterns[rand_idx].clone()  # (n, sub_d)

            mask = (center_subsections == s)
            if mask.any():
                center_block = self.centers_block_signs[center_indices[mask]]  # (#mask, sub_d)
                block_patterns[mask] = center_block

            blocks.append(block_patterns)

        means = torch.cat(blocks, dim=1)  # (n, d)

        # Add Gaussian noise if configured
        if self.sigma is not None:
            noise = torch.randn(n, self.d, generator=self.generator, device=self.device, dtype=torch.float32) * float(self.sigma)
            x = means + noise
        else:
            x = means.clone()

        # Labels are the class of the chosen center
        y = self.center_to_class[center_indices].to(torch.int64)

        # Apply label noise based on percent_correct
        num_incorrect = round(n * (1.0 - percent_correct))
        if num_incorrect > 0:
            # Distribute incorrect samples across centers, aligned with sample distribution
            # Use the same logic as sample distribution for consistency
            base_incorrect_per_center = num_incorrect // self.perms
            remainder_incorrect = num_incorrect % self.perms
            
            # Each center gets base_incorrect_per_center incorrect samples
            incorrect_per_center = torch.full((self.perms,), base_incorrect_per_center, dtype=torch.int64, device=self.device)
            
            # Centers that got extra samples should also get extra incorrect samples first
            # Use the same random order as sample distribution
            if remainder_incorrect > 0:
                # Reuse the same center_order from sample distribution for consistency
                if remainder > 0:
                    # Use the same centers that got extra samples
                    extra_sample_centers = center_order[:remainder]
                    # Give priority to these centers for extra incorrect samples
                    num_extra_from_sample_centers = min(remainder_incorrect, remainder)
                    incorrect_per_center[extra_sample_centers[:num_extra_from_sample_centers]] += 1
                    remainder_incorrect -= num_extra_from_sample_centers
                
                # If there are still remaining incorrect samples to distribute, distribute to centers that didn't get extra samples
                if remainder_incorrect > 0:
                    if remainder + remainder_incorrect > self.perms:
                        raise ValueError("Should not happen")
                    selected_centers = center_order[remainder: remainder + remainder_incorrect]
                    incorrect_per_center[selected_centers] += 1
            
            # Build list of sample indices to make incorrect
            incorrect_indices = []
            for center_idx in range(self.perms):
                num_to_change = int(incorrect_per_center[center_idx].item())
                # Find all samples from this center
                center_sample_mask = (center_indices == center_idx)
                center_sample_indices = torch.nonzero(center_sample_mask, as_tuple=False).view(-1)
                if num_to_change > center_sample_indices.numel():
                    raise ValueError("Should not happen")
                incorrect_indices.extend(center_sample_indices[:num_to_change].tolist()) 
            
            # Convert to tensor and apply incorrect labels
            if incorrect_indices:
                incorrect_tensor = torch.tensor(incorrect_indices, dtype=torch.int64, device=self.device)
                
                # For each incorrect sample, assign a random different class
                for idx in incorrect_tensor:
                    original_class = int(y[idx].item())
                    # Choose a random class different from the original
                    other_classes = [c for c in range(self.num_class) if c != original_class]
                    if other_classes:
                        new_class = other_classes[torch.randint(len(other_classes), (1,), generator=self.generator, device=self.device).item()]
                        y[idx] = new_class

        if shuffle:
            perm = torch.randperm(n, generator=self.generator, device=self.device)
            x = x[perm]
            y = y[perm]

        return x, y
