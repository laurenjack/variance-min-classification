from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple, List, Optional, Union

import math
import torch
from scipy.stats import norm


class Problem(ABC):
    """
    Abstract base class for data generation problems.

    Implementations must return features and labels as tensors.
    """

    @property
    @abstractmethod
    def d(self) -> int:
        """
        The total dimensionality of the feature space.
        
        Returns:
            int: Total number of dimensions in generated features
        """
        raise NotImplementedError

    @abstractmethod
    def generate_dataset(
        self,
        n: int,
        percent_correct: Optional[float] = None,
        shuffle: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate a dataset of size n.

        Args:
            n: Number of samples to generate.
            percent_correct: Fraction of samples with correct labels (None or 1.0 = all correct).
            shuffle: If True, randomly permute the resulting dataset.

        Returns:
            (x, y, center_indices):
              - x: shape (n, d) float32 tensor of features
              - y: shape (n,) int64 tensor of class labels
              - center_indices: shape (n,) int64 tensor of center index for each sample
        """
        raise NotImplementedError


class SubDirections(Problem):
    """
    Subsection-based center generator in R^d with center-balanced sampling.

    - true_d is partitioned into S = true_d // sub_d contiguous subsections of length sub_d.
      subsection s covers indices [s*sub_d, (s+1)*sub_d).
    - There are 2^sub_d possible sign patterns per subsection.
    - centers are assigned round-robin to subsections (0,1,...,S-1, 0,1,...),
      choosing a unique sign pattern (no duplicates) within each subsection from
      its 2^sub_d possibilities.
    - Each center is assigned a class label in [0, num_class) such that within
      each subsection, class labels are as balanced as possible across classes
      (order randomized), to avoid coupling a class to a single subsection.
    - To generate a sample: uses center-balanced sampling to ensure equal representation
      across all centers. Each center gets approximately n/centers samples.
      The chosen center's subsection block uses the center's fixed sign pattern. 
      All other subsection blocks use independently sampled random sign patterns, 
      drawn uniformly with replacement from the subsection's patterns EXCLUDING 
      that subsection's center patterns. If `sigma` was provided at construction 
      (and > 0), Gaussian noise N(0, sigma^2 I) is added over R^d where d = true_d + noisy_d. 
      The sample's label is the center's class. Label noise can be controlled via a
      constructor argument `percent_correct`, which can be a scalar in [0,1] or a
      tensor of shape (centers,) with per-center values in [0,1]. The
      `generate_dataset` method exposes a `use_percent_correct` flag to enable/disable
      applying this label noise at generation time.
    - noisy_d additional dimensions are appended with random {-1, +1} values and standard Gaussian noise.
    - random_basis applies a fixed random orthonormal transformation across all d dimensions.

    Constraints:
    - true_d >= sub_d > 0
    - true_d % sub_d == 0
    - ceil(centers / (true_d // sub_d)) < 2^sub_d  (leave at least one non-center pattern)
    - sigma: optional; if provided must be >= 0
    - noisy_d: optional; if provided must be >= 0
    """

    def __init__(
        self,
        true_d: int,
        sub_d: int,
        centers: int,
        num_class: int,
        sigma: Optional[float] = None,
        noisy_d: int = 0,
        random_basis: bool = False,
        percent_correct: Union[float, torch.Tensor] = 1.0,
        device: torch.device | None = None,
        generator: Optional[torch.Generator] = None,
    ) -> None:
        if sub_d <= 0:
            raise ValueError("sub_d must be positive")
        if true_d < sub_d:
            raise ValueError("true_d must be >= sub_d")
        if true_d % sub_d != 0:
            raise ValueError("true_d % sub_d must be zero")
        if centers <= 0:
            raise ValueError("centers must be positive")
        if num_class <= 0:
            raise ValueError("num_class must be positive")
        if sigma is not None and sigma < 0:
            raise ValueError("sigma must be >= 0 if provided")
        if noisy_d < 0:
            raise ValueError("noisy_d must be non-negative")

        self.true_d: int = int(true_d)  # The true dimensionality for subsection logic
        self.sub_d: int = int(sub_d)
        self.centers: int = int(centers)
        self.num_class: int = int(num_class)
        self.sigma: Optional[float] = float(sigma) if sigma is not None else None
        self.noisy_d: int = int(noisy_d)
        self.random_basis: bool = random_basis
        self.device: torch.device = device if device is not None else torch.device("cpu")
        self.generator: Optional[torch.Generator] = generator
        
        # Total dimensionality
        self._d: int = self.true_d + self.noisy_d

        # Subsection structure (based on true_d only)
        self.num_subsections: int = self.true_d // self.sub_d
        self.subsection_ranges: List[tuple[int, int]] = [
            (s * self.sub_d, (s + 1) * self.sub_d) for s in range(self.num_subsections)
        ]

        # Capacity constraint per subsection
        num_patterns = 1 << self.sub_d
        max_centers_per_subsection = (self.centers + self.num_subsections - 1) // self.num_subsections
        if max_centers_per_subsection >= num_patterns:
            raise ValueError(
                "Requested centers leaves no non-center patterns for sampling in some subsection: "
                f"ceil(centers/S)={max_centers_per_subsection} >= 2^sub_d={num_patterns}"
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
        centers_block_signs = torch.zeros(self.centers, self.sub_d, dtype=torch.float32, device=self.device)
        center_subsection = torch.empty(self.centers, dtype=torch.int64, device=self.device)

        # Track how many centers already assigned per subsection to take next unique pattern
        taken_count = [0 for _ in range(self.num_subsections)]

        for i in range(self.centers):
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

        self.center_subsection: torch.Tensor = center_subsection  # (centers,)
        self.centers_block_signs: torch.Tensor = centers_block_signs  # (centers, sub_d)

        # Assign classes balanced within each subsection to avoid coupling a class to a subsection
        center_to_class = torch.empty(self.centers, dtype=torch.int64, device=self.device)
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

        self.center_to_class: torch.Tensor = center_to_class  # (centers,)

        # Initialize per-center percent_correct tensor from constructor argument
        if isinstance(percent_correct, (int, float)):
            p_scalar = float(percent_correct)
            if not (0.0 <= p_scalar <= 1.0):
                raise ValueError("percent_correct scalar must be between 0.0 and 1.0")
            self.percent_correct_per_center: torch.Tensor = torch.full(
                (self.centers,), p_scalar, dtype=torch.float32, device=self.device
            )
        elif isinstance(percent_correct, torch.Tensor):
            if percent_correct.dim() != 1 or percent_correct.shape[0] != self.centers:
                raise ValueError("percent_correct tensor must have shape (centers,)")
            pc = percent_correct.to(device=self.device, dtype=torch.float32)
            if not torch.all((pc >= 0.0) & (pc <= 1.0)):
                raise ValueError("percent_correct tensor values must be in [0, 1]")
            self.percent_correct_per_center = pc
        else:
            raise TypeError("percent_correct must be a float or a torch.Tensor of shape (centers,)")

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
        
        # Apply a single random orthonormal basis change across all dims if requested
        if self.random_basis:
            A = torch.randn(self._d, self._d, generator=self.generator, device=self.device)
            Q, _ = torch.linalg.qr(A)
            # Ensure right-handed coordinate system (det(Q)=+1)
            if torch.det(Q) < 0:
                Q[:, 0] *= -1
            self.basis: Optional[torch.Tensor] = Q
        else:
            self.basis: Optional[torch.Tensor] = None

    @property
    def d(self) -> int:
        """
        The total dimensionality of the feature space.
        
        Returns:
            int: Total number of dimensions in generated features
        """
        return self._d

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
        use_percent_correct: bool = True,
        shuffle: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if n <= 0:
            raise ValueError("n must be positive")

        # Balance samples across centers (not classes)
        base_per_center = n // self.centers
        remainder = n % self.centers
        
        # Each center gets base_per_center samples
        counts_per_center = torch.full((self.centers,), base_per_center, dtype=torch.int64, device=self.device)
        
        # Randomly assign the remainder samples to centers
        center_order = torch.randperm(self.centers, generator=self.generator, device=self.device)
        counts_per_center[center_order[:remainder]] += 1

        # Build the per-sample center index list
        center_indices = torch.repeat_interleave(
            torch.arange(self.centers, device=self.device), counts_per_center
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

        means_true = torch.cat(blocks, dim=1)  # (n, true_d)
        
        # Append random {-1, +1} valued noisy dims
        if self.noisy_d > 0:
            noisy_signs = torch.randint(
                0, 2, (n, self.noisy_d), 
                generator=self.generator, 
                device=self.device,
                dtype=torch.float32
            ) * 2.0 - 1.0  # Convert {0,1} to {-1,+1}
            means = torch.cat([means_true, noisy_signs], dim=1)  # (n, d)
        else:
            means = means_true

        # Add Gaussian noise if configured
        if self.sigma is not None:
            noise = torch.randn(n, self._d, generator=self.generator, device=self.device, dtype=torch.float32) * float(self.sigma)
            x = means + noise
        else:
            x = means.clone()
        
        # Apply random basis transformation if enabled
        if self.basis is not None:
            x = x @ self.basis

        # Labels are the class of the chosen center
        y = self.center_to_class[center_indices].to(torch.int64)

        # Apply label noise based on per-center percent_correct if requested
        if use_percent_correct:
            incorrect_indices = []
            for center_idx in range(self.centers):
                center_percent_correct = float(self.percent_correct_per_center[center_idx].item())
                # Find all samples from this center
                center_sample_mask = (center_indices == center_idx)
                center_sample_indices = torch.nonzero(center_sample_mask, as_tuple=False).view(-1)
                num_center_samples = center_sample_indices.numel()
                if num_center_samples > 0:
                    num_incorrect_for_center = round(num_center_samples * (1.0 - center_percent_correct))
                    if num_incorrect_for_center > 0:
                        perm = torch.randperm(num_center_samples, generator=self.generator, device=self.device)
                        incorrect_indices.extend(center_sample_indices[perm[:num_incorrect_for_center]].tolist())

            if incorrect_indices:
                incorrect_tensor = torch.tensor(incorrect_indices, dtype=torch.int64, device=self.device)
                for idx in incorrect_tensor:
                    original_class = int(y[idx].item())
                    other_classes = [c for c in range(self.num_class) if c != original_class]
                    if other_classes:
                        new_class = other_classes[torch.randint(len(other_classes), (1,), generator=self.generator, device=self.device).item()]
                        y[idx] = new_class

        if shuffle:
            perm = torch.randperm(n, generator=self.generator, device=self.device)
            x = x[perm]
            y = y[perm]
            center_indices = center_indices[perm]

        return x, y, center_indices


class HyperXorNormal(Problem):
    """
    This class creates 2^d corners of a d-dimensional hypercube as mean vectors.
    The label is determined by the parity of the corner's bits.

    The parameter 'percent_correct' (passed to generate_dataset) indicates the 
    probability that a point from one corner (N(mu_i, I)) is classified 
    (by nearest-mean rule) as belonging to that same corner, rather than to 
    one of its d neighbors.

    The parameter 'random_basis' (default False) applies a fixed random orthonormal
    transformation to mix all dimensions (true + noisy),
    so that each observed feature is an equal mixture of the underlying dims.

    Summarily: from each corner, sum of misclassification probabilities to
    its d neighbors = (1 - percent_correct).
    Hence each corner has p = (1 - percent_correct)/d chance to go to a
    specific neighbor, leading to c = sqrt(d) * Phi^{-1}(1 - p).
    """

    def __init__(
        self,
        true_d: int,
        noisy_d: int = 0,
        random_basis: bool = False,
        device: torch.device | None = None,
        generator: Optional[torch.Generator] = None,
    ) -> None:
        """
        Args:
            true_d: dimension of the underlying hypercube
            noisy_d: number of extra noise-only dimensions to append
            random_basis: if True, apply a single random orthonormal rotation
                          across all d = true_d + noisy_d dimensions
            device: torch device to use for tensors
            generator: optional random generator for reproducibility
        """
        if true_d <= 0:
            raise ValueError("true_d must be positive")
        if noisy_d < 0:
            raise ValueError("noisy_d must be non-negative")

        self.true_d: int = int(true_d)
        self.noisy_d: int = int(noisy_d)
        self.random_basis: bool = random_basis
        self.device: torch.device = device if device is not None else torch.device("cpu")
        self.generator: Optional[torch.Generator] = generator

        # Total dimensionality
        self._d: int = self.true_d + self.noisy_d

        # Build the 2^true_d corner means in {+1,-1}^true_d
        self.num_corners: int = 1 << self.true_d
        
        # corners_true: (2^d_true, true_d)
        corners_true = torch.tensor(
            [
                [(1.0 if (i >> bit) & 1 else -1.0) for bit in range(self.true_d)]
                for i in range(self.num_corners)
            ], 
            dtype=torch.float32, 
            device=self.device
        )

        # Compute parity labels (0 or 1)
        bits = (corners_true > 0).int()
        parity = bits.sum(dim=1) % 2
        self.labels: torch.Tensor = parity  # (2^true_d,)

        # Store base corners for later scaling
        self.base_corners_true: torch.Tensor = corners_true

        # Apply a single random orthonormal basis change across all dims if requested
        if self.random_basis:
            A = torch.randn(self.d, self.d, generator=self.generator, device=self.device)
            Q, _ = torch.linalg.qr(A)
            # Ensure right-handed coordinate system (det(Q)=+1)
            if torch.det(Q) < 0:
                Q[:, 0] *= -1
            self.basis: Optional[torch.Tensor] = Q
        else:
            self.basis: Optional[torch.Tensor] = None

    @property
    def d(self) -> int:
        """
        The total dimensionality of the feature space.
        
        Returns:
            int: Total number of dimensions in generated features
        """
        return self._d

    def generate_dataset(
        self,
        n: int,
        percent_correct: Optional[float] = None,
        shuffle: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate a dataset of size n.

        Args:
            n: Number of samples to generate.
            percent_correct: Probability of correct nearest-mean classification (None or 1.0 = all correct).
            shuffle: If True, randomly permute the resulting dataset.

        Returns:
            (x, y, center_indices):
              - x: shape (n, d) float32 tensor of features
              - y: shape (n,) int64 tensor of class labels (0 or 1)
              - center_indices: shape (n,) int64 tensor of corner index for each sample
        """
        if n <= 0:
            raise ValueError("n must be positive")
        
        # Treat None as 1.0 (no label noise)
        if percent_correct is None:
            percent_correct = 1.0
        
        if not (0.0 <= percent_correct <= 1.0):
            raise ValueError("percent_correct must be between 0.0 and 1.0")

        # Validate percent_correct
        misclassification = 1.0 - percent_correct
        if not 0.0 <= misclassification <= 1.0:
            raise ValueError("percent_correct must be between 0 and 1.")

        # Probability of misclassifying to a single neighbor
        p = misclassification / self.true_d

        # Compute c via inverse Gaussian CDF: Φ(c/√d) = 1 - p
        alpha = 1.0 - p
        # Handle edge case when alpha = 1.0 (perfect classification)
        if alpha >= 1.0:
            # Use a very large but finite value instead of infinity
            z = 8.0  # Approximately norm.ppf(0.99999999)
        else:
            z = norm.ppf(alpha)
        c = math.sqrt(self.true_d) * z

        # Scale corners to length c in true_d
        corners_true = self.base_corners_true * (c / math.sqrt(self.true_d))

        # Append zero-valued noisy dims
        if self.noisy_d > 0:
            zeros = torch.zeros(self.num_corners, self.noisy_d, device=self.device)
            corners = torch.cat([corners_true, zeros], dim=1)
        else:
            corners = corners_true

        # 1) Pick random corner indices
        idx = torch.randint(
            low=0, 
            high=self.num_corners, 
            size=(n,), 
            generator=self.generator, 
            device=self.device
        )

        # 2) Gather the chosen means
        chosen_means = corners[idx]

        # 3) Add standard normal noise
        noise = torch.randn(n, self._d, generator=self.generator, device=self.device)
        x = chosen_means + noise

        # 4) Downscale (optional)
        x /= (2 * z)

        # 5) Rotate if basis transformation is enabled
        if self.basis is not None:
            x = x @ self.basis

        # 6) Labels
        y = self.labels[idx].to(torch.int64)

        # 7) Shuffle
        if shuffle:
            perm = torch.randperm(n, generator=self.generator, device=self.device)
            x = x[perm]
            y = y[perm]
            idx = idx[perm]

        return x, y, idx


class Gaussian(Problem):
    """
    Generates random Gaussian data with 2 classes.
    
    Features are sampled from N(0, 1) in d dimensions.
    Class labels are either:
    - Perfectly balanced (half class 0, half class 1) if perfect_class_balance=True
    - Randomly assigned if perfect_class_balance=False
    
    The percent_correct parameter controls label noise by flipping some labels.
    """
    
    NUM_CLASS = 2
    STANDARD_DEV = 1.0
    
    def __init__(
        self,
        d: int,
        perfect_class_balance: bool = True,
        device: torch.device | None = None,
        generator: Optional[torch.Generator] = None,
    ) -> None:
        """
        Args:
            d: Dimensionality of the feature space
            perfect_class_balance: If True, generate exactly n//2 samples per class.
                                 If False, assign classes randomly.
            device: torch device to use for tensors
            generator: optional random generator for reproducibility
        """
        if d <= 0:
            raise ValueError("d must be positive")
            
        self._d: int = int(d)
        self.perfect_class_balance: bool = perfect_class_balance
        self.device: torch.device = device if device is not None else torch.device("cpu")
        self.generator: Optional[torch.Generator] = generator
    
    @property
    def d(self) -> int:
        """
        The total dimensionality of the feature space.
        
        Returns:
            int: Total number of dimensions in generated features
        """
        return self._d
    
    def generate_dataset(
        self,
        n: int,
        percent_correct: Optional[float] = None,
        shuffle: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate a dataset of size n.

        Args:
            n: Number of samples to generate.
            percent_correct: UNUSED - labels are randomly assigned, so there's no 
                           concept of "correct" labels for this problem.
            shuffle: If True, randomly permute the resulting dataset.

        Returns:
            (x, y, center_indices):
              - x: shape (n, d) float32 tensor of features sampled from N(0, 1)
              - y: shape (n,) int64 tensor of class labels (0 or 1)
              - center_indices: shape (n,) int64 tensor of zeros (all samples from same center)
        """
        if n <= 0:
            raise ValueError("n must be positive")
        # Note: percent_correct is ignored for this implementation
        
        # Generate features from standard normal distribution
        x = torch.normal(
            mean=0.0, 
            std=self.STANDARD_DEV, 
            size=(n, self._d),
            generator=self.generator,
            device=self.device,
            dtype=torch.float32
        )
        
        # Generate labels
        if self.perfect_class_balance:
            y = self._gen_class_balanced_labels(n)
        else:
            y = torch.randint(
                0, 
                self.NUM_CLASS, 
                (n,), 
                dtype=torch.int64,
                generator=self.generator,
                device=self.device
            )
        
        # All samples come from the same center (index 0)
        center_indices = torch.zeros(n, dtype=torch.int64, device=self.device)
        
        # Shuffle if requested
        if shuffle:
            perm = torch.randperm(n, generator=self.generator, device=self.device)
            x = x[perm]
            y = y[perm]
            # center_indices doesn't need to be shuffled since all are 0
        
        return x, y, center_indices
    
    def _gen_class_balanced_labels(self, n: int) -> torch.Tensor:
        """Generate perfectly balanced class labels."""
        class_n = n // self.NUM_CLASS
        
        # Create labels for each class
        labels = []
        for class_idx in range(self.NUM_CLASS):
            class_labels = torch.full(
                (class_n,), 
                class_idx, 
                dtype=torch.int64, 
                device=self.device
            )
            labels.append(class_labels)
        
        # Handle remainder if n is not perfectly divisible by NUM_CLASS
        remainder = n % self.NUM_CLASS
        if remainder > 0:
            # Randomly assign the remainder samples to classes
            remaining_classes = torch.randint(
                0, 
                self.NUM_CLASS, 
                (remainder,), 
                generator=self.generator, 
                device=self.device,
                dtype=torch.int64
            )
            labels.append(remaining_classes)
        
        return torch.cat(labels)


class TwoGaussians(Problem):
    """
    Generates binary classification data from two Gaussian distributions.
    
    Creates two centers at μ and -μ where μ is randomly sampled and then scaled
    such that samples from N(μ, I) have 'percent_correct' probability of being
    closer to μ than to -μ (and vice versa for samples from N(-μ, I)).
    
    Class 0 has mean μ, class 1 has mean -μ. Samples are drawn with class balance
    (approximately n/2 from each center), and labels always match the center
    (no label noise - the overlap is controlled by the geometric separation).
    """
    
    NUM_CLASS = 2
    
    def __init__(
        self,
        true_d: int,
        noisy_d: int = 0,
        percent_correct: float = 1.0,
        device: torch.device | None = None,
    ) -> None:
        """
        Args:
            true_d: Dimensionality of the true feature space (where centers are defined)
            noisy_d: Number of extra noise-only dimensions to append
            percent_correct: Probability that a sample is closer to its own center than
                           the other center. Must be in (0.5, 1.0]. Controls geometric
                           separation of the two Gaussians.
            device: torch device to use for tensors
        """
        if true_d <= 0:
            raise ValueError("true_d must be positive")
        if noisy_d < 0:
            raise ValueError("noisy_d must be non-negative")
        if percent_correct == 1.0:
            raise ValueError("percent_correct can't be exactly 1.0 because that would produce infinite mean distances")
        if not (0.5 < percent_correct < 1.0):
            raise ValueError("percent_correct must be in (0.5, 1.0) for meaningful separation")
        
        self.true_d: int = int(true_d)
        self.noisy_d: int = int(noisy_d)
        self.percent_correct: float = float(percent_correct)
        self.device: torch.device = device if device is not None else torch.device("cpu")
        
        # Total dimensionality
        self._d: int = self.true_d + self.noisy_d
        
        # Sample random direction for mu from standard normal
        mu_direction = torch.randn(self.true_d, device=self.device)
        
        # Scale to achieve desired percent_correct
        # For x ~ N(mu, I), P(x^T mu > 0) = Phi(||mu||) = percent_correct
        # So ||mu|| = Phi^(-1)(percent_correct)
        target_norm = norm.ppf(self.percent_correct)
        mu_direction = mu_direction / torch.norm(mu_direction) * target_norm
        
        # Store the scaled mean vector for class 0 (class 1 will use -mu)
        self.mu: torch.Tensor = mu_direction  # (true_d,)
    
    @property
    def d(self) -> int:
        """
        The total dimensionality of the feature space.
        
        Returns:
            int: Total number of dimensions in generated features
        """
        return self._d
    
    def generate_dataset(
        self,
        n: int,
        percent_correct: Optional[float] = None,
        shuffle: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate a dataset of size n.
        
        Args:
            n: Number of samples to generate.
            percent_correct: Must be None. The percent_correct is set in the constructor
                           and controls the geometric separation of centers.
            shuffle: If True, randomly permute the resulting dataset.
        
        Returns:
            (x, y, center_indices):
              - x: shape (n, d) float32 tensor of features
              - y: shape (n,) int64 tensor of class labels (0 or 1)
              - center_indices: shape (n,) int64 tensor of center index for each sample
                               (0 for class 0 center, 1 for class 1 center)
        """
        if n <= 0:
            raise ValueError("n must be positive")
        
        if percent_correct is not None:
            raise ValueError(
                "percent_correct cannot be specified in generate_dataset for TwoGaussians. "
                "It must be set in the constructor as it controls the geometric separation."
            )
        
        # Balance samples across the two centers
        n_per_class = [n // self.NUM_CLASS, n // self.NUM_CLASS]
        remainder = n % self.NUM_CLASS
        if remainder > 0:
            # Randomly assign remainder to one of the classes
            extra_class = torch.randint(0, self.NUM_CLASS, (1,), device=self.device).item()
            n_per_class[extra_class] += remainder
        
        # Generate samples for each class
        x_list = []
        y_list = []
        center_indices_list = []
        
        for class_idx in range(self.NUM_CLASS):
            n_samples = n_per_class[class_idx]
            
            if n_samples == 0:
                continue
            
            # Determine mean for this class
            if class_idx == 0:
                mean_true = self.mu  # shape (true_d,)
            else:
                mean_true = -self.mu  # shape (true_d,)
            
            # Sample from N(mean, I) in true_d dimensions
            noise_true = torch.randn(n_samples, self.true_d, device=self.device, dtype=torch.float32)
            x_true = mean_true.unsqueeze(0) + noise_true  # (n_samples, true_d)
            
            # Add noisy dimensions if specified
            if self.noisy_d > 0:
                noise_extra = torch.randn(n_samples, self.noisy_d, device=self.device, dtype=torch.float32)
                x_class = torch.cat([x_true, noise_extra], dim=1)  # (n_samples, d)
            else:
                x_class = x_true
            
            # Labels and center indices
            y_class = torch.full((n_samples,), class_idx, dtype=torch.int64, device=self.device)
            center_idx_class = torch.full((n_samples,), class_idx, dtype=torch.int64, device=self.device)
            
            x_list.append(x_class)
            y_list.append(y_class)
            center_indices_list.append(center_idx_class)
        
        # Concatenate all samples
        x = torch.cat(x_list, dim=0)
        y = torch.cat(y_list, dim=0)
        center_indices = torch.cat(center_indices_list, dim=0)
        
        # Shuffle if requested
        if shuffle:
            perm = torch.randperm(n, device=self.device)
            x = x[perm]
            y = y[perm]
            center_indices = center_indices[perm]
        
        return x, y, center_indices


class TwoDirections(Problem):
    """
    Generates binary classification data from two fixed directions in feature space.
    
    Creates two centers at μ and -μ where μ is a fixed random unit direction.
    Unlike TwoGaussians, the percent_correct parameter controls label noise 
    (random mislabeling) rather than geometric separation.
    
    Class 0 has mean μ, class 1 has mean -μ. Samples are drawn with class balance
    (approximately n/2 from each center). Label noise is applied by randomly 
    flipping labels for (1 - percent_correct) fraction of samples, balanced across classes.
    
    Optional Gaussian noise (sigma) and random basis transformation can be applied.
    """
    
    NUM_CLASS = 2
    
    def __init__(
        self,
        true_d: int,
        noisy_d: int = 0,
        percent_correct: float = 1.0,
        sigma: Optional[float] = None,
        random_basis: bool = False,
        device: torch.device | None = None,
    ) -> None:
        """
        Args:
            true_d: Dimensionality of the true feature space (where centers are defined)
            noisy_d: Number of extra noise-only dimensions to append
            percent_correct: Probability that a sample retains its correct label. Must be 
                           in [0.5, 1.0]. Controls label noise (not geometric separation).
            sigma: Optional standard deviation for Gaussian noise added to samples
            random_basis: If True, apply a single random orthonormal rotation
                         across all d = true_d + noisy_d dimensions
            device: torch device to use for tensors
        """
        if true_d <= 0:
            raise ValueError("true_d must be positive")
        if noisy_d < 0:
            raise ValueError("noisy_d must be non-negative")
        if not (0.5 <= percent_correct <= 1.0):
            raise ValueError("percent_correct must be in [0.5, 1.0]")
        if sigma is not None and sigma < 0:
            raise ValueError("sigma must be >= 0 if provided")
        
        self.true_d: int = int(true_d)
        self.noisy_d: int = int(noisy_d)
        self.percent_correct: float = float(percent_correct)
        self.sigma: Optional[float] = float(sigma) if sigma is not None else None
        self.random_basis: bool = random_basis
        self.device: torch.device = device if device is not None else torch.device("cpu")
        
        # Total dimensionality
        self._d: int = self.true_d + self.noisy_d
        
        # Sample random unit direction for mu
        mu_direction = torch.randn(self.true_d, device=self.device)
        mu_direction = mu_direction / torch.norm(mu_direction)
        
        # Store the unit direction for class 0 (class 1 will use -mu)
        self.mu: torch.Tensor = mu_direction  # (true_d,)
        
        # Apply a single random orthonormal basis change across all dims if requested
        if self.random_basis:
            A = torch.randn(self._d, self._d, device=self.device)
            Q, _ = torch.linalg.qr(A)
            # Ensure right-handed coordinate system (det(Q)=+1)
            if torch.det(Q) < 0:
                Q[:, 0] *= -1
            self.basis: Optional[torch.Tensor] = Q
        else:
            self.basis: Optional[torch.Tensor] = None
    
    @property
    def d(self) -> int:
        """
        The total dimensionality of the feature space.
        
        Returns:
            int: Total number of dimensions in generated features
        """
        return self._d
    
    def generate_dataset(
        self,
        n: int,
        use_percent_correct: bool = True,
        shuffle: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate a dataset of size n.
        
        Args:
            n: Number of samples to generate.
            use_percent_correct: If True, apply label noise according to percent_correct.
                               If False, generate perfect labels (as if percent_correct=1.0).
            shuffle: If True, randomly permute the resulting dataset.
        
        Returns:
            (x, y, center_indices):
              - x: shape (n, d) float32 tensor of features
              - y: shape (n,) int64 tensor of class labels (0 or 1), with label noise applied if use_percent_correct=True
              - center_indices: shape (n,) int64 tensor of center index for each sample
                               (0 for class 0 center, 1 for class 1 center)
        """
        if n <= 0:
            raise ValueError("n must be positive")
        
        # Balance samples across the two centers
        n_per_class = [n // self.NUM_CLASS, n // self.NUM_CLASS]
        remainder = n % self.NUM_CLASS
        if remainder > 0:
            # Randomly assign remainder to one of the classes
            extra_class = torch.randint(0, self.NUM_CLASS, (1,), device=self.device).item()
            n_per_class[extra_class] += remainder
        
        # Generate samples for each class
        x_list = []
        y_list = []
        center_indices_list = []
        
        for class_idx in range(self.NUM_CLASS):
            n_samples = n_per_class[class_idx]
            
            if n_samples == 0:
                continue
            
            # Determine mean for this class
            if class_idx == 0:
                mean_true = self.mu  # shape (true_d,)
            else:
                mean_true = -self.mu  # shape (true_d,)
            
            # Start with the mean in true_d dimensions
            x_true = mean_true.unsqueeze(0).expand(n_samples, -1)  # (n_samples, true_d)
            
            # Add random {-1, +1} valued noisy dimensions if specified
            if self.noisy_d > 0:
                noisy_signs = torch.randint(
                    0, 2, (n_samples, self.noisy_d), 
                    device=self.device,
                    dtype=torch.float32
                ) * 2.0 - 1.0  # Convert {0,1} to {-1,+1}
                x_class = torch.cat([x_true, noisy_signs], dim=1)  # (n_samples, d)
            else:
                x_class = x_true
            
            # Labels and center indices (true labels, before noise)
            y_class = torch.full((n_samples,), class_idx, dtype=torch.int64, device=self.device)
            center_idx_class = torch.full((n_samples,), class_idx, dtype=torch.int64, device=self.device)
            
            x_list.append(x_class)
            y_list.append(y_class)
            center_indices_list.append(center_idx_class)
        
        # Concatenate all samples
        x = torch.cat(x_list, dim=0)
        y = torch.cat(y_list, dim=0)
        center_indices = torch.cat(center_indices_list, dim=0)
        
        # Add Gaussian noise if configured (across all dimensions)
        if self.sigma is not None:
            noise = torch.randn(n, self._d, device=self.device, dtype=torch.float32) * float(self.sigma)
            x = x + noise
        
        # Apply random basis transformation if enabled
        if self.basis is not None:
            x = x @ self.basis
        
        # Apply label noise: flip labels for (1 - percent_correct) fraction of samples
        # Balance the flipping across classes
        if use_percent_correct and self.percent_correct < 1.0:
            num_incorrect = round(n * (1.0 - self.percent_correct))
            
            # Balance incorrect samples across classes (round-robin)
            base_incorrect_per_class = num_incorrect // self.NUM_CLASS
            remainder_incorrect = num_incorrect % self.NUM_CLASS
            
            incorrect_per_class = [base_incorrect_per_class] * self.NUM_CLASS
            # Distribute remainder to random classes
            if remainder_incorrect > 0:
                extra_classes = torch.randperm(self.NUM_CLASS, device=self.device)[:remainder_incorrect]
                for ec in extra_classes:
                    incorrect_per_class[int(ec.item())] += 1
            
            # Flip labels for selected samples from each class
            for class_idx in range(self.NUM_CLASS):
                num_to_flip = incorrect_per_class[class_idx]
                if num_to_flip > 0:
                    # Find all samples from this center (use center_indices, not y, since y gets modified)
                    class_mask = (center_indices == class_idx)
                    class_sample_indices = torch.nonzero(class_mask, as_tuple=False).view(-1)
                    
                    # Randomly select which ones to flip (first num_to_flip in a random permutation)
                    perm = torch.randperm(class_sample_indices.numel(), device=self.device)
                    indices_to_flip = class_sample_indices[perm[:num_to_flip]]
                    
                    # Flip their labels (0 -> 1, 1 -> 0)
                    y[indices_to_flip] = 1 - y[indices_to_flip]
        
        # Shuffle if requested
        if shuffle:
            perm = torch.randperm(n, device=self.device)
            x = x[perm]
            y = y[perm]
            center_indices = center_indices[perm]
        
        return x, y, center_indices
