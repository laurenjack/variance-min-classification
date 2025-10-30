from abc import ABC, abstractmethod
from typing import Tuple, List, Optional, Union

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
        clean_mode: bool = False,
        shuffle: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate a dataset of size n.

        Args:
            n: Number of samples to generate.
            clean_mode: If True, generates a clean version of the problem for analysis
                       If False, generates from the true distribution, inlucding all noise.
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
      `generate_dataset` method exposes a `clean_mode` flag: when False (default),
      label noise is applied; when True, perfect labels are generated.
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
        clean_mode: bool = False,
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
        if not clean_mode:
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
        clean_mode: bool = False,
        shuffle: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate a dataset of size n.

        Args:
            n: Number of samples to generate.
            clean_mode: Ignored for Gaussian (included for consistency with Problem interface).
            shuffle: If True, randomly permute the resulting dataset.

        Returns:
            (x, y, center_indices):
              - x: shape (n, d) float32 tensor of features sampled from N(0, 1)
              - y: shape (n,) int64 tensor of class labels (0 or 1)
              - center_indices: shape (n,) int64 tensor of zeros (all samples from same center)
        """
        if n <= 0:
            raise ValueError("n must be positive")
        
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
        clean_mode: bool = False,
        shuffle: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate a dataset of size n.
        
        Args:
            n: Number of samples to generate.
            clean_mode: If True, generate batches of n samples and only keep those that are
                       closer to their own center than to the opposite center. Continue until
                       at least n clean samples are collected, then randomly sample exactly n.
                       If False (default), use standard Gaussian sampling from N(mu, I).
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
        
        if clean_mode:
            # Generate batches until we have at least n clean samples
            clean_x_list = []
            clean_y_list = []
            clean_center_indices_list = []
            
            while sum(len(x) for x in clean_x_list) < n:
                # Generate a batch of n samples
                batch_x, batch_y, batch_center_indices = self._generate_batch(n)
                
                # Filter for clean samples: those closer to their own center
                # For class 0 (center at mu): keep samples where dist to mu < dist to -mu
                # For class 1 (center at -mu): keep samples where dist to -mu < dist to mu
                
                # Extract only the true_d dimensions for distance computation
                batch_x_true = batch_x[:, :self.true_d]  # (batch_size, true_d)
                
                # Compute distances to both centers
                dist_to_mu = torch.norm(batch_x_true - self.mu.unsqueeze(0), dim=1)  # (batch_size,)
                dist_to_minus_mu = torch.norm(batch_x_true - (-self.mu).unsqueeze(0), dim=1)  # (batch_size,)
                
                # Clean samples are those closer to their own center
                clean_mask = torch.zeros(len(batch_y), dtype=torch.bool, device=self.device)
                clean_mask[batch_center_indices == 0] = (dist_to_mu < dist_to_minus_mu)[batch_center_indices == 0]
                clean_mask[batch_center_indices == 1] = (dist_to_minus_mu < dist_to_mu)[batch_center_indices == 1]
                
                # Keep only clean samples
                if clean_mask.any():
                    clean_x_list.append(batch_x[clean_mask])
                    clean_y_list.append(batch_y[clean_mask])
                    clean_center_indices_list.append(batch_center_indices[clean_mask])
            
            # Concatenate all clean samples
            x = torch.cat(clean_x_list, dim=0)
            y = torch.cat(clean_y_list, dim=0)
            center_indices = torch.cat(clean_center_indices_list, dim=0)
            
            # Randomly sample exactly n clean samples (without replacement)
            n_clean = len(x)
            if n_clean > n:
                indices = torch.randperm(n_clean, device=self.device)[:n]
                x = x[indices]
                y = y[indices]
                center_indices = center_indices[indices]
        else:
            # Standard generation
            x, y, center_indices = self._generate_batch(n)
        
        # Shuffle if requested
        if shuffle:
            perm = torch.randperm(n, device=self.device)
            x = x[perm]
            y = y[perm]
            center_indices = center_indices[perm]
        
        return x, y, center_indices
    
    def _generate_batch(
        self,
        n: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate a single batch of n samples from the two Gaussians.
        
        Args:
            n: Number of samples to generate.
        
        Returns:
            (x, y, center_indices): Tuple of tensors as in generate_dataset.
        """
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
        
        return x, y, center_indices


class TwoDirections(Problem):
    """
    Generates binary classification data from two fixed directions in feature space.
    
    Creates two centers at μ and -μ where μ is sampled from hypercube vertices 
    (each coordinate is randomly -1 or +1).
    
    Class 0 has mean μ, class 1 has mean -μ. Samples are drawn with class balance
    (approximately n/2 from each center). The percent_correct parameter controls
    noise application for (1 - percent_correct) fraction of samples, balanced across classes:
    - noise_type="mislabel": Randomly flips labels (label noise)
    - noise_type="missing_feature": Zeros out the true_d dimensions while keeping correct labels
    
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
        noise_type: str = "mislabel",
        device: torch.device | None = None,
    ) -> None:
        """
        Args:
            true_d: Dimensionality of the true feature space (where centers are defined)
            noisy_d: Number of extra noise-only dimensions to append
            percent_correct: Probability that a sample is "correct". Must be 
                           in [0.5, 1.0]. The meaning depends on noise_type:
                           - "mislabel": Controls label noise (random mislabeling)
                           - "missing_feature": Controls feature corruption (zeroing true_d dims)
            sigma: Optional standard deviation for Gaussian noise added to samples
            random_basis: If True, apply a single random orthonormal rotation
                         across all d = true_d + noisy_d dimensions
            noise_type: Type of noise to apply. Either "mislabel" (flip labels for incorrect samples)
                       or "missing_feature" (zero out true_d dimensions for incorrect samples)
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
        if noise_type not in ["mislabel", "missing_feature"]:
            raise ValueError("noise_type must be either 'mislabel' or 'missing_feature'")
        
        self.true_d: int = int(true_d)
        self.noisy_d: int = int(noisy_d)
        self.percent_correct: float = float(percent_correct)
        self.sigma: Optional[float] = float(sigma) if sigma is not None else None
        self.random_basis: bool = random_basis
        self.noise_type: str = noise_type
        self.device: torch.device = device if device is not None else torch.device("cpu")
        
        # Total dimensionality
        self._d: int = self.true_d + self.noisy_d
        
        # Sample random direction from hypercube vertices: each coordinate is -1 or +1
        mu_direction = torch.randint(0, 2, (self.true_d,), device=self.device, dtype=torch.float32) * 2.0 - 1.0
        
        # Store the direction for class 0 (class 1 will use -mu)
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
        clean_mode: bool = False,
        shuffle: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate a dataset of size n.
        
        Args:
            n: Number of samples to generate.
            clean_mode: If True, generate clean samples (no noise applied).
                       If False (default), apply noise according to noise_type and percent_correct:
                       - "mislabel": Flip labels for (1 - percent_correct) fraction
                       - "missing_feature": Zero out true_d dimensions for (1 - percent_correct) fraction
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
        
        # Apply noise based on noise_type: either label noise or feature corruption
        # Balance the noise application across classes
        if not clean_mode and self.percent_correct < 1.0:
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
            
            # Apply noise to selected samples from each class
            for class_idx in range(self.NUM_CLASS):
                num_to_corrupt = incorrect_per_class[class_idx]
                if num_to_corrupt > 0:
                    # Find all samples from this center (use center_indices, not y, since y might get modified)
                    class_mask = (center_indices == class_idx)
                    class_sample_indices = torch.nonzero(class_mask, as_tuple=False).view(-1)
                    
                    # Randomly select which ones to corrupt (first num_to_corrupt in a random permutation)
                    perm = torch.randperm(class_sample_indices.numel(), device=self.device)
                    indices_to_corrupt = class_sample_indices[perm[:num_to_corrupt]]
                    
                    if self.noise_type == "mislabel":
                        # Flip their labels (0 -> 1, 1 -> 0)
                        y[indices_to_corrupt] = 1 - y[indices_to_corrupt]
                    elif self.noise_type == "missing_feature":
                        # Zero out the true_d dimensions (keep labels correct)
                        x[indices_to_corrupt, :self.true_d] = 0.0

        # Apply random basis transformation if enabled
        if self.basis is not None:
            x = x @ self.basis
        
        # Shuffle if requested
        if shuffle:
            perm = torch.randperm(n, device=self.device)
            x = x[perm]
            y = y[perm]
            center_indices = center_indices[perm]
        
        return x, y, center_indices
