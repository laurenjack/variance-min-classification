from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import torch

from jl.problem import Problem


def _validate_positive(value: int, name: str) -> int:
    """Ensure value is a positive integer."""
    int_value = int(value)
    if int_value <= 0:
        raise ValueError(f"{name} must be positive")
    return int_value


def _orthonormal_rows(
    num_rows: int,
    dimension: int,
    generator: Optional[torch.Generator],
    device: torch.device,
) -> torch.Tensor:
    """Generate num_rows orthonormal vectors in R^dimension."""
    if num_rows > dimension:
        raise ValueError("num_rows must be <= dimension to form orthonormal rows")

    # Generate a random matrix with shape (dimension, num_rows) and run QR
    A = torch.randn(dimension, num_rows, generator=generator, device=device)
    Q_full, _ = torch.linalg.qr(A, mode="reduced")  # (dimension, num_rows)
    return Q_full.T  # (num_rows, dimension)


def _standard_basis_rows(
    num_rows: int,
    dimension: int,
    generator: Optional[torch.Generator],
    device: torch.device,
) -> torch.Tensor:
    """Select num_rows standard basis vectors uniformly without replacement."""
    if num_rows > dimension:
        raise ValueError("num_rows must be <= dimension to sample standard basis rows")

    indices = torch.randperm(dimension, generator=generator, device=device)[:num_rows]
    basis = torch.eye(dimension, device=device, dtype=torch.float32)
    return basis.index_select(0, indices)


def _construct_random_features(
    num_rows: int,
    dimension: int,
    generator: Optional[torch.Generator],
    device: torch.device,
) -> torch.Tensor:
    """
    Construct random unit-norm features with rejection sampling to avoid
    vectors that are too close (cosine similarity threshold cosine(2π/d)).
    """
    threshold = math.cos(math.pi /(2 * dimension))
    max_attempts_per_feature = 100

    features: List[torch.Tensor] = []
    for i in range(num_rows):
        attempts = 0
        while attempts < max_attempts_per_feature:
            v = torch.randn(dimension, generator=generator, device=device)
            v = v / torch.norm(v)

            if not features:
                features.append(v)
                break

            dots = torch.stack([torch.abs(torch.dot(v, existing)) for existing in features])
            if torch.all(dots <= threshold):
                features.append(v)
                break

            attempts += 1

        if len(features) <= i:
            raise ValueError(
                "Could not generate the requested number of features without violating the "
                f"dot product threshold cos(2π/{dimension}) = {threshold:.4f}. "
                "Consider reducing the number of features or increasing the dimension."
            )

    return torch.stack(features)


class SingleFeatures(Problem):
    """
    Generates classification data where each sample has exactly one active feature.
    
    Each feature is encoded as a specific direction in the input space R^d.
    The f features are the f rows of matrix Q ∈ R^(f×d).
    
    Two modes are supported:
    1. orthogonal_as_possible=True (default):
       - If f <= d: Q has orthonormal rows (orthonormal basis)
       - If f > d: Q rows form a Unit Norm Tight Frame (UNTF), where Q.T @ Q = (f/d) * I_d
    
    2. orthogonal_as_possible=False:
       - Q rows are random unit-norm vectors from standard Gaussian
       - Features are generated with rejection sampling to ensure they are not too close
       - Dot product between any two features must be <= cos(2π/d)
    
    Each sample activates exactly one feature (and corresponds to one class).
    
    The number of classes equals the number of features: num_class = f.
    """
    
    def __init__(
        self,
        d: int,
        f: int,
        orthogonal_as_possible: bool = True,
        n_per_f: Optional[List[int]] = None,
        device: Optional[torch.device] = None,
        generator: Optional[torch.Generator] = None,
    ) -> None:
        """
        Args:
            d: Dimensionality of the input space
            f: Number of features (and number of classes)
            orthogonal_as_possible: If True, generate orthonormal features (f <= d) or UNTF (f > d).
                                    If False, generate random unit-norm features with rejection sampling
                                    to avoid features that are too close (dot product > cos(2π/d)).
            n_per_f: The frequency of each feature in the dataset. If None, all features are equally frequent.
            device: torch device to use for tensors
            generator: optional random generator for reproducibility
        """
        self._d: int = _validate_positive(d, "d")
        self.f: int = _validate_positive(f, "f")
        self.num_class: int = self.f
        self.orthogonal_as_possible: bool = orthogonal_as_possible
        self.device: torch.device = device if device is not None else torch.device("cpu")
        self.generator: Optional[torch.Generator] = generator
        
        # Validate and store n_per_f
        if n_per_f is not None:
            if len(n_per_f) != self.f:
                raise ValueError(f"n_per_f must have length f={self.f}, got length {len(n_per_f)}")
            if any(x <= 0 for x in n_per_f):
                raise ValueError("All elements of n_per_f must be positive")
        self.n_per_f = n_per_f
        
        # Generate Q ∈ R^(f×d)
        if self.orthogonal_as_possible:
            if self.f <= self._d:
                self.Q: torch.Tensor = _orthonormal_rows(
                    self.f,
                    self._d,
                    generator=self.generator,
                    device=self.device,
                )
            else:
                print("WARNING: This is not a harmonic UNTF, so the points are not equally distributed on the sphere.")
                # Case 2: f > d - construct Unit Norm Tight Frame (UNTF)
                # A UNTF satisfies: Q.T @ Q = (f/d) * I_d and each row has unit norm
                self.Q = self._construct_untf()
        else:
            # Case 3: Random unit-norm features with rejection sampling
            self.Q = _construct_random_features(
                self.f,
                self._d,
                generator=self.generator,
                device=self.device,
            )
    
    def _construct_untf(self) -> torch.Tensor:
        """
        Construct a Unit Norm Tight Frame (UNTF) for f > d.
        
        A UNTF satisfies:
        - Each row has unit norm: ||Q[i]|| = 1 for all i
        - Tight frame property: Q.T @ Q = (f/d) * I_d
        
        Returns:
            Q: shape (f, d) tensor with unit norm rows forming a tight frame
        """
        # Start with f random unit vectors
        A = torch.randn(self.f, self._d, generator=self.generator, device=self.device)
        # Normalize each row to unit norm
        Q = A / torch.norm(A, dim=1, keepdim=True)  # (f, d)
        
        # Iteratively refine to satisfy tight frame property Q.T @ Q = (f/d) * I_d
        # Using alternating projection method
        target_frame_constant = self.f / self._d
        
        for _ in range(50):  # Max iterations
            # Compute current Gram matrix
            G = Q.T @ Q  # (d, d)
            
            # Check if we're close enough
            target_G = target_frame_constant * torch.eye(self._d, device=self.device)
            if torch.allclose(G, target_G, atol=1e-6):
                break
            
            # Project to satisfy tight frame property
            # Compute G^(-1/2) using eigendecomposition for numerical stability
            eigenvals, eigenvecs = torch.linalg.eigh(G)
            eigenvals = torch.clamp(eigenvals, min=1e-8)  # Avoid division by zero
            G_inv_sqrt = eigenvecs @ torch.diag(1.0 / torch.sqrt(eigenvals)) @ eigenvecs.T
            
            # Transform: Q_new = sqrt(f/d) * Q @ G^(-1/2)
            Q = math.sqrt(target_frame_constant) * Q @ G_inv_sqrt
            
            # Renormalize rows to unit norm
            row_norms = torch.norm(Q, dim=1, keepdim=True)
            Q = Q / row_norms
        
        return Q
    
    @property
    def d(self) -> int:
        """
        The total dimensionality of the feature space.
        
        Returns:
            int: Total number of dimensions in generated features
        """
        return self._d
    
    def num_classes(self) -> int:
        """
        Returns the number of classes for this classification problem.
        
        Returns:
            int: Number of classes
        """
        return self.num_class
    
    def generate_dataset(
        self,
        n: int,
        clean_mode: bool = False,
        shuffle: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate a dataset of size n.
        
        Each sample has exactly one active feature, corresponding to one class.
        
        If n_per_f is None (default):
            Samples are class-balanced (each class gets approximately n/f samples).
        
        If n_per_f is specified:
            Feature i appears with relative frequency n_per_f[i].
            n must be a multiple of sum(n_per_f), otherwise an error is raised.
        
        Args:
            n: Number of samples to generate.
            clean_mode: Unused for this problem (no noise yet).
            shuffle: If True, randomly permute the resulting dataset.
        
        Returns:
            (x, y, center_indices):
              - x: shape (n, d) float32 tensor of features
              - y: shape (n,) int64 tensor of class labels (in range [0, f))
              - center_indices: shape (n,) int64 tensor, same as y for this problem
        """
        if n <= 0:
            raise ValueError("n must be positive")
        
        if self.n_per_f is not None:
            # Validate that n is a multiple of sum(n_per_f)
            total_freq = sum(self.n_per_f)
            if n % total_freq != 0:
                raise ValueError(
                    f"When n_per_f is specified, n must be a multiple of sum(n_per_f)={total_freq}. "
                    f"Got n={n}."
                )
            
            # Generate samples according to specified frequencies
            multiplier = n // total_freq
            
            # Build x_standard by stacking one-hot vectors according to frequencies
            x_standard_list: List[torch.Tensor] = []
            y_list: List[torch.Tensor] = []
            
            for feature_idx in range(self.f):
                num_samples = self.n_per_f[feature_idx] * multiplier
                # Create one-hot vectors for this feature
                one_hot = torch.zeros(num_samples, self.f, device=self.device, dtype=torch.float32)
                one_hot[:, feature_idx] = 1.0
                x_standard_list.append(one_hot)
                
                # Create labels for this feature
                labels = torch.full((num_samples,), feature_idx, device=self.device, dtype=torch.int64)
                y_list.append(labels)
            
            x_standard = torch.cat(x_standard_list, dim=0)  # shape (n, f)
            y = torch.cat(y_list, dim=0)  # shape (n,)
        else:
            # Default behavior: approximately balanced classes
            # Stack ceiling(n/f) identity matrices vertically
            num_repeats = math.ceil(n / self.f)
            
            # Create I_f identity matrix
            I_f = torch.eye(self.f, device=self.device, dtype=torch.float32)
            
            # Stack num_repeats copies vertically
            x_standard = I_f.repeat(num_repeats, 1)  # shape (f * num_repeats, f)
            
            # Truncate to first n rows
            x_standard = x_standard[:n]  # shape (n, f)
            
            # Labels: y[i] = argmax(x_standard[i]) = i mod f
            # Since x_standard[i] is a one-hot vector, argmax gives us the column index
            y = torch.arange(n, device=self.device, dtype=torch.int64) % self.f
        
        # Compute x = x_standard @ Q
        x = x_standard @ self.Q  # shape (n, d)
        
        # center_indices is the same as y for this problem
        center_indices = y.clone()
        
        # Shuffle if requested
        if shuffle:
            perm = torch.randperm(n, generator=self.generator, device=self.device)
            x = x[perm]
            y = y[perm]
            center_indices = center_indices[perm]
        
        return x, y, center_indices


class Kaleidoscope(Problem):
    """
    Multi-layer additive feature problem described in system_design.md.

    Each layer l has C_l centers, represented by a matrix Q_l ∈ R^{C_l × d}.
    Samples are generated by selecting one center per layer, scaling it by
    1 / 2^{l/2}, and summing across layers. The label corresponds to the
    selected center in the final layer.

    When is_standard_basis=True, the rows of every Q_l are sampled from the
    standard basis (canonical unit vectors) instead of random orthonormal
    directions.
    """

    def __init__(
        self,
        d: int,
        centers: Sequence[int],
        device: torch.device | None = None,
        generator: Optional[torch.Generator] = None,
        is_standard_basis: bool = False,
    ) -> None:
        self._d: int = _validate_positive(d, "d")
        if not centers:
            raise ValueError("centers must be a non-empty sequence")

        self.centers: List[int] = []
        for idx, c in enumerate(centers):
            c = _validate_positive(c, f"C_{idx}")
            if c > self._d:
                raise ValueError(f"C_{idx} must be <= d (got {c} > {self._d})")
            self.centers.append(c)

        self.num_class = self.centers[-1]
        self.device = device if device is not None else torch.device("cpu")
        self.generator = generator
        self.is_standard_basis = is_standard_basis

        self.Q_layers: List[torch.Tensor] = []
        for c in self.centers:
            if self.is_standard_basis:
                Q_l = _standard_basis_rows(c, self._d, self.generator, self.device)
            else:
                Q_l = _orthonormal_rows(c, self._d, self.generator, self.device)
            self.Q_layers.append(Q_l.to(dtype=torch.float32))

    @property
    def d(self) -> int:
        return self._d

    def num_classes(self) -> int:
        return self.num_class

    def generate_dataset(
        self,
        n: int,
        clean_mode: bool = False,
        shuffle: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        del clean_mode  # not used for this problem (no noise yet)

        n = _validate_positive(n, "n")
        x = torch.zeros(n, self._d, device=self.device, dtype=torch.float32)
        center_indices_list: List[torch.Tensor] = []

        for layer_idx, (Q_l, c) in enumerate(zip(self.Q_layers, self.centers)):
            # Generate balanced center assignments for this layer
            # Stack identity-like pattern: [0, 1, 2, ..., c-1, 0, 1, 2, ..., c-1, ...]
            num_repeats = math.ceil(n / c)
            indices = torch.arange(c, device=self.device, dtype=torch.int64).repeat(num_repeats)
            indices = indices[:n]  # Truncate to exactly n samples
            
            # Shuffle this layer independently to remove correlation with other layers
            layer_perm = torch.randperm(n, generator=self.generator, device=self.device)
            indices = indices[layer_perm]
            
            center_indices_list.append(indices)

            if layer_idx == 0:
                scale = 1.0
            else:
                scale = math.pow(2.0, - layer_idx)
            x = x + scale * Q_l[indices]

        if not center_indices_list:
            raise RuntimeError("No layers defined; centers must be non-empty.")

        # The label is the center index of the last layer
        y = center_indices_list[-1].clone()

        if shuffle:
            perm = torch.randperm(n, generator=self.generator, device=self.device)
            x = x[perm]
            y = y[perm]
            # Apply the same permutation to all center indices
            center_indices_list = [indices[perm] for indices in center_indices_list]

        return x, y, center_indices_list


class TiltedKaleidoscope(Problem):
    """
    Binary classification variant of Kaleidoscope with probabilistic center selection.

    Unlike Kaleidoscope where the last layer's center determines the class, here we
    have a fixed binary classification (2 classes). Each layer's centers are split
    50/50 into two groups: those "tilted" toward class 0 and those toward class 1.

    The tilt per layer increases with layer index:
        tilt = 0.5 + tilt_increment * (layer_index + 1)
    where tilt_increment = 0.5 / len(centers).

    This means:
    - Earlier layers have tilt closer to 0.5 (more uniform selection)
    - The last layer always has tilt = 1.0 (deterministic assignment)

    Points of class 0 select centers tilted toward class 0 with probability
    tilt / (num_centers / 2), and centers tilted toward class 1 with probability
    (1 - tilt) / (num_centers / 2). Vice versa for class 1 points.
    """

    def __init__(
        self,
        d: int,
        centers: Sequence[int],
        device: torch.device | None = None,
        generator: Optional[torch.Generator] = None,
        is_standard_basis: bool = False,
    ) -> None:
        """
        Args:
            d: Dimensionality of the input space
            centers: Sequence of center counts per layer. Each must be divisible by 2.
            device: torch device to use for tensors
            generator: optional random generator for reproducibility
            is_standard_basis: If True, use standard basis vectors instead of random
                               orthonormal directions.
        """
        self._d: int = _validate_positive(d, "d")
        if not centers:
            raise ValueError("centers must be a non-empty sequence")

        self.centers: List[int] = []
        for idx, c in enumerate(centers):
            c = _validate_positive(c, f"C_{idx}")
            if c > self._d:
                raise ValueError(f"C_{idx} must be <= d (got {c} > {self._d})")
            if c % 2 != 0:
                raise ValueError(f"C_{idx} must be divisible by 2 (got {c})")
            self.centers.append(c)

        self.num_class = 2  # Always binary classification
        self.device = device if device is not None else torch.device("cpu")
        self.generator = generator
        self.is_standard_basis = is_standard_basis

        # Compute tilt_increment: ensures last layer has tilt = 1.0
        self.tilt_increment = 0.5 / len(self.centers)

        # Build Q_layers and tilt assignments
        self.Q_layers: List[torch.Tensor] = []
        self.tilt_assignments: List[torch.Tensor] = []  # 0 or 1 for each center

        for c in self.centers:
            if self.is_standard_basis:
                Q_l = _standard_basis_rows(c, self._d, self.generator, self.device)
            else:
                Q_l = _orthonormal_rows(c, self._d, self.generator, self.device)
            self.Q_layers.append(Q_l.to(dtype=torch.float32))

            # Randomly assign 50% of centers to class 0, 50% to class 1
            # Create [0, 0, ..., 1, 1, ...] and shuffle
            half = c // 2
            assignment = torch.cat([
                torch.zeros(half, device=self.device, dtype=torch.int64),
                torch.ones(half, device=self.device, dtype=torch.int64),
            ])
            perm = torch.randperm(c, generator=self.generator, device=self.device)
            assignment = assignment[perm]
            self.tilt_assignments.append(assignment)

    @property
    def d(self) -> int:
        return self._d

    def num_classes(self) -> int:
        return self.num_class

    def generate_dataset(
        self,
        n: int,
        clean_mode: bool = False,
        shuffle: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """
        Generate a dataset of size n with exactly 50/50 class balance.

        Args:
            n: Number of samples to generate. Must be even for exact 50/50 balance.
            clean_mode: Unused, kept for API compatibility.
            shuffle: If True, randomly permute the resulting dataset.

        Returns:
            (x, y, center_indices_list):
              - x: shape (n, d) float32 tensor of features
              - y: shape (n,) int64 tensor of class labels (0 or 1)
              - center_indices_list: List of tensors, one per layer, each shape (n,)
        """
        del clean_mode  # unused
        n = _validate_positive(n, "n")
        if n % 2 != 0:
            raise ValueError(f"n must be even for 50/50 class balance, got {n}")

        # Generate class labels: first half class 0, second half class 1
        y = torch.cat([
            torch.zeros(n // 2, device=self.device, dtype=torch.int64),
            torch.ones(n // 2, device=self.device, dtype=torch.int64),
        ])

        L = len(self.centers)
        x = torch.zeros(n, self._d, device=self.device, dtype=torch.float32)
        center_indices_list: List[torch.Tensor] = []

        for layer_idx, (Q_l, c, assignment) in enumerate(
            zip(self.Q_layers, self.centers, self.tilt_assignments)
        ):
            half_c = c // 2

            # Compute per-layer tilt: increases from ~0.5 at layer 0 to 1.0 at last layer
            layer_tilt = 0.5 + self.tilt_increment * (layer_idx + 1)

            # Get indices of centers tilted to class 0 and class 1
            centers_tilted_0 = (assignment == 0).nonzero(as_tuple=True)[0]  # shape (half_c,)
            centers_tilted_1 = (assignment == 1).nonzero(as_tuple=True)[0]  # shape (half_c,)

            if layer_idx == L - 1:
                # Last layer: deterministic assignment (tilt = 1.0)
                # Class 0 points get centers tilted to 0, class 1 points get centers tilted to 1
                indices_class0 = centers_tilted_0[
                    torch.randint(
                        half_c,
                        (n // 2,),
                        generator=self.generator,
                        device=self.device,
                    )
                ]
                indices_class1 = centers_tilted_1[
                    torch.randint(
                        half_c,
                        (n // 2,),
                        generator=self.generator,
                        device=self.device,
                    )
                ]
                indices = torch.cat([indices_class0, indices_class1])
            else:
                # Probabilistic sampling based on layer_tilt
                p_same = layer_tilt / half_c
                p_diff = (1.0 - layer_tilt) / half_c

                # For class 0: higher probability for centers tilted to 0
                probs_class0 = torch.where(
                    assignment == 0,
                    torch.tensor(p_same, device=self.device),
                    torch.tensor(p_diff, device=self.device),
                )

                # For class 1: higher probability for centers tilted to 1
                probs_class1 = torch.where(
                    assignment == 1,
                    torch.tensor(p_same, device=self.device),
                    torch.tensor(p_diff, device=self.device),
                )

                indices_class0 = torch.multinomial(
                    probs_class0.unsqueeze(0).expand(n // 2, -1),
                    num_samples=1,
                    replacement=True,
                    generator=self.generator,
                ).squeeze(-1)

                indices_class1 = torch.multinomial(
                    probs_class1.unsqueeze(0).expand(n // 2, -1),
                    num_samples=1,
                    replacement=True,
                    generator=self.generator,
                ).squeeze(-1)

                indices = torch.cat([indices_class0, indices_class1])

            center_indices_list.append(indices)

            # Compute contribution to x
            if layer_idx == 0:
                scale = 1.0
            else:
                scale = math.pow(2.0, -layer_idx)
            x = x + scale * Q_l[indices]

        if shuffle:
            perm = torch.randperm(n, generator=self.generator, device=self.device)
            x = x[perm]
            y = y[perm]
            center_indices_list = [indices[perm] for indices in center_indices_list]

        return x, y, center_indices_list
