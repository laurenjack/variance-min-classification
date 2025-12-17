from __future__ import annotations

from typing import List, Optional, Tuple

import torch

from jl.problem import Problem
from jl.feature_experiments.feature_problem import _orthonormal_rows


# Constants
ATOMIC_D = 4  # Dimensionality of each atomic subsection
NUM_FEATURES = 4  # Number of possible features per subsection


def _build_consolidation_mapping(
    generator: Optional[torch.Generator],
    device: torch.device,
) -> torch.Tensor:
    """
    Build a consolidation mapping using completely random assignment.

    Generates a random permutation of the 16 combinations and assigns
    the first 4 to feature 0, next 4 to feature 1, etc.

    This does NOT guarantee linear separability.

    Returns:
        shape (16,) tensor mapping combination index (i*4 + j) to feature (0-3)
    """
    # Generate random permutation of 16 combinations
    perm = torch.randperm(16, generator=generator, device=device)

    # Assign features: first 4 in permutation → feature 0, next 4 → feature 1, etc.
    mapping = torch.zeros(16, dtype=torch.int64, device=device)
    for feature_idx in range(4):
        start = feature_idx * 4
        end = start + 4
        for pos in range(start, end):
            combo_idx = perm[pos].item()
            mapping[combo_idx] = feature_idx

    return mapping


class FeatureCombinations(Problem):
    """
    Hierarchical feature combination problem for binary classification.

    The input space has d = 2 * 2^num_layers dimensions. At the atomic level
    (layer 0), the space is divided into num_subs = d // 4 subsections of 4
    dimensions each (atomic_d = 4).

    Each atomic subsection has its own orthonormal basis of 4 random feature
    vectors (a 4×4 orthonormal matrix). When choosing an atomic feature for a
    subsection, we select one of these 4 basis vectors.

    Higher layers consolidate pairs of adjacent subsections using completely
    random assignment. Each pair of subsections has 4×4 = 16 possible combinations,
    which are partitioned into 4 groups of 4 via random permutation:
    1. Generate a random permutation of the 16 combinations
    2. Assign first 4 to feature 0, next 4 to feature 1, etc.

    This does NOT guarantee linear separability at each layer.

    At the final layer (1 subsection with 4 possible features), 2 features are
    randomly assigned to class 0 and 2 to class 1.

    Attributes:
        num_layers: Number of layers in the hierarchy.
        d: Dimensionality of input space (2 * 2^num_layers).
        num_class: Always 2 (binary classification).
    """

    def __init__(
        self,
        num_layers: int,
        random_basis: bool = False,
        device: Optional[torch.device] = None,
        generator: Optional[torch.Generator] = None,
    ) -> None:
        """
        Args:
            num_layers: Number of layers. Must be >= 2. Determines d = 2 * 2^num_layers.
            random_basis: If True, rotate the input space with a random orthonormal matrix.
            device: torch device to use for tensors.
            generator: Optional random generator for reproducibility.
        """
        if num_layers < 2:
            raise ValueError(f"num_layers must be >= 2, got {num_layers}")

        self.num_layers = num_layers
        self._d = 2 * (2 ** num_layers)
        self.num_class = 2
        self.random_basis = random_basis
        self.device = device if device is not None else torch.device("cpu")
        self.generator = generator

        # Number of atomic subsections
        self.num_atomic_subsections = self._d // ATOMIC_D

        # Generate orthonormal basis for each atomic subsection
        # atomic_bases[i] is a (4, 4) orthonormal matrix for subsection i
        self.atomic_bases: List[torch.Tensor] = []
        for _ in range(self.num_atomic_subsections):
            Q_i = _orthonormal_rows(
                NUM_FEATURES, ATOMIC_D, self.generator, self.device
            )
            self.atomic_bases.append(Q_i)

        # Build consolidation mappings for each layer transition
        # consolidation_mappings[layer][subsection] is a (16,) tensor
        # mapping combination index (i*4 + j) to output feature (0-3)
        self.consolidation_mappings: List[List[torch.Tensor]] = []

        num_subsections = self.num_atomic_subsections
        for layer in range(1, self.num_layers):
            num_subsections_this_layer = num_subsections // 2
            layer_mappings: List[torch.Tensor] = []
            for _ in range(num_subsections_this_layer):
                mapping = _build_consolidation_mapping(self.generator, self.device)
                layer_mappings.append(mapping)
            self.consolidation_mappings.append(layer_mappings)
            num_subsections = num_subsections_this_layer

        # At the final layer, randomly assign 2 features to class 0, 2 to class 1
        assignment = torch.tensor([0, 0, 1, 1], dtype=torch.int64, device=self.device)
        perm = torch.randperm(4, generator=self.generator, device=self.device)
        self.final_class_assignment = assignment[perm]

        # If random_basis, generate a random orthonormal rotation matrix
        self.Q_rotation: Optional[torch.Tensor] = None
        if self.random_basis:
            self.Q_rotation = _orthonormal_rows(
                self._d, self._d, self.generator, self.device
            )

    @property
    def d(self) -> int:
        """The total dimensionality of the feature space."""
        return self._d

    def num_classes(self) -> int:
        """Returns the number of classes (always 2 for this problem)."""
        return self.num_class

    def _consolidate_features(
        self,
        features_a: torch.Tensor,
        features_b: torch.Tensor,
        mapping: torch.Tensor,
    ) -> torch.Tensor:
        """
        Consolidate two adjacent subsections' features using pre-computed mapping.

        Args:
            features_a: shape (n,) feature indices (0-3) from first subsection
            features_b: shape (n,) feature indices (0-3) from second subsection
            mapping: shape (16,) tensor mapping combination index to output feature

        Returns:
            shape (n,) consolidated feature indices (0-3)
        """
        combo_idx = features_a * 4 + features_b
        return mapping[combo_idx]

    def generate_dataset(
        self,
        n: int,
        clean_mode: bool = False,
        shuffle: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """
        Generate a dataset of size n.

        Args:
            n: Number of samples to generate.
            clean_mode: Unused (no noise in this problem).
            shuffle: If True, randomly permute the resulting dataset.

        Returns:
            (x, y, center_indices_list):
              - x: shape (n, d) float32 tensor of features
              - y: shape (n,) int64 tensor of class labels (0 or 1)
              - center_indices_list: List of tensors, one per layer.
                  - center_indices_list[0]: shape (n, num_atomic_subsections) atomic features
                  - center_indices_list[l]: shape (n, num_subsections_at_layer_l)
                  - center_indices_list[-1]: shape (n,) final layer features
        """
        del clean_mode  # unused

        if n <= 0:
            raise ValueError("n must be positive")

        # Generate atomic features: shape (n, num_atomic_subsections)
        # Each element is uniformly random in {0, 1, 2, 3}
        atomic_features = torch.randint(
            0, NUM_FEATURES,
            (n, self.num_atomic_subsections),
            generator=self.generator,
            device=self.device,
            dtype=torch.int64,
        )

        # Track features at each layer
        center_indices_list: List[torch.Tensor] = [atomic_features.clone()]

        # Propagate through layers
        current_features = atomic_features  # shape (n, num_subsections)

        for layer_mappings in self.consolidation_mappings:
            num_subsections = current_features.shape[1]
            num_new_subsections = num_subsections // 2

            new_features = torch.zeros(
                n, num_new_subsections,
                dtype=torch.int64,
                device=self.device,
            )

            for sub_idx in range(num_new_subsections):
                features_a = current_features[:, 2 * sub_idx]
                features_b = current_features[:, 2 * sub_idx + 1]
                mapping = layer_mappings[sub_idx]
                new_features[:, sub_idx] = self._consolidate_features(
                    features_a, features_b, mapping
                )

            current_features = new_features

            # Store: if final layer, squeeze to (n,), else keep (n, num_subsections)
            if current_features.shape[1] == 1:
                center_indices_list.append(current_features.squeeze(1))
            else:
                center_indices_list.append(current_features.clone())

        # Final layer features determine class
        final_features = center_indices_list[-1]  # shape (n,)
        y = self.final_class_assignment[final_features]

        # Construct x from atomic features using per-subsection orthonormal bases
        # For each subsection i, look up atomic_bases[i][feature_idx] to get the 4D vector
        x = torch.zeros(n, self._d, device=self.device, dtype=torch.float32)

        for sub_idx in range(self.num_atomic_subsections):
            basis = self.atomic_bases[sub_idx]  # (4, 4)
            features = atomic_features[:, sub_idx]  # (n,)
            vectors = basis[features]  # (n, 4)

            start_dim = sub_idx * ATOMIC_D
            end_dim = start_dim + ATOMIC_D
            x[:, start_dim:end_dim] = vectors

        # Apply random basis rotation if enabled
        if self.Q_rotation is not None:
            x = x @ self.Q_rotation.T

        # Shuffle if requested
        if shuffle:
            perm = torch.randperm(n, generator=self.generator, device=self.device)
            x = x[perm]
            y = y[perm]
            center_indices_list = [indices[perm] for indices in center_indices_list]

        return x, y, center_indices_list
