from abc import ABC, abstractmethod
from typing import Tuple

import torch


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
    def num_classes(self) -> int:
        """
        Returns the number of classes for this classification problem.
        
        Returns:
            int: Number of classes (>= 2)
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

