"""CIFAR-10 data loading with label noise."""

import torch
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms
import numpy as np
from typing import Tuple


class NoisyCIFAR10(Dataset):
    """CIFAR-10 dataset with corrupted training labels.

    The label noise is applied once at dataset creation (not per-epoch).
    Test labels are never corrupted.
    """

    def __init__(
        self,
        root: str,
        train: bool,
        noise_prob: float,
        transform=None,
        download: bool = True,
        seed: int = 42,
    ):
        """Initialize NoisyCIFAR10.

        Args:
            root: Root directory for CIFAR-10 data.
            train: If True, load training set; else test set.
            noise_prob: Probability of corrupting each training label.
            transform: Optional transform to apply to images.
            download: Whether to download if not present.
            seed: Random seed for reproducible noise.
        """
        self.cifar = torchvision.datasets.CIFAR10(
            root=root,
            train=train,
            download=download,
        )
        self.transform = transform
        self.noise_prob = noise_prob
        self.train = train

        # Copy original labels
        self.labels = list(self.cifar.targets)

        # Apply label noise to training set only
        if train and noise_prob > 0:
            self._apply_noise(seed)

    def _apply_noise(self, seed: int):
        """Corrupt labels with probability noise_prob."""
        rng = np.random.RandomState(seed)
        n_samples = len(self.labels)
        n_classes = 10

        # Determine which samples get corrupted
        corrupt_mask = rng.rand(n_samples) < self.noise_prob

        # For corrupted samples, assign random wrong label
        for i in range(n_samples):
            if corrupt_mask[i]:
                original_label = self.labels[i]
                # Choose a different label uniformly at random
                wrong_labels = [l for l in range(n_classes) if l != original_label]
                self.labels[i] = rng.choice(wrong_labels)

    def __len__(self) -> int:
        return len(self.cifar)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image, _ = self.cifar[idx]  # Ignore original label
        label = self.labels[idx]

        if self.transform is not None:
            image = self.transform(image)

        return image, label


def load_cifar10_with_noise(
    noise_prob: float,
    batch_size: int,
    data_augmentation: bool = True,
    data_dir: str = "./data",
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader]:
    """Load CIFAR-10 with label noise on training set.

    Args:
        noise_prob: Probability of corrupting each training label (e.g., 0.15).
        batch_size: Batch size for data loaders.
        data_augmentation: Whether to apply data augmentation to training.
        data_dir: Directory to store/load CIFAR-10 data.
        num_workers: Number of data loading workers.

    Returns:
        Tuple of (train_loader, test_loader).
    """
    # Normalization values for CIFAR-10
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
    )

    # Training transforms
    if data_augmentation:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

    # Test transforms (no augmentation)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    # Create datasets
    train_dataset = NoisyCIFAR10(
        root=data_dir,
        train=True,
        noise_prob=noise_prob,
        transform=train_transform,
    )

    test_dataset = NoisyCIFAR10(
        root=data_dir,
        train=False,
        noise_prob=0.0,  # No noise on test set
        transform=test_transform,
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,  # For consistent batch sizes
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, test_loader
