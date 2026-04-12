"""CIFAR-10 data loading with label noise."""

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, Subset
import torchvision
import torchvision.transforms as transforms
import numpy as np
from typing import Optional, Tuple


# Deterministic seed for the 45K/5K train/val split. Fixed so the same
# val set is reserved across all k values and all reruns.
VAL_SPLIT_SEED = 73132
DEFAULT_VAL_SIZE = 5000
DEFAULT_NOISE_SEED = 42


def download_cifar10(data_dir: str = "./data") -> None:
    """Download CIFAR-10 dataset if not already present.

    Call this before spawning multiprocessing workers to avoid
    concurrent download issues.

    Args:
        data_dir: Directory to store CIFAR-10 data.
    """
    # Download train and test sets
    torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True)
    torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True)


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
    num_workers: int = 0,
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



def compute_val_split_indices(
    total_samples: int = 50000,
    val_size: int = DEFAULT_VAL_SIZE,
    seed: int = VAL_SPLIT_SEED,
) -> Tuple[np.ndarray, np.ndarray]:
    """Deterministic split of CIFAR-10 training indices into train and val.

    Uses a fixed seed (default 73132) so the same val set is reserved across
    all k values and reruns. First (total - val_size) indices in the shuffled
    permutation go to train; the last val_size go to val.
    """
    rng = np.random.RandomState(seed)
    perm = rng.permutation(total_samples)
    return perm[:-val_size], perm[-val_size:]


def save_val_set(
    output_path: str,
    data_dir: str = "./data",
    noise_prob: float = 0.15,
    noise_seed: int = DEFAULT_NOISE_SEED,
    val_split_seed: int = VAL_SPLIT_SEED,
    val_size: int = DEFAULT_VAL_SIZE,
) -> Path:
    """Build and save the held-out validation set to <output_path>/val.pt.

    The val set is carved from the full noised training set (noise applied to
    all 50K with noise_seed first, THEN split via val_split_seed). Val labels
    are therefore noisy — same distribution as the training labels. Val uses
    NON-augmented transforms (test-time normalization only).

    Saved file format (torch.save dict):
        {
            "images":      FloatTensor [val_size, 3, 32, 32]  normalized
            "labels":      LongTensor  [val_size]              (possibly noisy)
            "indices":     LongTensor  [val_size]              into CIFAR-10 train
            "noise_seed":  int
            "split_seed":  int
            "noise_prob":  float
        }
    """
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
    )
    test_transform = transforms.Compose([transforms.ToTensor(), normalize])

    full_train = NoisyCIFAR10(
        root=data_dir,
        train=True,
        noise_prob=noise_prob,
        transform=test_transform,
        seed=noise_seed,
    )

    _, val_indices = compute_val_split_indices(
        total_samples=len(full_train),
        val_size=val_size,
        seed=val_split_seed,
    )

    val_images = []
    val_labels = []
    for idx in val_indices:
        img, label = full_train[int(idx)]
        val_images.append(img)
        val_labels.append(label)

    val_images_tensor = torch.stack(val_images)                 # [val_size, 3, 32, 32]
    val_labels_tensor = torch.tensor(val_labels, dtype=torch.long)
    val_indices_tensor = torch.tensor(val_indices, dtype=torch.long)

    val_path = Path(output_path) / "val.pt"
    val_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "images": val_images_tensor,
            "labels": val_labels_tensor,
            "indices": val_indices_tensor,
            "noise_seed": noise_seed,
            "split_seed": val_split_seed,
            "noise_prob": noise_prob,
        },
        val_path,
    )
    return val_path


def load_cifar10_with_noise_val_split(
    noise_prob: float,
    batch_size: int,
    data_augmentation: bool = True,
    data_dir: str = "./data",
    num_workers: int = 0,
    val_size: int = DEFAULT_VAL_SIZE,
    val_split_seed: int = VAL_SPLIT_SEED,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Load CIFAR-10 with noise, split 45K train / 5K val / full test.

    Mirrors load_cifar10_with_noise except it holds out a deterministic 5K
    validation subset (seed=val_split_seed) from the noised 50K training
    set. Val samples use the non-augmented test_transform; train uses the
    augmented transform as before. Val labels are noisy (same distribution
    as train) since noise is applied before the split.

    Both train and val use the same NoisyCIFAR10 underlying dataset in two
    copies with different transforms — labels are byte-identical because
    they share the same (default) noise seed.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
    )

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

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    train_full = NoisyCIFAR10(
        root=data_dir,
        train=True,
        noise_prob=noise_prob,
        transform=train_transform,
    )
    val_full = NoisyCIFAR10(
        root=data_dir,
        train=True,
        noise_prob=noise_prob,
        transform=test_transform,
    )

    train_indices, val_indices = compute_val_split_indices(
        total_samples=len(train_full),
        val_size=val_size,
        seed=val_split_seed,
    )

    train_subset = Subset(train_full, train_indices.tolist())
    val_subset = Subset(val_full, val_indices.tolist())

    test_dataset = NoisyCIFAR10(
        root=data_dir,
        train=False,
        noise_prob=0.0,
        transform=test_transform,
    )

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
