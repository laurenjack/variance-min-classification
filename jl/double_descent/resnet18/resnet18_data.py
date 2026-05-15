"""CIFAR-10 data loading with label noise."""

from pathlib import Path

import torch
import torch.nn.functional as F
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

# Deterministic seed for the disjoint-train partition used by variance mode.
# Different from VAL_SPLIT_SEED so the val carve and the variance carve are
# independent shuffles.
VARIANCE_SPLIT_SEED = 31337


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
        (train_loader, val_loader, test_loader).
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


# ---------------------------------------------------------------------------
# GPU-resident data pipeline
# ---------------------------------------------------------------------------
#
# The CPU DataLoader path above is fine for portability but is dominated by
# PIL decoding + RandomCrop + RandomHorizontalFlip on the worker process's
# main thread, which leaves the H100 at ~10% util. The classes/functions below
# preload the whole (already-normalized) CIFAR-10 train/test as FP32 tensors
# on the target GPU, then do RandomCrop(32, padding=4) + RandomHorizontalFlip
# as batched tensor ops on-device every step. CIFAR fits trivially: train is
# 50000 * 3 * 32 * 32 * 4 B = ~614 MB per GPU.

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)


def _build_gpu_cifar_tensors(
    data_dir: str, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load CIFAR-10 once and return UNnormalized FP32 train/test tensors on device.

    Values are in [0, 1] (uint8 / 255). Augmentation runs in this raw pixel
    space — pad-with-zero is then byte-equivalent to torchvision's PIL
    RandomCrop with the default constant fill=0. Normalization happens
    after augmentation, inside the loader's per-batch _normalize.

    Returns (train_images, train_orig_labels, test_images, test_labels).
    """
    train = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=False)
    test = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=False)

    def to_raw(ds: torchvision.datasets.CIFAR10) -> torch.Tensor:
        # ds.data is a uint8 numpy array of shape [N, 32, 32, 3].
        imgs = torch.from_numpy(ds.data).to(device=device, dtype=torch.float32)
        return imgs.permute(0, 3, 1, 2).contiguous() / 255.0

    train_images = to_raw(train)
    test_images = to_raw(test)
    train_orig_labels = torch.tensor(train.targets, dtype=torch.long, device=device)
    test_labels = torch.tensor(test.targets, dtype=torch.long, device=device)
    return train_images, train_orig_labels, test_images, test_labels


def _apply_label_noise_tensor(
    orig_labels: torch.Tensor,
    noise_prob: float,
    seed: int,
    n_classes: int = 10,
) -> torch.Tensor:
    """Apply label noise to a labels tensor. Returns the noisy labels tensor.

    Uses the same procedure (numpy RandomState, per-sample wrong-label
    sampling) as NoisyCIFAR10 so the label sequence is byte-identical to
    the CPU path under the same seed.
    """
    rng = np.random.RandomState(seed)
    orig = orig_labels.detach().cpu().numpy().copy()
    noisy = orig.copy()
    n = len(noisy)
    corrupt_mask = rng.rand(n) < noise_prob
    for i in range(n):
        if corrupt_mask[i]:
            wrong = [l for l in range(n_classes) if l != orig[i]]
            noisy[i] = rng.choice(wrong)
    return torch.from_numpy(noisy).to(orig_labels.device).long()


def gpu_random_crop_flip(images: torch.Tensor, padding: int = 4) -> torch.Tensor:
    """RandomCrop(32, padding=4) + RandomHorizontalFlip, vectorized on device.

    Pads zero (the original PIL behavior with padding_mode='constant'), picks
    one random crop offset and one flip decision per sample in the batch.
    """
    B, C, H, W = images.shape
    flip = torch.rand(B, device=images.device) < 0.5
    flipped = images.flip(-1)
    images = torch.where(flip.view(B, 1, 1, 1), flipped, images)

    padded = F.pad(images, (padding, padding, padding, padding), mode="constant", value=0.0)
    max_offset = 2 * padding + 1
    dy = torch.randint(0, max_offset, (B,), device=images.device)
    dx = torch.randint(0, max_offset, (B,), device=images.device)
    rows = dy.view(B, 1, 1, 1) + torch.arange(H, device=images.device).view(1, 1, H, 1)
    cols = dx.view(B, 1, 1, 1) + torch.arange(W, device=images.device).view(1, 1, 1, W)
    batch_idx = torch.arange(B, device=images.device).view(B, 1, 1, 1)
    chan_idx = torch.arange(C, device=images.device).view(1, C, 1, 1)
    return padded[batch_idx, chan_idx, rows, cols]


class _SizedDataset:
    """Stand-in for DataLoader.dataset that only needs to support len()."""

    def __init__(self, n: int):
        self._n = n

    def __len__(self) -> int:
        return self._n


def _normalize_buffers(device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    mean = torch.tensor(CIFAR10_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR10_STD, device=device).view(1, 3, 1, 1)
    return mean, std


class GPUTrainLoader:
    """Iterator yielding (images, labels) batches on device.

    Both tensors are on `images.device`. Each step: shuffle indices, gather
    the raw batch, apply on-GPU RandomCrop+HFlip (if augment=True),
    normalize, emit.
    """

    def __init__(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        batch_size: int,
        augment: bool = True,
        drop_last: bool = True,
    ):
        self.images = images
        self.labels = labels
        self.batch_size = batch_size
        self.augment = augment
        self.drop_last = drop_last
        self.n = images.size(0)
        self.dataset = _SizedDataset(self.n)
        self._mean, self._std = _normalize_buffers(images.device)

    def __len__(self) -> int:
        if self.drop_last:
            return self.n // self.batch_size
        return (self.n + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        perm = torch.randperm(self.n, device=self.images.device)
        end = (self.n // self.batch_size) * self.batch_size if self.drop_last else self.n
        for start in range(0, end, self.batch_size):
            idx = perm[start:start + self.batch_size]
            imgs = self.images[idx]
            if self.augment:
                imgs = gpu_random_crop_flip(imgs)
            imgs = (imgs - self._mean) / self._std
            yield imgs, self.labels[idx]


class GPUEvalLoader:
    """Sequential iterator yielding (images, labels) batches on device."""

    def __init__(self, images: torch.Tensor, labels: torch.Tensor, batch_size: int):
        self.images = images
        self.labels = labels
        self.batch_size = batch_size
        self.n = images.size(0)
        self.dataset = _SizedDataset(self.n)
        self._mean, self._std = _normalize_buffers(images.device)

    def __len__(self) -> int:
        return (self.n + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        for start in range(0, self.n, self.batch_size):
            end = min(start + self.batch_size, self.n)
            imgs = (self.images[start:end] - self._mean) / self._std
            yield imgs, self.labels[start:end]


def load_cifar10_with_noise_val_split_gpu(
    noise_prob: float,
    batch_size: int,
    device: torch.device,
    data_dir: str = "./data",
    data_augmentation: bool = True,
    noise_seed: int = DEFAULT_NOISE_SEED,
    val_size: int = DEFAULT_VAL_SIZE,
    val_split_seed: int = VAL_SPLIT_SEED,
) -> Tuple[GPUTrainLoader, GPUEvalLoader, GPUEvalLoader]:
    """GPU-resident counterpart to load_cifar10_with_noise_val_split.

    Returns (train_loader, val_loader, test_loader); loaders emit tensors
    already on `device`. Label noise + split semantics exactly match the CPU
    path under the same seeds (noise applied to all 50K first, THEN split
    via val_split_seed).
    """
    train_images, train_orig, test_images, test_labels = (
        _build_gpu_cifar_tensors(data_dir, device)
    )
    train_noisy = _apply_label_noise_tensor(
        train_orig, noise_prob, noise_seed
    )

    train_indices, val_indices = compute_val_split_indices(
        total_samples=train_images.size(0),
        val_size=val_size,
        seed=val_split_seed,
    )
    train_idx_t = torch.from_numpy(train_indices).to(device)
    val_idx_t = torch.from_numpy(val_indices).to(device)

    train_sub_imgs = train_images.index_select(0, train_idx_t).contiguous()
    train_sub_labels = train_noisy.index_select(0, train_idx_t).contiguous()
    val_sub_imgs = train_images.index_select(0, val_idx_t).contiguous()
    val_sub_labels = train_noisy.index_select(0, val_idx_t).contiguous()

    train_loader = GPUTrainLoader(
        train_sub_imgs, train_sub_labels, batch_size,
        augment=data_augmentation, drop_last=True,
    )
    val_loader = GPUEvalLoader(val_sub_imgs, val_sub_labels, batch_size)
    test_loader = GPUEvalLoader(test_images, test_labels, batch_size)
    return train_loader, val_loader, test_loader


# ---------------------------------------------------------------------------
# Variance mode: disjoint training splits
# ---------------------------------------------------------------------------


def compute_disjoint_split_indices(
    pool_indices: np.ndarray,
    num_splits: int,
    seed: int = VARIANCE_SPLIT_SEED,
) -> list:
    """Shuffle pool_indices and partition into num_splits disjoint chunks.

    pool_indices is the array of CIFAR-10 indices eligible for training
    (e.g. the 45K train indices left after the val carve). Returns a list
    of length num_splits, each entry a numpy int array of size
    len(pool_indices) // num_splits.
    """
    rng = np.random.RandomState(seed)
    shuffled = rng.permutation(pool_indices)
    chunk = len(shuffled) // num_splits
    return [shuffled[i * chunk:(i + 1) * chunk] for i in range(num_splits)]


def load_cifar10_variance_split_gpu(
    noise_prob: float,
    batch_size: int,
    device: torch.device,
    split_id: int,
    num_splits: int,
    data_dir: str = "./data",
    data_augmentation: bool = True,
    noise_seed: int = DEFAULT_NOISE_SEED,
    val_size: int = DEFAULT_VAL_SIZE,
    val_split_seed: int = VAL_SPLIT_SEED,
    split_seed: int = VARIANCE_SPLIT_SEED,
) -> Tuple[GPUTrainLoader, GPUEvalLoader, GPUEvalLoader]:
    """GPU-resident CIFAR-10 with disjoint training splits for variance mode.

    Pipeline:
      1. Load all 50K train + 10K test as FP32 tensors on device.
      2. Apply label noise to all 50K (single noise_seed) so noise is shared
         across splits — each image, if it appears in any split, sees the
         same noisy label as it would in any other split that contained it.
      3. Carve a deterministic 5K val from the noised 50K (val_split_seed).
      4. Partition the remaining 45K into num_splits disjoint chunks
         (split_seed). Train on chunk `split_id`.

    val + test are shared across all (k, split) models; train is the
    disjoint chunk for this split_id.
    """
    if split_id < 0 or split_id >= num_splits:
        raise ValueError(f"split_id must be in [0, {num_splits - 1}], got {split_id}")

    train_images, train_orig, test_images, test_labels = (
        _build_gpu_cifar_tensors(data_dir, device)
    )
    train_noisy = _apply_label_noise_tensor(
        train_orig, noise_prob, noise_seed
    )

    train_pool, val_indices = compute_val_split_indices(
        total_samples=train_images.size(0),
        val_size=val_size,
        seed=val_split_seed,
    )

    chunks = compute_disjoint_split_indices(
        pool_indices=train_pool,
        num_splits=num_splits,
        seed=split_seed,
    )
    chunk_indices = chunks[split_id]

    train_idx_t = torch.from_numpy(chunk_indices.copy()).to(device)
    val_idx_t = torch.from_numpy(val_indices).to(device)

    train_sub_imgs = train_images.index_select(0, train_idx_t).contiguous()
    train_sub_labels = train_noisy.index_select(0, train_idx_t).contiguous()
    val_sub_imgs = train_images.index_select(0, val_idx_t).contiguous()
    val_sub_labels = train_noisy.index_select(0, val_idx_t).contiguous()

    train_loader = GPUTrainLoader(
        train_sub_imgs, train_sub_labels, batch_size,
        augment=data_augmentation, drop_last=True,
    )
    val_loader = GPUEvalLoader(val_sub_imgs, val_sub_labels, batch_size)
    test_loader = GPUEvalLoader(test_images, test_labels, batch_size)
    return train_loader, val_loader, test_loader
