"""ImageNet-1K data loading via HuggingFace datasets.

Training uses RandomResizedCrop(224) + RandomHorizontalFlip.
Evaluation uses Resize(256) + CenterCrop(224).
Standard ImageNet normalization for both.

The 50K HuggingFace validation split is used as the test set (ImageNet
has no public test split).
"""

import logging
from typing import Tuple

import torch
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class HFImageNetDataset(torch.utils.data.Dataset):
    """Wraps a HuggingFace ImageNet split with torchvision transforms."""

    def __init__(self, hf_dataset, transform):
        self.hf_dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        example = self.hf_dataset[idx]
        img = example["image"]
        if img.mode != "RGB":
            img = img.convert("RGB")
        img = self.transform(img)
        return img, example["label"]


def build_imagenet_loaders(
    batch_size: int = 256,
    num_workers: int = 8,
    data_augmentation: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """Build ImageNet-1K train and validation DataLoaders from HuggingFace."""
    from datasets import load_dataset
    from torchvision import transforms

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    if data_augmentation:
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])

    eval_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    logger.info("Loading ImageNet-1K from HuggingFace (or cache)...")
    train_hf = load_dataset("ILSVRC/imagenet-1k", split="train")
    val_hf = load_dataset("ILSVRC/imagenet-1k", split="validation")
    logger.info(f"HuggingFace: train={len(train_hf)}, val={len(val_hf)}")

    train_dataset = HFImageNetDataset(train_hf, train_transform)
    val_dataset = HFImageNetDataset(val_hf, eval_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    return train_loader, val_loader
