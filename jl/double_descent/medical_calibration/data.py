"""APTOS-2019 data loading, splitting, and augmentation."""

import csv
import logging
import os
from pathlib import Path
from typing import Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from jl.double_descent.medical_calibration.config import MedCalConfig

logger = logging.getLogger(__name__)

APTOS_CLASSES = {
    0: "No DR",
    1: "Mild NPDR",
    2: "Moderate NPDR",
    3: "Severe NPDR",
    4: "Proliferative DR",
}


class APTOSDataset(Dataset):
    """APTOS-2019 dataset from a split manifest."""

    def __init__(self, manifest_path: str, image_dir: str, transform=None):
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.samples = []

        with open(manifest_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.samples.append((row["image_id"], int(row["label"])))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_id, label = self.samples[idx]
        img_path = self.image_dir / f"{image_id}.png"
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def download_aptos(data_dir: str) -> Path:
    """Download APTOS-2019 from Kaggle if not present.

    Requires KAGGLE_USERNAME and KAGGLE_KEY environment variables.

    Returns:
        Path to the extracted data directory.
    """
    data_path = Path(data_dir)
    image_dir = data_path / "train_images"

    if image_dir.exists() and any(image_dir.iterdir()):
        logger.info(f"APTOS-2019 already exists at {data_path}")
        return data_path

    logger.info("Downloading APTOS-2019 from Kaggle...")
    os.environ.setdefault("KAGGLE_USERNAME", "")
    os.environ.setdefault("KAGGLE_KEY", "")

    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    api.competition_download_files(
        "aptos2019-blindness-detection", path=str(data_path)
    )

    # Extract
    import zipfile
    zip_path = data_path / "aptos2019-blindness-detection.zip"
    if zip_path.exists():
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(data_path)
        zip_path.unlink()
        logger.info(f"Extracted APTOS-2019 to {data_path}")

    return data_path


def create_split_manifest(
    data_dir: str, output_dir: str, config: MedCalConfig
) -> Tuple[str, str, str]:
    """Create stratified train/val/test split manifest CSVs.

    Args:
        data_dir: Directory containing train.csv and train_images/.
        output_dir: Directory to write manifest CSVs.
        config: Config with split ratios and seed.

    Returns:
        (train_manifest, val_manifest, test_manifest) paths.
    """
    import random

    data_path = Path(data_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    train_manifest = output_path / "split_train.csv"
    val_manifest = output_path / "split_val.csv"
    test_manifest = output_path / "split_test.csv"

    # Check if manifests already exist
    if train_manifest.exists() and val_manifest.exists() and test_manifest.exists():
        logger.info("Split manifests already exist, skipping creation")
        return str(train_manifest), str(val_manifest), str(test_manifest)

    # Read original labels
    labels_csv = data_path / "train.csv"
    samples_by_class = {}
    with open(labels_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = int(row["diagnosis"])
            image_id = row["id_code"]
            samples_by_class.setdefault(label, []).append(image_id)

    # Stratified split
    rng = random.Random(config.split_seed)
    train_samples, val_samples, test_samples = [], [], []

    for label in sorted(samples_by_class.keys()):
        ids = samples_by_class[label][:]
        rng.shuffle(ids)
        n = len(ids)
        n_train = int(n * config.train_ratio)
        n_val = int(n * config.val_ratio)

        train_samples.extend((img_id, label) for img_id in ids[:n_train])
        val_samples.extend((img_id, label) for img_id in ids[n_train : n_train + n_val])
        test_samples.extend((img_id, label) for img_id in ids[n_train + n_val :])

    def _write_manifest(path, samples):
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["image_id", "label"])
            for img_id, label in samples:
                writer.writerow([img_id, label])

    _write_manifest(train_manifest, train_samples)
    _write_manifest(val_manifest, val_samples)
    _write_manifest(test_manifest, test_samples)

    logger.info(
        f"Created splits: train={len(train_samples)}, "
        f"val={len(val_samples)}, test={len(test_samples)}"
    )
    for label in sorted(samples_by_class.keys()):
        train_count = sum(1 for _, l in train_samples if l == label)
        val_count = sum(1 for _, l in val_samples if l == label)
        test_count = sum(1 for _, l in test_samples if l == label)
        logger.info(
            f"  Grade {label} ({APTOS_CLASSES[label]}): "
            f"train={train_count}, val={val_count}, test={test_count}"
        )

    return str(train_manifest), str(val_manifest), str(test_manifest)


def build_train_transform(config: MedCalConfig) -> transforms.Compose:
    """Build training augmentation pipeline."""
    return transforms.Compose([
        transforms.Resize((config.input_size, config.input_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(
            brightness=config.color_jitter,
            contrast=config.color_jitter,
            saturation=config.color_jitter,
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        transforms.RandomErasing(p=config.reprob),
    ])


def build_eval_transform(config: MedCalConfig) -> transforms.Compose:
    """Build evaluation transform (no augmentation)."""
    return transforms.Compose([
        transforms.Resize((config.input_size, config.input_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def build_dataloaders(
    data_dir: str,
    train_manifest: str,
    val_manifest: str,
    test_manifest: str,
    config: MedCalConfig,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Build train, val, and test dataloaders.

    Args:
        data_dir: Directory containing train_images/.
        train_manifest: Path to train split CSV.
        val_manifest: Path to val split CSV.
        test_manifest: Path to test split CSV.
        config: Experiment config.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    image_dir = str(Path(data_dir) / "train_images")

    train_dataset = APTOSDataset(
        train_manifest, image_dir, transform=build_train_transform(config)
    )
    val_dataset = APTOSDataset(
        val_manifest, image_dir, transform=build_eval_transform(config)
    )
    test_dataset = APTOSDataset(
        test_manifest, image_dir, transform=build_eval_transform(config)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
