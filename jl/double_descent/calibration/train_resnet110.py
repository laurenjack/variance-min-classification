"""Train ResNet-110 on CIFAR-10 or CIFAR-100.

Follows the He et al. 2015 / Guo et al. 2017 recipe:
- 200 epochs, SGD momentum 0.9, weight decay 1e-4, batch size 128
- LR 0.1, dropped by 10x at epochs 100 and 150
- Standard CIFAR augmentation: RandomCrop(32, padding=4), RandomHorizontalFlip
- 50K train split into 45K train + 5K val (fixed seed)

Usage:
    python -m jl.double_descent.calibration.train_resnet110 \
        --dataset cifar10 --data-path ./data --output-path ./output
"""

import argparse
import logging
import time
from pathlib import Path

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from jl.double_descent.calibration.resnet110 import make_resnet110

logger = logging.getLogger(__name__)

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)


def get_cifar_splits(dataset: str, data_path: str, seed: int):
    """Load CIFAR and split 50K train into 45K train + 5K val.

    Returns (train_dataset, val_dataset, test_dataset) with appropriate transforms.
    Train has augmentation; val and test have only normalize.
    """
    if dataset == "cifar10":
        mean, std = CIFAR10_MEAN, CIFAR10_STD
        DatasetClass = torchvision.datasets.CIFAR10
    else:
        mean, std = CIFAR100_MEAN, CIFAR100_STD
        DatasetClass = torchvision.datasets.CIFAR100

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # Full 50K training set (with augmentation for train, without for val)
    full_train_aug = DatasetClass(data_path, train=True, download=True, transform=train_transform)
    full_train_eval = DatasetClass(data_path, train=True, download=True, transform=eval_transform)
    test_dataset = DatasetClass(data_path, train=False, download=True, transform=eval_transform)

    # Split 50K into 45K train + 5K val
    n = len(full_train_aug)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    train_idx = perm[:45000].tolist()
    val_idx = perm[45000:].tolist()

    train_dataset = torch.utils.data.Subset(full_train_aug, train_idx)
    val_dataset = torch.utils.data.Subset(full_train_eval, val_idx)

    return train_dataset, val_dataset, test_dataset


def evaluate(model: nn.Module, loader, device: torch.device):
    """Compute loss and accuracy on a data loader."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss(reduction="sum")
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            total_loss += criterion(logits, labels).item()
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
    return total_loss / total, correct / total


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Train ResNet-110 on CIFAR")
    parser.add_argument("--dataset", type=str, required=True, choices=["cifar10", "cifar100"])
    parser.add_argument("--data-path", type=str, default="./data")
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    num_classes = 10 if args.dataset == "cifar10" else 100
    output_dir = Path(args.output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / f"resnet110_{args.dataset}.pt"

    if checkpoint_path.exists():
        logger.info(f"Checkpoint already exists: {checkpoint_path}. Skipping training.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training ResNet-110 on {args.dataset} ({num_classes} classes), device={device}")

    # Set seed for reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Data
    train_dataset, val_dataset, _ = get_cifar_splits(args.dataset, args.data_path, args.seed)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=128, shuffle=True, num_workers=2, pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=128, shuffle=False, num_workers=2, pin_memory=True
    )
    logger.info(f"Train: {len(train_dataset)} samples, Val: {len(val_dataset)} samples")

    # Model
    model = make_resnet110(num_classes).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {param_count:,}")

    # Optimizer and scheduler
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * labels.size(0)
            train_correct += (logits.argmax(dim=1) == labels).sum().item()
            train_total += labels.size(0)

        scheduler.step()

        train_loss /= train_total
        train_acc = train_correct / train_total

        val_loss, val_acc = evaluate(model, val_loader, device)
        elapsed = time.time() - start

        logger.info(
            f"Epoch {epoch:3d}/{args.epochs} ({elapsed:.1f}s) | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
            f"lr={optimizer.param_groups[0]['lr']:.0e}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), checkpoint_path)
            logger.info(f"  -> New best val_acc={val_acc:.4f}, saved to {checkpoint_path}")

    logger.info(f"Training complete. Best val_acc={best_val_acc:.4f}")


if __name__ == "__main__":
    main()
