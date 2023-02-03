import numpy as np
import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader, SubsetRandomSampler, TensorDataset


BIG_BATCH = 500
CIFAR_NUM_CLASSES = 10


def cifar10(data_dir, examples_per_class, batch_size, m):
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
    )

    # define transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize,
    ])

    original_dataset = datasets.CIFAR10(
        root=data_dir, train=True,
        download=True, transform=transform,
    )

    data_loader = DataLoader(original_dataset, batch_size=BIG_BATCH, shuffle=True)

    # Get the first BIG_BATCH images and labels
    for images, labels in data_loader:
        break

    indices = []
    # For each class get every index where it occurs
    per_class_indices_list = []
    for c in range(CIFAR_NUM_CLASSES):
        is_c = torch.eq(labels, c)
        indices_of_c = torch.nonzero(is_c)
        per_class_indices_list.append(indices_of_c)
    # Now take from each class one by one
    total_examples_per_class = examples_per_class * 3
    for i in range(total_examples_per_class):
        for c in range(CIFAR_NUM_CLASSES):
            index = per_class_indices_list[c][i]
            indices.append(index)

    balanced_images = images[indices]
    balanced_labels = labels[indices]
    balanced_dataset = TensorDataset(balanced_images, balanced_labels)

    total_n = total_examples_per_class * CIFAR_NUM_CLASSES
    n = total_n // 3
    indices = list(range(total_n))
    all_train_idx, valid_idx = indices[n:], indices[:n]

    splits = m // 2
    train_idx_list = []
    for s in range(splits):
        np.random.shuffle(all_train_idx)
        train_idx_list.append(all_train_idx[:n])
        train_idx_list.append(all_train_idx[n:])

    train_loaders = []
    for train_idx in train_idx_list:
        train_sampler = SubsetRandomSampler(train_idx)
        train_loader = DataLoader(balanced_dataset, batch_size=batch_size, sampler=train_sampler)
        train_loaders.append(train_loader)

    valid_sampler = SubsetRandomSampler(valid_idx)
    valid_loader = torch.utils.data.DataLoader(balanced_dataset, batch_size=batch_size, sampler=valid_sampler)
    return train_loaders, valid_loader
