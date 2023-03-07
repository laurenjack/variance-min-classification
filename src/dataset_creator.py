import math
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


def class_pattern_with_noise(n, num_class, noisy_d, percent_correct=1.0, noisy_dim_scalar=1.0):
    num_correct = int(round(percent_correct * n))
    class_d = math.ceil(np.log2(num_class))
    ones = np.ones(class_d)
    class_to_perm = np.array(get_perms(class_d, ones)).astype(np.float32)
    class_to_perm = torch.tensor(class_to_perm)
    # Now shuffle, so that each class has a random permutation
    shuffler = torch.randperm(num_class)
    class_to_perm = class_to_perm[shuffler]
    x = []
    y = []
    for i in range(n):
        c = i % num_class
        first_part = class_to_perm[c]
        # Once we get more than percent_correct through the dataset, start randomly changing the class.
        # Yes this throws off perfect class balance, but still a uniform prob of each class
        if i >= num_correct:
            # Starting at 1 means we cant get the same class back
            random_offset = torch.randint(low=1, high=num_class, size=(1,))[0].item()
            c = (c + random_offset) % num_class
        noisy_dims = float(noisy_dim_scalar) * (torch.randint(low=0, high=2, size=(noisy_d,)) * 2 - 1)
        example = torch.cat([first_part, noisy_dims])
        x.append(example)
        y.append(c)

    return TensorDataset(torch.stack(x), torch.tensor(y))


def get_perms(i, perm):
    if i == 0:
        return [perm]
    i -= 1
    left = perm.copy()
    left[i] = -1.0
    left_perms = get_perms(i, left)
    right = perm.copy()
    right[i] = 1.0
    right_perms = get_perms(i, right)
    return left_perms + right_perms
