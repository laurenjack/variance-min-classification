import math
import numpy as np
import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader, SubsetRandomSampler, TensorDataset, Subset


CIFAR_NUM_CLASSES = 10


def cifar10(data_dir, examples_per_class):
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

    # Split the dataset into class-specific indices
    class_indices_list = [[] for _ in range(CIFAR_NUM_CLASSES)]
    for index, label in enumerate(original_dataset.targets):
        class_indices_list[label].append(index)

    # Randomly sample the required number of examples from each class
    sampled_indices = []
    for class_indices in class_indices_list:
        # Generate a random permutation of the indices and select the first examples_per_class indices
        permuted_indices = torch.randperm(len(class_indices))
        selected_indices = [class_indices[i] for i in permuted_indices[:examples_per_class]]
        sampled_indices.extend(selected_indices)

    # Shuffle the combined indices
    sampled_indices = torch.tensor(sampled_indices)
    permuted_indices = torch.randperm(len(sampled_indices))
    sampled_indices = sampled_indices[permuted_indices].tolist()

    # Create a Subset dataset using the combined indices
    balanced_dataset = Subset(original_dataset, sampled_indices)
    return balanced_dataset


def binary_class_pattern_with_noise(n, num_class, noisy_d, percent_correct=1.0, noisy_dim_scalar=1.0):
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
