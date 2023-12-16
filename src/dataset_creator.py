from typing import List
from dataclasses import dataclass
import math
from collections import deque
import numpy as np
import torch
from torch import Tensor
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader, SubsetRandomSampler, TensorDataset, Subset, Dataset


CIFAR_NUM_CLASSES = 10


# @dataclass
# class DatasetProperties:
#
#     dataset: Dataset
#     input_shape: List[int]


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


class BinaryRandomAssigned():

    def __init__(self, num_class: int, num_input_bits: int, noisy_d=0, percent_correct=1.0):
        self.num_class = num_class
        all_possible_patterns = get_perms(num_input_bits)
        self.num_patterns = all_possible_patterns.shape[0]
        # Shuffle the patterns so class assignment is random
        self.all_possible_patterns = all_possible_patterns[torch.randperm(self.num_patterns)]

        assert self.num_patterns >= num_class, "Must be at least one pattern per class"
        # Assign each of the possible inputs to a class
        self.classes = [p % num_class for p in range(self.num_patterns)]
        self.percent_correct = percent_correct
        self.noisy_d = noisy_d
        self.num_input = num_input_bits + noisy_d

    def generate_dataset(self, n, shuffle=True):
        all_inputs = []
        all_labels = []
        first_incorrect_index = round(n * self.percent_correct)
        i = 0
        # Choose a random class offset. This is used to change one class into another, i.e. create an incorrect example
        # This amount is added to each class, so that the correctness of each class equals percent_correct
        offset = _random_int(1, self.num_class)
        class_index = 0  # This variable is used to track when the offset should be incremented
        while i < n:
            # We'll build the dataset by adding each pattern 1 at a time, in the same order
            places_left = min(n - i, self.num_patterns)
            for p in range(places_left):
                pattern = self.all_possible_patterns[p]
                c = self.classes[p]
                if i >= first_incorrect_index:  # i > n * percent_correct:
                    # Increment the offset once we've used the same offset on every class
                    if class_index == self.num_class:
                        class_index = 0
                        offset += 1
                        # Rather than using the modulo operator, reset to 1 (a zero offset would not change the class)
                        if offset == self.num_class:
                            offset = 1
                    class_index += 1
                    c = (c + offset) % self.num_class
                if self.noisy_d > 0:
                    noisy_bits = _random_bits((self.noisy_d,))
                    pattern = torch.cat([pattern, noisy_bits])
                all_inputs.append(pattern)
                all_labels.append(c)
                i += 1
        x = torch.stack(all_inputs)
        y = torch.tensor(all_labels)
        if shuffle:
            all_indices = torch.randperm(n)
            x = x[all_indices]
            y = y[all_indices]
        dataset = TensorDataset(x, y)
        return dataset


def binary_class_pattern_with_noise(n, num_class, noisy_d, percent_correct=1.0, noisy_dim_scalar=1.0):
    num_correct = int(round(percent_correct * n))
    class_d = math.ceil(np.log2(num_class))
    class_to_perm = get_perms(class_d)  # np.array  .astype(np.float32)
    # class_to_perm = torch.tensor(class_to_perm)
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
            random_offset = _random_int(1, num_class)
            c = (c + random_offset) % num_class
        noisy_dims = float(noisy_dim_scalar) * _random_bits((noisy_d,))
        example = torch.cat([first_part, noisy_dims])
        x.append(example)
        y.append(c)

    x = torch.stack(x)
    y = torch.tensor(y)
    return TensorDataset(x, y), x.shape


class BalancedDataset(TensorDataset):

    def __init__(self, x: Tensor, y: Tensor, num_patterns: int, first_incorrect_index: int):
        super().__init__(x, y)
        self.x = x
        self.y = y
        self.num_patterns = num_patterns
        self.first_incorrect_index = first_incorrect_index
        self.n = len(self)
        self.num_correct = first_incorrect_index

    def subset(self, n, forbidden_indices=[]):
        assert n < len(self)
        num_correct = int(round(self.num_correct * n / self.n))
        num_incorrect = n - num_correct
        correct_indices = _pattern_balanced_subset(0, num_correct, self.num_patterns, forbidden_indices)
        incorrect_indices = _pattern_balanced_subset(self.first_incorrect_index, num_incorrect, self.num_patterns, forbidden_indices)
        subset_indices = correct_indices + incorrect_indices
        return subset_indices


def _pattern_balanced_subset(starting_index, sub_n, num_patterns, forbidden_indices):
    indices_shuffle = torch.randperm(sub_n, dtype=torch.int64) + starting_index
    # Group all the indices of each single pattern together, inside a deque
    pattern_to_input = {p: deque() for p in range(num_patterns)}
    for i in indices_shuffle:
        i = i.item()
        if i not in forbidden_indices:
            p = i % num_patterns
            pattern_to_input[p].append(i)
    # Now add examples from each pattern until the dataset is full
    sub_count = 0
    p = _random_int(0, num_patterns)
    sub_indices = []
    while sub_count < sub_n:
        q = pattern_to_input[p]
        assert len(q) > 0, "Not enough examples in this dataset for class balance, create more."
        sub_indices.append(q.pop())
        sub_count += 1
        p += 1
        p %= num_patterns  # Loop back around
    return sub_indices







def get_perms(d: int):
    as_list = _get_perms(d, np.ones(d, dtype=np.float32))
    return torch.stack(as_list)


def _get_perms(i, perm):
    if i == 0:
        return [torch.tensor(perm)]
    i -= 1
    left = perm.copy()
    left[i] = -1.0
    left_perms = _get_perms(i, left)
    right = perm.copy()
    right[i] = 1.0
    right_perms = _get_perms(i, right)
    return left_perms + right_perms


def _random_int(low, high):
    """low <= ret < high"""
    return torch.randint(low=low, high=high, size=(1,))[0].item()


def _random_bits(shape):
    return (torch.randint(low=0, high=2, size=shape) * 2 - 1).float()
