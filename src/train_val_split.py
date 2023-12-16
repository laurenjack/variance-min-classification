from collections import deque

import torch
from torch.utils.data import Dataset, Subset


def on_validation_n(dataset: Dataset, val_n: int, is_class_balanced=False, num_class=None, m: int = 1):
    """Given a dataset and the percentage of the data to be allocated to the validation set, produce a training and
       validation set.

    Args:
         dataset - The dataset in question
         val_percentage - The percentage of data in the validation set.
         is_class_balanced - If True, the validation set will be as class balanced as possible, if False the split will
         be uniform random across the whole dataset.
         num_class - Must be set to the number of classes in the dataset, if is_class_balanced=True. When
         is_class_balanced=False this parameter does nothing.
         m - The number of models. If 1, will return a normal Dataset for training. If greater than one, will return
         a MultiModelDataSet (see below)

    Returns:
        A tuple, the training set followed by the validation set.
    """
    total_n = len(dataset)
    assert total_n > val_n
    indices = torch.randperm(total_n, dtype=torch.int64)
    # Do a class balanced subset, if num_class is set.
    if is_class_balanced:
        assert num_class
        val_indices = class_balanced_subset(dataset, val_n, num_class)
        val_indices = torch.tensor(val_indices, dtype=torch.int64)
        mask = torch.isin(indices, val_indices)
        not_in = ~mask
        train_indices = indices[not_in]
    else:
        val_indices = indices[:val_n]
        train_indices = indices[val_n:]
    train_set = Subset(dataset, train_indices)
    validation_set = Subset(dataset, val_indices)
    if m > 1:
        # Use a special data loader for the training set that also returns train_with and train_without
        train_set = MultiModelDataSet(train_set, m)
    return train_set, validation_set


def on_percentage(dataset: Dataset, val_percentage=0.5, is_class_balanced=False, num_class=None, m: int = 1):
    """Given a dataset and the percentage of the data to be allocated to the validation set, produce a training and
       validation set.

    Args:
         dataset - The dataset in question
         val_percentage - The percentage of data in the validation set.
         is_class_balanced - If True, the validation set will be as class balanced as possible, if False the split will
         be uniform random across the whole dataset.
         num_class - Must be set to the number of classes in the dataset, if is_class_balanced=True. When
         is_class_balanced=False this parameter does nothing.
         m - The number of models. If 1, will return a normal Dataset for training. If greater than one, will return
         a MultiModelDataSet (see below)

    Returns:
        A tuple, the training set followed by the validation set.
    """
    total_n = len(dataset)
    val_n = int(round(total_n * val_percentage))
    return on_validation_n(dataset, val_n, num_class, m)


def class_balanced_subset(dataset, sub_n, num_class):
    """
    Take a class balanced subset of dataset, of size n, where num_class is the number of classes in the dataset.

    Assumes that the classes are labeled 0, num_class-1. Note, this function's worst case time complexity is:
    O(num_class * len(dataset)), only if the dataset is highly unbalanced. Assuming balanced, it's O(len(dataset)).

    Returns:
        The indices of the subset, which correspond to the examples selected from the dataset.
    """
    indices = torch.randperm(len(dataset), dtype=torch.int64)
    # Group all the indices of each single class together, inside a deque
    class_to_input = {c: deque() for c in range(num_class)}
    for i in indices:
        i = i.item()
        _, c = dataset[i]
        c = c.item()
        class_to_input[c].append(i)
    # Now add examples from each class until the dataset is full
    sub_count = 0
    c = 0
    sub_indices = []
    while sub_count < sub_n:
        q = class_to_input[c]
        if len(q) > 0:
            sub_indices.append(q.pop())
            sub_count += 1
        c += 1
        c %= num_class  # Loop back around
    return sub_indices


class MultiModelDataSet(Dataset):

    def __init__(self, dataset, m):
        n = len(dataset)
        self.dataset = dataset
        self.train_with, self.train_without = _get_model_indices(n, m)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        foo = self.dataset[index]
        x, y = self.dataset[index]
        return x, y, self.train_with[index], self.train_without[index]


def _get_model_indices(n, m):
    assert m % 2 == 0
    models_per_example = m // 2
    train_with = torch.zeros(n, models_per_example, dtype=torch.int64)
    train_without = torch.zeros(n, models_per_example, dtype=torch.int64)
    for i in range(n):
        random_perm = torch.randperm(m, dtype=torch.int64)
        train_with[i] = random_perm[:models_per_example]
        train_without[i] = random_perm[models_per_example:]
    return train_with, train_without