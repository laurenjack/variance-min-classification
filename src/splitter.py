import torch
from torch.utils.data import Dataset, DataLoader, Subset


def train_val_split(dataset, batch_size, m, val_percentage=0.5):
    """Take a Dataset and return:

    1) A MultiModelDataLoader - over the training set
    2) A DataLoader - over the validation set
    """
    n_total = len(dataset)
    n_train = int(round(n_total * (1 - val_percentage)))
    indices = torch.randperm(len(dataset), dtype=torch.int64)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]
    train_set = Subset(dataset, train_indices)
    validation_set = Subset(dataset, val_indices)
    # Use a special data loader for the training set that also returns train_with and train_without
    multi_train_set = MultiModelDataSet(train_set, m)
    train_loader = DataLoader(multi_train_set, batch_size=batch_size)
    val_loader = DataLoader(validation_set, batch_size=batch_size)
    return train_loader, val_loader


class MultiModelDataSet(Dataset):

    def __init__(self, dataset, m):
        n = len(dataset)
        self.dataset = dataset
        self.train_with, self.train_without = _get_model_indices(n, m)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
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