import torch
from torch.utils.data import DataLoader, SubsetRandomSampler, TensorDataset

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
    train_loader = MultModelDataLoader(dataset, batch_size, m, train_indices)
    val_loader = DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(val_indices))
    return train_loader, val_loader


class MultiModelIterator(object):

    def __init__(self, index_iterator, dataset, train_with, train_without):
        self.index_iterator = index_iterator
        self.dataset = dataset
        self.train_with = train_with
        self.train_without = train_without

    def __next__(self):
        batch_indices = next(self.index_iterator)
        x, y = self.dataset[batch_indices]
        train_with = self.train_with[batch_indices]
        train_without = self.train_without[batch_indices]
        return x, y, train_with, train_without
class MultModelDataLoader(object):

    def __init__(self, dataset, batch_size, m, subset_indices=None):
        self.dataset = dataset
        n = len(dataset)
        # if subset_indices is set, some of these will never be used, but that is totally OK
        self.train_with, self.train_without = _get_model_indices(n, m)
        if subset_indices is None:
            self.indices = torch.arange(n, dtype=torch.int64)
        else:
            self.indices = subset_indices
        self.index_loader = DataLoader(TensorDataset(self.indices), batch_size, shuffle=True)

    def __iter__(self):
        index_iterator = self.index_loader.__iter__()
        return MultiModelIterator(index_iterator, self.dataset, self.train_with, self.train_without)

    def num_input(self):
        return self.dataset[0][0].shape[0]


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



