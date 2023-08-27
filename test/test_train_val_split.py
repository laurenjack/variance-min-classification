import torch
from torch.utils.data import TensorDataset, DataLoader

from src import train_val_split


def test_class_balanced_sample(dataset):
    indices = train_val_split.class_balanced_subset(dataset, 6,  3)
    sub_y = y[indices]
    _assert_class_balanced(sub_y)


def test_train_test_split_with_class_balance(dataset):
    train, validation = train_val_split.on_validation_n(dataset, 6, is_class_balanced=True, num_class=3)
    sub_x_val, sub_y_val = _extract_tensors(validation)
    sub_x_train, _ = _extract_tensors(train)
    _assert_class_balanced(sub_y_val)
    val_set = set(sub_x_val.tolist())
    train_set = set(sub_x_train.tolist())
    assert len(val_set.intersection(train_set)) == 0


def _assert_class_balanced(sub_y):
    assert (sub_y == 0).sum().item() == 2
    assert (sub_y == 1).sum().item() == 2
    assert (sub_y == 2).sum().item() == 2


def _extract_tensors(dataset):
    return next(iter(DataLoader(dataset, batch_size=len(dataset))))


if __name__ == '__main__':
    torch.manual_seed(5726311)
    x = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=torch.int64)
    y = torch.tensor([1, 0, 0, 1, 0, 2, 0, 0, 2, 0])
    dataset = TensorDataset(x, y)
    test_class_balanced_sample(dataset)
    test_train_test_split_with_class_balance(dataset)
