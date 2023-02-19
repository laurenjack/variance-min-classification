import torch
from torch.utils.data import TensorDataset
from src.splitter import MultModelDataLoader


def test_when_iterate_over_data_loader_then_with_and_without_disjoint_and_consistent():
    x_full = torch.zeros(20, 3)
    y_full = torch.arange(20, dtype=torch.int64)
    dataset = TensorDataset(x_full, y_full)
    data_loader = MultModelDataLoader(dataset, batch_size=10, m=4)
    # Iterate once, check for disjointness, hold the indices per label to check consistency
    consistency_checker = {}
    for x, y, train_with, train_without in data_loader:
        assert train_with.shape == (10, 2)
        assert train_without.shape == (10, 2)
        for i in range(10):
            set_train_with = set(train_with[i])
            set_train_without = set(train_without[i])
            # Assert each contains unique elements
            assert len(set_train_with) == 2
            assert len(set_train_without) == 2
            # Assert the sets are disjoint
            assert not set_train_with.intersection(set_train_without)
            # Assert all elements are 0 <= element < m
            assert torch.all(train_with[i] >= 0) and torch.all(train_with[i] < 4)
            assert torch.all(train_without[i] >= 0) and torch.all(train_without[i] < 4)
            # Now store the mapping between the example and the model indices, to see if they remain consistent on the
            # next iteration
            consistency_checker[y[i].item()] = (train_with[i], train_without[i])

    # Check consistency now
    for _, y, train_with, train_without in data_loader:
        for i in range(10):
            expected_train_with, expected_train_without = consistency_checker[y[i].item()]
            assert (expected_train_with == train_with[i]).all()
            assert (expected_train_without == train_without[i]).all()


if __name__ == '__main__':
    test_when_iterate_over_data_loader_then_with_and_without_disjoint_and_consistent()
    print('Tests passed')


