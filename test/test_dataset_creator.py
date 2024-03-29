import torch
from src import dataset_creator


def test_binary_random_assigned_large():
    dataset = dataset_creator.BinaryRandomAssigned(4, 8).generate_dataset(1027)
    assert len(dataset) == 1027
    x, y = dataset[0:1027]
    assert (1027, 8) == x.shape
    assert (1027,) == y.shape

    # class_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    # for i in range(1027):
    #     class_counts[y[i].item()] += 1
    # count_257 = 0
    # count_256 = 0
    # for c, count in class_counts.items():
    #     num_c = y[y == c].shape[0]
    #     if num_c == 257:
    #         count_257 += 1
    #     elif num_c == 256:
    #         count_256 += 1
    #     else:
    #         raise AssertionError(f'Unexpected count {num_c} of class {c}')
    # assert count_257 == 3
    # assert count_256 == 1

    # Every input configuration should occur 4 times, except for 3 of them which should occur 5 times,
    # with different classes
    input_trackers = {}
    for i in range(1027):
        input = x[i]
        c = y[i].item()
        input_key = _pattern_key(input)
        if input_key not in input_trackers:
            input_trackers[input_key] = InputTracker(c)
        input_trackers[input_key].add_occurence(c)
    # Now check each input occured the right amount of times
    count_4 = 0
    count_5_classes = []
    for input_tracker in input_trackers.values():
        count_input = input_tracker.count
        if count_input == 4:
            count_4 += 1
        elif count_input == 5:
            count_5_classes.append(input_tracker.c)
        else:
            raise AssertionError(f'An input occurred {count_input} times, expected it to occur 4 or 5 times')
    assert count_4 == 253
    assert len(count_5_classes) == 3


def test_binary_random_assigned_small_and_some_incorrect():
    dataset = dataset_creator.BinaryRandomAssigned(4, 3, percent_correct=0.75).generate_dataset(32)
    assert len(dataset) == 32
    x, y = dataset[0:32]
    assert (32, 3) == x.shape
    assert (32,) == y.shape

    # For the first set of patterns, check they are distinct, and record their class and count
    patterns = {_pattern_key(x[i]): InputTracker(y[i].item(), 1) for i in range(8)}
    assert len(patterns) == 8  # Verifies distinctness
    for i in range(8, 32):
        input_tracker = patterns[_pattern_key(x[i])]
        input_tracker.add_occurence()
        c = input_tracker.c
        if i < 24:
            assert y[i] == c
        else:
            assert y[i] != c

    for pattern_key, input_tracker in patterns.items():
        assert input_tracker.count == 4




def _pattern_key(pattern):
    num_bits = len(pattern)
    bit_value = 2 ** (num_bits - 1)
    total = 0
    for i in range(num_bits):
        if pattern[i] > 0:
            total += bit_value
        bit_value //= 2
    return total

class InputTracker(object):

    def __init__(self, c, count=0):
        self.c = c
        self.count = count

    def add_occurence(self, c=None):
        # Only check c matches if the caller wants to
        if c is not None:
            assert c == self.c, f'The class should always be the same for a given input'
        self.count += 1


def test_when_class_pattern_with_noise_first_class_matches_then_class_doesnt_match():
    dataset, returned_shape = dataset_creator.binary_class_pattern_with_noise(100, 16, 40, percent_correct=0.8, noisy_dim_scalar=3.0)
    x, y = dataset[0:100]
    assert (100, 44) == x.shape == returned_shape
    assert (100,) == y.shape
    # Check each class has the same pattern
    class_pattern = {}
    for i in range(100):
        example = x[i]
        c = y[i].item()
        if c in class_pattern:
            # Class should match, the first 80 elements should be correct because percent correct of 80 was specified
            if i < 80:
                assert torch.allclose(class_pattern[c], example[:4])
            # Now the class pattern should not match, these are all the false examples now
            else:
                assert not torch.allclose(class_pattern[c], example[:4])
        else:
            class_pattern[c] = example[:4]
        # Check the noise is set properly
        torch.allclose(torch.abs(example[4:]), 3.0 * torch.ones(40))


if __name__ == '__main__':
    torch.manual_seed(23721)
    test_binary_random_assigned_small_and_some_incorrect()
    test_binary_random_assigned_large()
    test_when_class_pattern_with_noise_first_class_matches_then_class_doesnt_match()
    print('All tests passed')