import torch
from jl.posterior_minimizer import dataset_creator
from jl.posterior_minimizer import helpers


def test_binary_random_assigned_large():
    x, y, _ = dataset_creator.BinaryRandomAssigned(4, 8).generate_dataset(1027)
    assert (1027, 8) == x.shape
    assert (1027,) == y.shape

    # Every input configuration should occur 4 times, except for 3 of them which should occur 5 times,
    # with different classes
    input_trackers = {}
    for i in range(1027):
        input = x[i]
        c = y[i].item()
        input_key = helpers.pattern_key(input)
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


def test_binary_random_assigned_medium_and_some_incorrect():
    x, y, _ = dataset_creator.BinaryRandomAssigned(4, 4, noisy_d=3).generate_dataset(192, percent_correct=0.5, shuffle=False)
    assert (192, 7) == x.shape
    assert (192,) == y.shape
    # Build up counts of each class for each input.
    pattern_counters = helpers.fill_pattern_counters(x, y, 4, 4)
    assert len(pattern_counters) == 16
    for pattern_counter in pattern_counters.values():
        class_map = pattern_counter.class_map
        true_class = pattern_counter.first_class # The dataset was not shuffled so the first class visited is correct.
        for c in range(4):
            if c == true_class:
                assert class_map[c] == 6
            else:
                assert class_map[c] == 2





def test_binary_random_assigned_small_and_some_incorrect():
    x, y, _ = dataset_creator.BinaryRandomAssigned(4, 3).generate_dataset(32, percent_correct=0.75, shuffle=False)
    assert (32, 3) == x.shape
    assert (32,) == y.shape

    # For the first set of patterns, check they are distinct, and record their class and count
    patterns = {helpers.pattern_key(x[i]): InputTracker(y[i].item(), 1) for i in range(8)}
    assert len(patterns) == 8  # Verifies distinctness
    offset = (y[24] - y[0]) % 4
    assert 0 < offset < 4
    for i in range(8, 32):
        input_tracker = patterns[helpers.pattern_key(x[i])]
        input_tracker.add_occurence()
        c = input_tracker.c
        if i < 24:
            assert y[i] == c
        elif i < 28:
            # The offset should be the same
            assert y[i] == (c + offset) % 4
        else:
            # The offset should have increased
            if offset == 3:
                new_offset = 1
            else:
                new_offset = offset + 1
            assert y[i] == (c + new_offset) % 4


    for pattern_key, input_tracker in patterns.items():
        assert input_tracker.count == 4


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


def test_distinct_inputs_for_features():
    data_gen = dataset_creator.DistinctInputsForFeatures(2, 3, 4, noisy_d=7)
    x, y, _ = data_gen.generate_dataset(10, 9, shuffle=False)
    assert x.shape == (60, 31)
    assert y.shape == (60,)
    first_pattern = [x[0, 0:4], x[1, 4:8], x[2, 8:12], x[3, 12:16], x[4, 16:20], x[5, 20:24]]
    for i in range(54):
        p = i % 6
        for p_anti in range(6):
            does_match = _matches_first(first_pattern, x, i, p_anti)
            if p_anti == p:
                assert does_match
            else:
                assert not does_match
    for i in range(54, 60, 1):
        for p_anti in range(6):
            assert not _matches_first(first_pattern, x, i, p_anti)
    assert torch.equal(torch.tensor([0.0, 1.0]).repeat(30), y)


def _matches_first(first_pattern, x, i, p):
    return x[i, 4*p] == first_pattern[p][0] and x[i, 4*p+1] == first_pattern[p][1] and x[i, 4*p+2] == first_pattern[p][2] and x[i, 4*p+3] == first_pattern[p][3]


if __name__ == '__main__':
    torch.manual_seed(23721)
    test_binary_random_assigned_small_and_some_incorrect()
    test_binary_random_assigned_medium_and_some_incorrect()
    test_binary_random_assigned_large()
    test_when_class_pattern_with_noise_first_class_matches_then_class_doesnt_match()
    test_distinct_inputs_for_features()
    print('All tests passed!')