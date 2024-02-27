import torch


class PatternCounter(object):

    def __init__(self, first_class, num_class):
        self.first_class = first_class
        self.class_map = {c: 0 for c in range(num_class)}

    def percent_max_class(self):
        max_class, max_count = self.max_class()
        total_count = 0
        for c, count in self.class_map.items():
            total_count += count
        return max_count / total_count


    def max_class(self):
        max_class = 0
        max_count = 0
        for c, count in self.class_map.items():
            if count > max_count:
                max_class = c
                max_count = count
        return max_class, max_count


def pattern_key(pattern):
    num_bits = len(pattern)
    bit_value = 2 ** (num_bits - 1)
    total = 0
    for i in range(num_bits):
        if pattern[i] > 0:
            total += bit_value
        bit_value //= 2
    return total


def index_key(index):
    key = 0
    for i in index:
        key += 2 ** i
    return key


def fill_pattern_counters(dataset, true_input_bits, num_class):
    n = len(dataset)
    x, y = dataset[0:n]
    pattern_counters = {}
    for i in range(n):
        p = pattern_key(x[i, 0:true_input_bits])
        c = y[i].item()
        if p not in pattern_counters:
            pattern_counters[p] = PatternCounter(first_class=c, num_class=num_class)
        pattern_counter = pattern_counters[p]
        pattern_counter.class_map[c] += 1
    return pattern_counters


def report_patternwise_class_balance(dataset, true_input_bits, num_class):
    pattern_counters = fill_pattern_counters(dataset, true_input_bits, num_class)
    for p, pattern_counter in pattern_counters.items():
        print(f'{p}:')
        for c, count in pattern_counter.class_map.items():
            print(f'    {c}: {count}')


def report_patternwise_accurarices(dataset, true_input_bits, num_class):
    pattern_counters = fill_pattern_counters(dataset, true_input_bits, num_class)
    # The max class will be the true class, unless the percent_correct is very low (near chance)
    # pattern_true_class = {p: counter.max_class for p, counter in pattern_counters.items()}
    for p in range(2 ** true_input_bits):
        percent_correct = pattern_counters[p].percent_max_class()
        print(f'{p}: {percent_correct}')


def calc_dimension_purity(tensor_dataset):
    x, y = tensor_dataset.tensors
    y = y.bool()
    class1 = x[y]
    class0= x[~y]
    class_sum1 = torch.sum(class1, axis=0)
    class_sum0 = torch.sum(-class0, axis=0)
    return torch.abs(class_sum1 + class_sum0) / y.shape[0]




