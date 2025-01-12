import math
import numpy as np
import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import TensorDataset, Subset
from scipy.stats import norm


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


class DistinctInputsForFeatures:

    def __init__(self, num_class: int, patterns_per_class: int, bits_per_pattern: int, noisy_d=0, scale_by_root_d=False):
        self.num_class = num_class
        self.patterns_per_class = patterns_per_class
        self.num_patterns = num_class * patterns_per_class
        self.noisy_d = noisy_d
        all_patterns = get_perms(bits_per_pattern)
        self.patterns = [] # Keeps track of the specific pattern
        self.all_anti_patterns = [] # Keeps track permutation that is not the pattern
        for c in range(num_class):
            for pc in range(patterns_per_class):
                pattern, anti_patterns = _random_row(all_patterns, anti_too=True)
                self.patterns.append(pattern)
                self.all_anti_patterns.append(anti_patterns)
        self.scale_by_root_d = scale_by_root_d

    def generate_dataset(self, n_per_pattern, correct_per_pattern, shuffle=True):
        assert isinstance(n_per_pattern, int)
        assert isinstance(correct_per_pattern, int)
        assert 0 <= correct_per_pattern <= n_per_pattern
        x = []
        y = []
        for i in range(n_per_pattern):
            for p in range(self.num_patterns):
                # Choose a random set of anything but that pattern for all positions
                full_pattern = [_random_row(anti_patterns) for anti_patterns in self.all_anti_patterns]
                # Set the actual pattern, if this instance is supposed to be correct
                if i < correct_per_pattern:
                    full_pattern[p] = self.patterns[p]
                full_pattern = torch.cat(full_pattern)
                # Now add the noisy dimensions
                full_pattern = torch.cat([full_pattern, _random_bits((self.noisy_d,))])
                # Don't use a convnet on this problem coz classes are evenly spaced!
                c = p % self.num_class
                x.append(full_pattern)
                y.append(c)
        return _return_xy(x, y, shuffle, self.scale_by_root_d)


class BinaryRandomAssigned:

    def __init__(self, num_class: int, num_input_bits: int, noisy_d=0, scale_by_root_d=False):
        self.num_class = num_class
        all_possible_patterns = get_perms(num_input_bits)
        self.num_patterns = all_possible_patterns.shape[0]
        # Shuffle the patterns so class assignment is random
        self.all_possible_patterns = all_possible_patterns[torch.randperm(self.num_patterns)]

        assert self.num_patterns >= num_class, "Must be at least one pattern per class"
        # Assign each of the possible inputs to a class
        self.classes = [p % num_class for p in range(self.num_patterns)]
        self.noisy_d = noisy_d
        self.num_input = num_input_bits + noisy_d
        self.scale_by_root_d = scale_by_root_d

    def generate_dataset(self, n, percent_correct=1.0, shuffle=True):
        all_inputs = []
        all_labels = []
        first_incorrect_index = round(n * percent_correct)
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
        return _return_xy(all_inputs, all_labels, shuffle, self.scale_by_root_d)


class AllNoise:

    def __init__(self, num_class, d):
        self.num_class = num_class
        self.d = d

    def generate_dataset(self, n, shuffle=True):
        xs = []
        ys = []
        for i in range(n):
            xs.append(_random_bits((self.d,)))
            c = i % self.num_class
            ys.append(c)
        return _return_xy(x_list=xs, y_list=ys, shuffle=shuffle, scale_by_root_d=False)


class MultivariateNormal:

    def __init__(self, true_d, percent_correct, noisy_d=0):
        v = torch.randn(true_d)
        if noisy_d > 0:
            v = torch.cat([v, torch.zeros(noisy_d)])
        u = v / v.norm(p=2)
        c = norm.ppf(percent_correct)
        self.mew = c * u

    def generate_dataset(self, n, shuffle=True):
        d = self.mew.shape[0]
        y = torch.randint(low=0, high=2, size=(n,))
        ep = torch.randn(n, d)
        y_shift = (2 * y.float() - 1).view(n, 1)
        x = self.mew * y_shift + ep
        return x, y


class Gaussian:

    NUM_CLASS = 2
    STANDARD_DEV = 1.0

    def __init__(self, d, perfect_class_balance=True):
        self.d = d
        self.perfect_class_balance = perfect_class_balance

    def generate_dataset(self, n, shuffle=True):
        if self.perfect_class_balance:
            x, y = self._gen_class_balanced(n)
        else:
            x = self._rand_normal(n)
            y = torch.randint(0, 2, (n,), dtype=torch.int64)
        if shuffle:
            all_indices = torch.randperm(n)
            x = x[all_indices]
            y = y[all_indices]
        return x, y

    def _gen_class_balanced(self, n):
        class_n = n // 2
        x0, y0 = self._gen_single_class(class_n, 0)
        x1, y1 = self._gen_single_class(class_n, 1)
        x = torch.cat((x0, x1), dim=0)
        y = torch.cat((y0, y1))
        return x, y

    def _gen_single_class(self, class_n, target):
        x = self._rand_normal(class_n)
        y = target * torch.ones(class_n, dtype=torch.int64)
        return x, y

    def _rand_normal(self, n):
        return torch.normal(mean=0.0, std=Gaussian.STANDARD_DEV, size=(n, self.d))





# class SingleFeature:
#
#     def __init__(self, d, percent_correct, scale = 2.0):
#         self.num_class = 2
#         self.d = d
#         self.percent_corrrect = percent_correct
#         self.feature_vector = scale * (torch.rand(d) - 0.5) * _random_bits((d))
#
#     def generate_dataset(self, n, shuffle=True):
#         # In this case, we can either have the positive of the feature, or the negative of the feature
#         x = _random_bits((n, 1)) * self.feature_vector.view(1, self.d)
#         num_correct = percent_correct * n
#         for i in range(n):
#             c = i % self.num_class
#
#         y = torch.tensor([i % self.num_class for i in range(n)])
#         if shuffle:
#             all_indices = torch.randperm(n)
#             x = x[all_indices]
#             y = y[all_indices]
#         return x, y








def _return_xy(x_list, y_list, shuffle, scale_by_root_d):
    n = len(x_list)
    x = torch.stack(x_list)
    if scale_by_root_d:
        d = x.shape[1]
        # Scaling by the standard deviation of a binomial
        x /= d ** 0.5
    # x *= 2
    y = torch.tensor(y_list)
    if shuffle:
        all_indices = torch.randperm(n)
        x = x[all_indices]
        y = y[all_indices]
    return x, y


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


def _random_row(tensor, anti_too=False):
    num_rows = tensor.shape[0]
    random_index = torch.randint(0, num_rows, (1,)).item()
    # If this flag is set, return all the rows that are not the randomly selected row, in a second tensor
    if anti_too:
        anti_indices = [i for i in range(num_rows)]
        del anti_indices[random_index]
        return tensor[random_index], tensor[anti_indices]
    return tensor[random_index]


def _random_int(low, high):
    """low <= ret < high"""
    return torch.randint(low=low, high=high, size=(1,))[0].item()


def _random_bits(shape):
    return (torch.randint(low=0, high=2, size=shape) * 2 - 1).float()
