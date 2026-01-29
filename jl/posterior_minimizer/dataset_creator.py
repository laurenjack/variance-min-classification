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
        return x, y, None


class HyperXorNormal:
    """
    XOR classification on a hypercube with Gaussian noise.

    Creates 2^true_d corners of a true_d-dimensional hypercube as class centers.
    The label is determined by the parity of each corner's bits (0 = even, 1 = odd).
    Points are sampled by picking a corner uniformly, adding N(0, I) noise, then
    downscaling.

    The parameter 'percent_correct' sets the Bayes-optimal XOR classification
    accuracy (probability of predicting correct parity from a noisy observation).

    The parameter 'noisy_d' appends extra dimensions with zero-mean (no signal),
    making total dimensionality d = true_d + noisy_d.

    The parameter 'random_basis' applies a fixed random orthonormal transformation
    across all d dimensions, mixing signal and noise dimensions.

    Derivation: The number of dimensions where noise causes a "crossing" (point
    closer to adjacent corner) follows Binomial(true_d, p) where p = 1 - Φ(z).
    XOR is correct iff an even number of crossings occur:
        P(correct XOR) = (1 + (1-2p)^true_d) / 2
    Solving for p:
        p = (1 - (2*percent_correct - 1)^(1/true_d)) / 2
    """

    def __init__(
        self,
        true_d: int,
        percent_correct: float,
        noisy_d: int = 0,
        random_basis: bool = False,
    ):
        """
        Args:
            true_d: dimension of the underlying hypercube
            percent_correct: Bayes-optimal XOR classification accuracy (must be >= 0.5)
            noisy_d: number of extra noise-only dimensions to append
            random_basis: if True, apply a single random orthonormal rotation
                          across all d = true_d + noisy_d dimensions
        """
        # Validate percent_correct
        if not 0.5 <= percent_correct <= 1.0:
            raise ValueError("percent_correct must be between 0.5 and 1.0 for XOR.")

        # Compute per-dimension crossing probability p from desired XOR accuracy
        # P(correct XOR) = (1 + (1-2p)^true_d) / 2 = percent_correct
        # Solving: p = (1 - (2*percent_correct - 1)^(1/true_d)) / 2
        xor_term = (2.0 * percent_correct - 1.0) ** (1.0 / true_d)
        p = (1.0 - xor_term) / 2.0

        # Compute z such that p = 1 - Φ(z), i.e., z = Φ^(-1)(1 - p)
        self.z = norm.ppf(1.0 - p)
        c = math.sqrt(true_d) * self.z

        # Build the 2^true_d corner means in {+1,-1}^true_d
        num_corners = 1 << true_d
        # corners_true: (2^true_d, true_d)
        corners_true = torch.tensor(
            [
                [(1.0 if (i >> bit) & 1 else -1.0) for bit in range(true_d)]
                for i in range(num_corners)
            ], dtype=torch.float32
        )
        # Scale to length c in true_d
        corners_true = corners_true * (c / math.sqrt(true_d))

        # Append zero-valued noisy dims
        if noisy_d > 0:
            zeros = torch.zeros(num_corners, noisy_d)
            corners = torch.cat([corners_true, zeros], dim=1)
        else:
            corners = corners_true

        # Compute parity labels (0 or 1)
        bits = (corners_true > 0).int()
        parity = bits.sum(dim=1) % 2
        self.labels = parity

        # Total dimensionality
        self.d = true_d + noisy_d

        # Apply a single random orthonormal basis change across all dims
        if random_basis:
            A = torch.randn(self.d, self.d)
            Q, _ = torch.linalg.qr(A)
            # Ensure right-handed coordinate system (det(Q)=+1)
            if torch.det(Q) < 0:
                Q[:, 0] *= -1
            # corners = corners @ Q
            self.basis = Q
        else:
            self.basis = None

        self.means = corners
        self.c = c

    def generate_dataset(self, n: int, clean_mode: bool = False, shuffle: bool = True):
        """
        Sample 'n' data points from the hypercube corners plus Gaussian noise.

        Each point is generated by picking a corner uniformly at random, adding
        N(0, I) noise, then downscaling and optionally rotating.

        Args:
            n: Number of samples to generate.
            clean_mode: If True, use rejection sampling to only keep points that
                        are closest to their true generating corner. This yields
                        a dataset with 100% Bayes-optimal accuracy (no label noise).
                        If False, points may be closer to a different corner due
                        to noise, matching the percent_correct accuracy on average.
            shuffle: If True, randomly permute the resulting dataset.

        Returns:
            (x, y, center_indices):
              - x: shape (n, d) features (d = true_d + noisy_d)
              - y: shape (n,) labels (0 or 1, based on parity)
              - center_indices: shape (n,) corner index for each sample
        """
        num_corners = self.means.size(0)
        d = self.means.size(1)

        if clean_mode:
            # Rejection sampling: only keep points closest to their true corner
            collected_x = []
            collected_y = []
            collected_center_indices = []
            batch_size = max(n, 1000)

            while sum(t.size(0) for t in collected_x) < n:
                # Generate a batch
                idx = torch.randint(low=0, high=num_corners, size=(batch_size,))
                chosen_means = self.means[idx]
                noise = torch.randn(batch_size, d)
                x_batch = chosen_means + noise

                # Check which corner each point is closest to
                # x_batch: (batch_size, d), self.means: (num_corners, d)
                distances = torch.cdist(x_batch, self.means)
                nearest_corners = distances.argmin(dim=1)

                # Keep only points where nearest corner matches sampled corner
                clean_mask = (nearest_corners == idx)
                clean_x = x_batch[clean_mask]
                clean_idx = idx[clean_mask]

                collected_x.append(clean_x)
                collected_y.append(self.labels[clean_idx])
                collected_center_indices.append(clean_idx)

            # Concatenate and truncate to exactly n
            x = torch.cat(collected_x, dim=0)[:n]
            y = torch.cat(collected_y, dim=0)[:n]
            center_indices = torch.cat(collected_center_indices, dim=0)[:n]

            # Downscale
            x /= (2 * self.z)

            # Rotate
            if self.basis is not None:
                x = x @ self.basis
        else:
            # Original sampling (may include points that cross to other corners)
            # 1) Pick random corner indices
            idx = torch.randint(low=0, high=num_corners, size=(n,))

            # 2) Gather the chosen means
            chosen_means = self.means[idx]

            # 3) Add standard normal noise
            noise = torch.randn(n, d)
            x = chosen_means + noise

            # 4) Downscale
            x /= (2 * self.z)

            # 5) Rotate
            if self.basis is not None:
                x = x @ self.basis

            # 6) Labels and center indices
            y = self.labels[idx]
            center_indices = idx

        # Shuffle
        if shuffle:
            perm = torch.randperm(n)
            x = x[perm]
            y = y[perm]
            center_indices = center_indices[perm]

        return x, y, center_indices




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
        return x, y, None

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
    return x, y, None


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
