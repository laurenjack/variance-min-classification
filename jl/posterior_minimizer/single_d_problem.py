from collections import OrderedDict
import numpy as np
from jl.posterior_minimizer import helpers


class Pattern:

    def __init__(self, y):
        self.y = y
        self.total = 0
        self.position_dict = {}

    def __str__(self):
        sorted_probs = OrderedDict()
        for position, count in sorted(self.position_dict.items()):
            sorted_probs[position] = round(count / self.total, 4)
        return f"{self.y}: {sorted_probs}, {self.total}"


class EmpiricalRun:

    def __init__(self, y_dict):
        self.y_dict = y_dict


def generate(n, p, t):
    blue_successes = np.random.binomial(n, p, size=t)
    red_successes = np.random.binomial(n, p, size=t)
    n_array = n * np.ones(t, dtype=int)
    blue_failures = n_array - blue_successes
    red_failures = n_array - red_successes
    datasets = []
    y_dict = {}
    for i in range(t):
        left = generate_random_draws(red_failures[i], blue_successes[i])
        right = generate_random_draws(red_successes[i], blue_failures[i])
        y = np.concatenate([left, right])
        y_key = helpers.pattern_key(y)
        if y_key not in y_dict:
            y_dict[y_key] = Pattern(y)
        pattern = y_dict[y_key]
        pattern.total += 1
        position_dict = pattern.position_dict
        position = left.shape[0]
        if position not in position_dict:
            position_dict[position] = 0
        position_dict[position] += 1
        datasets.append(y)
    return y_dict


def generate_random_draws(r, b):
    # Create an array with r ones and b zeros
    y = np.array([1] * r + [0] * b)
    # Shuffle the array to randomize the order
    np.random.shuffle(y)
    return y


y_dict = generate(3, 5/6, 1000000)
pattern_keys = list(y_dict.keys())
pattern_keys.sort()

for y_key in pattern_keys:
    print(y_dict[y_key])





