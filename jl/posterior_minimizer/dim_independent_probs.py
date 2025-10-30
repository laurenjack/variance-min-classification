import numpy as np
from scipy.stats import binom


def denominator(y, p):
    left = prob_sum_across_direction(y, p)
    right = prob_sum_across_direction(y, p)
    return left + right


def prob_sum_across_direction(y, p):
    n = y.shape[0]
    y_after = np.sum(y, axis=0)
    current_success = y_after
    max_success = current_success.copy()
    denom = 0.0
    # Modify from 0 and 1 labels to -1 and 1 to make the calculation simpler
    y = y * 2 - 1
    for i in range(n):
        current_success -= y[i]
        prob = binom.pmf(n, current_success, p)
        denom += prob
    return max_success
