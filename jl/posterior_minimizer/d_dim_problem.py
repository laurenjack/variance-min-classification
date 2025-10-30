import itertools
import math
from itertools import combinations
import numpy as np
from scipy import special

from jl.posterior_minimizer import helpers


def gen(n, d, c=3):
    return np.random.uniform(low=-c, high=c, siz=(n, d))


def find_segmentations_dual(x):
    n, d = x.shape
    point_indices = [i for i in range(n)]
    b = np.ones(d)
    plane_counter = set()
    for plane_indices in combinations(point_indices, d):
        plane_indices = list(plane_indices)
        plane_points = x[plane_indices]
        inv = np.linalg.inv(plane_points)
        plane = inv.dot(-b)
        red_indices = []
        colour = x.dot(plane) + 1.0
        for i in range(n):
            if i not in plane_indices:
                if colour[i] > 0:
                    red_indices.append(i)
        for k in range(0, d+1):
            for sub_indices in itertools.combinations(plane_indices, k):
                sub_indices = list(sub_indices)
                new_red = red_indices + sub_indices
                plane_counter.add(helpers.dual_index_key(new_red, n))
    return plane_counter


def count_segmentations_new(x):
    n, d = x.shape
    point_indices = [i for i in range(n)]
    b = np.ones(d)
    plane_counter = set()
    for plane_indices in combinations(point_indices, d):
        plane_indices = list(plane_indices)
        plane_points = x[plane_indices]
        inv = np.linalg.inv(plane_points)
        plane = inv.dot(-b)
        red_indices = []
        blue_indices = []
        colour = x.dot(plane) + 1.0
        for i in range(n):
            if i not in plane_indices:
                if colour[i] > 0:
                    red_indices.append(i)
                else:
                    blue_indices.append(i)
        for k in range(0, d+1):
            for sub_indices in itertools.combinations(plane_indices, k):
                sub_indices = list(sub_indices)
                new_red = red_indices + sub_indices
                other_indices = [i for i in plane_indices if i not in sub_indices]
                new_blue = blue_indices + other_indices
                plane_counter.add(helpers.index_key(new_red))
                plane_counter.add(helpers.index_key(new_blue))
    return plane_counter


def most_likely_plane(segmentations, y):
    n = y.shape[0]
    normalizer = 0.0
    integrated_noramlizer = 0.0
    max_mle = 0.0
    max_mle_reg = 0.0
    reg_normalizer = 0.0
    max_correct = 0
    segs = 0
    for key in segmentations:
        segmentation_index = helpers.from_index(key)
        predictions = np.zeros(n, dtype=np.int64)
        predictions[segmentation_index] = 1
        is_correct = predictions == y
        n_correct = is_correct.sum()
        p_max = n_correct / n
        mle = p_max ** n_correct * (1 - p_max) ** (n - n_correct)
        p_max = (n_correct + 2) / (n + 4)
        mle_reg = p_max ** (n_correct + 2) * (1 - p_max) ** (n - n_correct + 2)
        max_mle_reg = max(max_mle_reg, mle_reg)
        max_mle = max(mle, max_mle)
        max_correct = max(max_correct, n_correct, n - n_correct)
        normalizer += mle
        reg_normalizer += mle_reg
        segs += 1
        integrated_noramlizer += special.beta(n_correct + 3, n - n_correct + 3)
    t_zero = 1 / segs
    return max_correct, max_mle, max_mle / normalizer, max_mle_reg / reg_normalizer, max_mle_reg / integrated_noramlizer, t_zero


def iterate_problem(x, y):
    n, d = x.shape
    for k in range(1, d+1):
        # Take k dimensions
        x_sub = x[:,:k]
        print(f'Dim: {k}')
        segmentations = find_segmentations_dual(x_sub)
        max_correct, mle, max_posterior, post1, post2, t_zero = most_likely_plane(segmentations, y)
        print(f'    Max correct: {max_correct} MLE: {mle} Posteriors: {max_posterior, post1, post2}, t zero: {t_zero}')


def analytic_segmentations(n, d):
    mid = (n + 1) // 2
    count = 0
    for k in range(mid):
        count += math.comb(n, min(k, d - 1))
    count *= 2
    if n % 2 == 0:
        count += math.comb(n, min(mid, d - 1))
    return count


def new_bound(n, d):
    if d == 0:
        return 1
    return math.comb(n, d) + new_bound(n, d - 1)


n = 10
d = 4
x = np.random.randn(n, d)
# x = np.arange(n) - (n - 1) / 2
# x /= n
half_zeros = np.zeros(n // 2, dtype=np.int64)
half_ones = np.ones(n // 2, dtype=np.int64)
y = np.concatenate((half_zeros, half_ones))
np.random.shuffle(y)
iterate_problem(x, y)
# x = np.random.randn(n)
# print(integral(x, y))





