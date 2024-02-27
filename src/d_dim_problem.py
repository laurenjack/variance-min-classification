import itertools
import math
from itertools import combinations
import numpy as np

from src import helpers


def gen(n, d, c=3):
    return np.random.uniform(low=-c, high=c, siz=(n, d))


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
    return len(plane_counter)


def _add_sub_indices(indices, plane_counter):
    d = len(indices)
    for k in range(1, d):
        for sub_indices in itertools.combinations(indices, k):
            index_key = helpers.index_key(sub_indices)
            plane_counter.add(index_key)



def count_segmentations(x):
    n, d = x.shape
    point_indices = [i for i in range(n)]
    b = np.ones(d)
    plane_counter = set()
    big_plan_counter = set()
    max_in_plane = 0
    min_out_of_plane = 100000
    edges = 0
    for plane_indices in combinations(point_indices, d):
        plane_indices = list(plane_indices)
        plane_points = x[plane_indices]
        # plane, _, _, _ = np.linalg.lstsq(plane_points, b, rcond=None)
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
                min_out_of_plane = abs(colour[i]) if abs(colour[i]) < min_out_of_plane else min_out_of_plane
            else:
                max_in_plane = abs(colour[i]) if abs(colour[i]) > max_in_plane else max_in_plane
        # if max_in_plane > min_out_of_plane:
        #     error_count += 1
        # assert len(blue_indices) + len(red_indices) + d == n
        if len(red_indices) == 0 or len(blue_indices) == 0:
            edges += 1
            for k in range(1, d):
                for sub_indices in itertools.combinations(plane_indices, k):
                    index_key = helpers.index_key(sub_indices)
                    plane_counter.add(index_key)
        plane_counter.add(helpers.index_key(red_indices))
        big_plan_counter.add(helpers.index_key(red_indices))
        plane_counter.add(helpers.index_key(blue_indices))
        big_plan_counter.add(helpers.index_key(blue_indices))
        red_indices.extend(plane_indices)
        blue_indices.extend(plane_indices)
        plane_counter.add(helpers.index_key(red_indices))
        big_plan_counter.add(helpers.index_key(red_indices))
        plane_counter.add(helpers.index_key(blue_indices))
        big_plan_counter.add(helpers.index_key(blue_indices))
    return len(plane_counter), len(big_plan_counter), min_out_of_plane, max_in_plane, edges


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





# n, d = 20, 10
# x = np.random.randn(n, d)
# print(math.comb(n, d))
# print(count_segmentations(x))
# print(analytic_segmentations(n, d))
# print(np.finfo(np.float32))
# print(np.finfo(np.float64))
# print(np.finfo(np.longdouble))
# print(np.finfo(np.float128))
