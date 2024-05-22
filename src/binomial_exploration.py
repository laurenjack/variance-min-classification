import numpy as np
from scipy.stats import binom
from scipy.special import comb


def analytic_p_single_dim(n, p):
    prob_X = np.zeros(shape=n)
    max_cuts = n - 1
    p_prod = p ** max_cuts
    for k in range(n):
        comb()
        prob_X[k]



def single_dim_problem(l, n, p):
    return np.less(np.random.rand(n, l), p).astype(int)


def iterate_single_dim(y):
    n = y.shape[0]
    y_after = np.sum(y, axis=0)
    current_success = y_after
    max_success = current_success.copy()
    # Modify from 0 and 1 labels to -1 and 1 to make the calculation simpler
    y = y * 2 - 1
    for i in range(n):
        current_success -= y[i]
        max_success = np.maximum(current_success, max_success)
    return max_success


def expected_success_single_dim(l, n, p):
    return np.mean(iterate_single_dim(single_dim_problem(l, n, p)))




def maximum_x_in_l_tries(l, t, n, p=0.5):
    t_tries = np.random.binomial(n, p, size=(t, l))
    return np.mean(np.max(t_tries, axis=0))


def expected_max_x(t, n, p=0.5):
    p_max_equals_x_dict = get_p_max_equals_x_dict(t, n, p)
    E_max = 0
    for x in range(0, n+1):
        p_max_equals_x = p_max_equals_x_dict[x]
        E_max += p_max_equals_x * x
    return E_max


def prob_p_equals_p(t, n, p, x):
    p_max_equals_x = get_p_max_equals_x_dict(t, n, p)
    return p_max_equals_x[x]


def get_p_max_equals_x_dict(t, n, p=0.5):
    p_max_equals_x_dict = {}
    p_X_gt_or_eq_x = 0.0
    p_max_lt_eq_x = 1.0
    for x in range(n, -1, -1):
        p_X_equals_x = binom.pmf(x, n, p)
        p_X_gt_or_eq_x += p_X_equals_x
        p_max_lt_x = (1.0 - p_X_gt_or_eq_x) ** t
        p_max_equals_x = p_max_lt_eq_x - p_max_lt_x
        p_max_equals_x_dict[x] = p_max_equals_x
        p_max_lt_eq_x = p_max_lt_x
    return p_max_equals_x_dict


if __name__ == '__main__':
    t = 1000
    n = 1000
    p = 0.5
    print(expected_max_x(t, n, p))
    # print(prob_p_equals_p(t, n, 0.5, 3))
    # print(prob_p_equals_p(t, n, 0.6, 3))
    # print(maximum_x_in_l_tries(10000000, t, n, p))
    # print(expected_success_single_dim(10000000, n, p))