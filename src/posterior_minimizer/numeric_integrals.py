from dataclasses import dataclass

import numpy as np
from scipy import integrate, optimize

def p(x, w):
    return 1.0 / (1 + np.exp(-np.matmul(x, w)))


def mle(x, y, w):
    a = p(x, w)
    bernoulli = a ** y * (1 - a) ** (1 - y)
    return np.prod(bernoulli)

def d_mle_dw(x, y, w):
    a = p(x, w)
    prod = mle(x, y, w)
    da = a ** (1 - y) * (1 - a) ** (y) * prod
    return da.T @ x

def negative_mle(w, x, y):
    return -mle(x, y, w)


def optimum(x, y):
    d = x.shape[1]
    initial_w = [0.0] * d
    return optimize.minimize(negative_mle, initial_w, args=(x, y)).x


def integral(x, y, w, fixed_i=None):
    d = x.shape[1]
    w_integral = np.zeros(d)
    indices_to_set = [i for i in range(d)]
    bounds = [(0, w[i]) for i in range(d)]
    opts = [{'epsabs': 0.0001}] * d
    if fixed_i is not None:
        w_integral[fixed_i] = w[fixed_i]
        del indices_to_set[fixed_i]
        del bounds[fixed_i]
        del opts[fixed_i]

    def mle_to_integrate(*ws):
        for i, wi in zip(indices_to_set, ws):
            w_integral[i] = wi
        return mle(x, y, w_integral)

    return integrate.nquad(mle_to_integrate, bounds, opts=opts)[0]


def d_posterior(x, y, w):
    d = w.shape[0]
    pd = integral(x, y, w)
    if d == 1:
        return mle(x, y, w) / pd
    dw = []
    for i in range(d):
        dwi = integral(x, y, w, fixed_i=i)
        dw.append(dwi)
    dw = np.array(dw)
    return dw / pd


@dataclass
class Fixed:
    w: float
    index: int