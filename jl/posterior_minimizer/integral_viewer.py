import torch
import numpy as np
from scipy import integrate, optimize
import matplotlib.pyplot as plt

from jl.posterior_minimizer import dataset_creator


def p(x, w):
    return 1.0 / (1 + np.exp(-np.matmul(x, w)))


def mle(x, y, w):
    a = p(x, w)
    bernoulli = a ** y * (1 - a) ** (1 - y)
    return np.prod(bernoulli)

def negative_mle(w, x, y):
    return -mle(x, y, w)


def optimum(x, y):
    d = x.shape[1]
    initial_w = [0.0] * d
    return optimize.minimize(negative_mle, initial_w, args=(x, y)).x


def integral(x, y, bound):
    d = x.shape[1]
    w = np.zeros(d)
    bounds = [(-bound, bound)] * d

    def mle_to_integrate(*ws):
        for i, wi in enumerate(ws):
            w[i] = wi
        return mle(x, y, w)

    return integrate.nquad(mle_to_integrate, bounds)[0]


if __name__ == '__main__':
    torch.manual_seed(8564376)
    n = 10
    num_class = 2
    real_d = 1
    noisy_d = 1
    percent_correct = 0.8
    bound = 10
    bra = dataset_creator.BinaryRandomAssigned(num_class, real_d, noisy_d=noisy_d)
    x, y, _ = bra.generate_dataset(n, percent_correct=percent_correct, shuffle=True)
    # x[0, 1] = 1.0
    print('Class 0')
    print(x[(1 - y).bool()])
    print('Class 1')
    print(x[y.bool()])
    x = x.numpy()
    y = y.numpy()
    optimal_w = np.array(optimum(x, y))
    print(optimal_w)
    print('Integral')
    print(integral(x, y, bound))

    plt.figure(figsize=(10, 6))

    w_points = 100
    w0s = [-bound + i * bound / w_points for i in range(w_points+1)]
    w0s = [w0 for w0 in w0s if w0 != 0]
    ws = [np.array([wi, 0]) for wi in w0s]
    # ws_normed = [1 * w / np.linalg.norm(w) for w in ws]
    # w0s_normed = [w[0] for w in ws_normed]
    cs = [1, 2]
    for c in cs:
        mles = [mle(x, y, c * w) for w in ws]
        w0cs = [c * w0 for w0 in w0s]
        plt.plot(w0s, mles, label=f'c={c}')

    plt.xlabel('weight_0')  # X-axis label
    plt.ylabel('prob')  # Y-axis label
    plt.title('Mle')  # Chart title
    plt.legend()  # Adds a legend to specify which line corresponds to which dataset

    plt.grid(True)  # Adds a grid for easier reading
    plt.tight_layout()  # Adjusts subplot params so that the subplot(s) fits in to the figure area
    plt.show()


    # def log_sigmoid(x):
    #     # For positive x
    #     if x >= 0:
    #         return -np.log(1 + np.exp(-x))
    #     # For negative x
    #     else:
    #         return x - np.log(1 + np.exp(x))
    #
    #
    # def to_min(x):
    #     return -mle(x[0], x[1])