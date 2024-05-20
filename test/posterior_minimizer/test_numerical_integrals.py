import math

import numpy as np
from scipy import integrate
from src.posterior_minimizer import numeric_integrals

def test_single_dim():
    def sigmoid(w):
        return 1 / (1 + np.exp(-w)) ** 2 * (1 - 1 / (1 + np.exp(-w))) * (1 - 1 / (1 + np.exp(w))) ** 2 * (1 / (1 + np.exp(w)))



    pc = sigmoid(2)
    pd = integrate.quad(sigmoid, 0, 2)[0]
    x = np.array([1.0, 1.0, -1.0, -1.0, -1.0, 1.0]).reshape([6, 1])
    y = np.array([1, 1, 1, 0, 0, 0])
    result = numeric_integrals.d_posterior(x, y, np.array([2]))
    expected = pc / pd
    assert math.isclose(expected, result)

    def sigmoid(w0, w1):
        ones = 1 / (1 + np.exp(-w0 + 0.5 * w1)) * 1 / (1 + np.exp(-w0 + -0.5 * w1)) * 1 / (1 + np.exp(w0 + 0.5 * w1))
        zeros = (1 - 1 / (1 + np.exp(w0 - 0.5 * w1))) * (1 - 1 / (1 + np.exp(w0 + 0.5 * w1))) * (1 - 1 / (1 + np.exp(-w0 - 0.5 * w1)))
        return ones * zeros

    w0_fixed = 2.5
    def partial_sig0(w1):
        return sigmoid(w0_fixed, w1)

    w1_fixed = -1.5
    def partial_sig1(w0):
        return sigmoid(w0, w1_fixed)

    pc = np.array([integrate.quad(partial_sig0,-1.5, 0)[0], integrate.quad(partial_sig1,0, 2.5)[0]])
    pd = integrate.nquad(sigmoid, [(0, 2.5), (-1.5, 0)])[0]
    expected = pc / pd
    print(pc)
    print(pd)

    x = np.array([[1.0, -0.5,], [1.0, 0.5], [-1.0, -0.5], [-1.0, 0.5], [-1.0, -0.5], [1.0, 0.5]])
    result = numeric_integrals.d_posterior(x, y, np.array([2.5, -1.5]))
    print(result)
    assert np.allclose(expected, result)


if __name__ == '__main__':
    test_single_dim()
    print('All tests passed!')