import numpy as np


class MeanUniform(object):

    def __init__(self, T):
        self.T = T

    def analytic(self):
        mean_prob = 0.0
        for t in range(self.T):
            max_mean = (1 - mean_prob) * (1 + mean_prob) / 2
            existing_mean = mean_prob * mean_prob
            mean_prob = max_mean + existing_mean
        return mean_prob

    def computational(self, n):
        x = np.random.rand(n, self.T)
        mean_prob = np.mean(np.max(x, axis=1))
        return mean_prob


mu = MeanUniform(4)
print(mu.analytic())
print(mu.computational(100000))
