from scipy.stats import binom

n = 100
n_test = 100
d = 1
desired_success_rate = 0.8
bp = (1 - desired_success_rate) / 2
success = binom.ppf(1 - bp, n, 0.5)
print(success)
