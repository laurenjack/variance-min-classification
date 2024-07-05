import torch
import numpy as np
from scipy.stats import binom, norm

from src import hyper_parameters

from src import dataset_creator
from src.posterior_minimizer import runner

torch.manual_seed(21342)

runs = 100
n = 100
n_test = 100
percent_correct = 0.8
real_d = 1
noisy_d = 50
d = real_d + noisy_d
sizes = [d, 1]
L = len(sizes) - 1

desired_success_rate = 0.728
bps = [(1 - desired_success_rate ** (1/(dl * L))) / 2 for dl in sizes[:-1]]
prop_product = percent_correct * (1 - percent_correct)
var = prop_product / n
sd = var ** 0.5
# zs = [norm.ppf(1 - bp) for bp in bps]
# post_constant = (prop_product / (n / z ** 2 + 1)) ** 0.5
post_constants = [norm.ppf(1 - bp, scale=sd) for bp in bps]
# a = n / z ** 2 + 1
# b = 1 - 2 * percent_correct # 2 * percent_correct - 1
# c = -percent_correct * (1 - percent_correct)
# roots = np.roots([a, b, c])
# print(roots)
# post_constant = max(roots)
print(f"Regularization Constant: {post_constants}")
# print(f"Checker: {z * sd}")

hp = hyper_parameters.HyperParameters(batch_size=n,
                                      epochs=300,
                                      learning_rate= 0.1,
                                      momentum=0.0,
                                      weight_decay=0.0,
                                      post_constants=post_constants,
                                      gamma=0.8,
                                      is_adam=True,
                                      all_linear=True,
                                      print_epoch=False,
                                      print_batch=False)

problem = dataset_creator.BinaryRandomAssigned(2, real_d, noisy_d=noisy_d, scale_by_root_d=False)
runner.run(problem, runs, n, n_test, sizes, hp, 1, percent_correct=percent_correct, shuffle=False)

