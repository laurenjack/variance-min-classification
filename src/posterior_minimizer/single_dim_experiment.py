import torch
from scipy.stats import binom, norm

from src import hyper_parameters

from src import dataset_creator
from src.posterior_minimizer import runner

torch.manual_seed(56433)

runs = 100
n = 100
n_test = 100
d = 1
desired_success_rate = 0.728
bp = (1 - desired_success_rate ** (1/d)) / 2
print(bp)
threshold_success = binom.ppf(1 - bp, n, 0.5) # - 1
deviation = threshold_success - n * 0.5
print(deviation)
# post_constant = threshold_success / n - 0.5
post_constant = norm.ppf(1 - bp, scale=0.5/n ** 0.5)
print(f"Regularization Constant: {post_constant}")

hp = hyper_parameters.HyperParameters(batch_size=n,
                                      epochs=300,
                                      learning_rate= 0.1,
                                      momentum=0.0,
                                      weight_decay=0.0,
                                      post_constant=post_constant,
                                      gamma=0.8,
                                      single_moving=None,
                                      is_adam=True,
                                      all_linear=True,
                                      print_epoch=False,
                                      print_batch=False)

problem = dataset_creator.AllNoise(num_class=2, d=d)
runner.run(problem, runs, n, n_test, hp, 0, deviation, shuffle=True)

