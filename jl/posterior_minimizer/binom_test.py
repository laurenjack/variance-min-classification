import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from scipy.stats import binom, norm

from jl.posterior_minimizer import hyper_parameters
from jl.posterior_minimizer import custom_modules as cm
from jl.posterior_minimizer.train import SigmoidBxeTrainer, BoundsAsParam
from jl.posterior_minimizer.regularizer import DirectReg, L1

from jl.posterior_minimizer import dataset_creator
from jl.posterior_minimizer import weight_tracker as wt

# torch.manual_seed(769)  # 392841 769

n = 100
n_test = 100
d = 1
desired_success_rate = 0.728
bp = (1 - desired_success_rate ** (1/d)) / 2
print(bp)
threshold_success = binom.ppf(1 - bp, n, 0.5) # - 1
post_constant = threshold_success / n - 0.5
print(f"Regularization Constant: {post_constant}")

x_normal = norm.ppf(1 - bp, scale=0.5/n ** 0.5)
print(x_normal)
