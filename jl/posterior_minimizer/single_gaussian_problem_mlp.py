import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from scipy.stats import binom, norm

from jl.posterior_minimizer import hyper_parameters
from jl.posterior_minimizer import regularizer as reg

# torch.manual_seed(3965) # 3917

from jl.posterior_minimizer import dataset_creator, train
from jl.posterior_minimizer import weight_tracker as wt, runner



n = 100
n_test = 10
percent_correct = 0.5
d = 2

dp = hyper_parameters.DataParameters(percent_correct, n, n_test, d)

hp = hyper_parameters.HyperParameters(batch_size=n,
                                      epochs=400,
                                      learning_rate=0.01,  # 1.0 / d ** 0.5,
                                      momentum=0.0,
                                      weight_decay=0.001,
                                      desired_success_rate=0.5,
                                      sizes=[d, 4, 4, 1],
                                      do_train=True,
                                      gamma=1.0,
                                      is_adam=True,
                                      all_linear=False,
                                      is_bias=False,
                                      reg_type="NoReg",
                                      weight_tracker_type='Gradient',
                                      reg_epsilon=0.0,
                                      print_epoch=True,
                                      print_batch=False)


problem = dataset_creator.Gaussian(d=d, perfect_class_balance=False)
# trainer = train.DirectMeanTrainer()
trainer = train.SigmoidBxeTrainer()

max_grad_before, max_grad_after, zero_state, preds = runner.single_run(problem, dp, hp, trainer, print_details=False, shuffle=True)
print(max_grad_before)
print(max_grad_after)
print(zero_state)
print(preds)