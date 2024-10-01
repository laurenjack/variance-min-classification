import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from scipy.stats import binom, norm

from src import hyper_parameters
from src.posterior_minimizer import regularizer as reg

from src import dataset_creator, train
from src.posterior_minimizer import weight_tracker as wt, runner

torch.manual_seed(521943)  # 392841 769

runs = 100
n = 100
n_test = 10
percent_correct = 0.5
d = 10

dp = hyper_parameters.DataParameters(percent_correct, n, n_test, d)

hp = hyper_parameters.HyperParameters(batch_size=n,
                                      epochs=400,
                                      learning_rate= 1.0 / d ** 0.5,
                                      momentum=0.0,
                                      weight_decay=0.0,
                                      desired_success_rate=0.5,
                                      sizes=[d, 1],
                                      gamma=0.85,
                                      is_adam=True,
                                      all_linear=True,
                                      reg_type="GradientWeightedNormed",
                                      reg_epsilon=0.0,
                                      print_epoch=False,
                                      print_batch=False)


problem = dataset_creator.SingleDirectionGaussian(d=d)
trainer = train.DirectMeanTrainer()
# trainer = train.SigmoidBxeTrainer()

runner.run(problem, runs, dp, hp, 0, trainer, shuffle=True)

