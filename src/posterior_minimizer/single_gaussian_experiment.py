import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from scipy.stats import binom, norm

from src import hyper_parameters
from src.posterior_minimizer import regularizer as reg

from src import dataset_creator, train
from src.posterior_minimizer import weight_tracker as wt, runner

# torch.manual_seed(5498)  # 392841 769
# torch.manual_seed(7612984)


runs = 100
n = 100
n_test = 10
percent_correct = 0.5
d = 50

dp = hyper_parameters.DataParameters(percent_correct, n, n_test, d)

hp = hyper_parameters.HyperParameters(batch_size=n,
                                      epochs=600,
                                      learning_rate=0.1,  # 1.0 / d ** 0.5,
                                      momentum=0.9,
                                      weight_decay=0.0,
                                      desired_success_rate=0.5,
                                      relu_bound=0.5,
                                      sizes=[d, 30, 1],  # 40, 30,
                                      do_train=True,
                                      gamma=1.0,
                                      is_adam=False,
                                      all_linear=True,
                                      is_bias=False,
                                      reg_type="L1",
                                      implementation='new',
                                      reg_epsilon=0.0,
                                      print_epoch=False,
                                      print_batch=False)



problem = dataset_creator.Gaussian(d=d, perfect_class_balance=False)
# trainer = train.DirectMeanTrainer()
trainer = train.SigmoidBxeTrainer()

runner.run(problem, runs, dp, hp, trainer, shuffle=True)

