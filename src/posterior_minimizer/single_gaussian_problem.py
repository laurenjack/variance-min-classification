import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from scipy.stats import binom, norm

from src import hyper_parameters
from src.posterior_minimizer import regularizer as reg

# torch.manual_seed(3955) # 3917

from src import dataset_creator, train
from src.posterior_minimizer import weight_tracker as wt, runner


n = 100
n_test = 10
percent_correct = 0.5
d = 100

dp = hyper_parameters.DataParameters(percent_correct, n, n_test, d)

hp = hyper_parameters.HyperParameters(batch_size=n // 4,
                                      epochs=250,
                                      learning_rate=0.1,  # 1.0 / d ** 0.5,
                                      momentum=0.9,
                                      weight_decay=0.00,
                                      desired_success_rate=0.5,
                                      sizes=[d, 50, 20, 1],  # 40, 30,
                                      do_train=True,
                                      gamma=1.0,
                                      is_adam=False,
                                      all_linear=False,
                                      reg_type="L1",  # InverseMagnitudeL2
                                      var_type="Analytical",
                                      reg_epsilon=0.0,
                                      print_epoch=False,
                                      print_batch=False)


problem = dataset_creator.Gaussian(d=d, perfect_class_balance=False)
# trainer = train.DirectMeanTrainer()
trainer = train.SigmoidBxeTrainer()

weight_tracker = wt.AllWeights(len(hp.sizes) - 1, node_limit=10)
max_grad_before, max_grad_after, zero_state, preds = runner.single_run(problem, dp, hp, trainer, weight_tracker=weight_tracker, print_details=False, shuffle=True)
print(max_grad_before)
print(max_grad_after)
print(zero_state)
print(preds)

weight_tracker.show()