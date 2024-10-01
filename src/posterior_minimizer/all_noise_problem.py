import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from scipy.stats import binom, norm

from src import hyper_parameters
from src.posterior_minimizer import regularizer as reg

from src import dataset_creator, train
from src.posterior_minimizer import weight_tracker as wt, runner

# torch.manual_seed(769)  # 392841 769

n = 100
n_test = 100
percent_correct = 0.5
# noisy_d = 1
# real_d = 1
# d = real_d + noisy_d
d = 10
sizes = [d, 1]

dp = hyper_parameters.DataParameters(percent_correct, n, n_test, d)

hp = hyper_parameters.HyperParameters(batch_size=n,
                                      epochs=500,
                                      learning_rate= 1.0 / d ** 0.5,
                                      momentum=0.0,
                                      weight_decay=0.0,
                                      desired_success_rate=0.5,
                                      sizes=sizes,
                                      gamma=0.85,
                                      is_adam=True,
                                      all_linear=True,
                                      reg_type="L1",
                                      print_epoch=False,
                                      print_batch=False)

all_noise = dataset_creator.AllNoise(num_class=2, d=d)
trainer = train.SigmoidBxeTrainer()

weight_tracker = wt.AllWeights(len(sizes) - 1, node_limit=10)
_, max_grad_before, max_grad_after, zero_state, preds = runner.single_run(all_noise, dp, hp, 0, trainer, weight_tracker=weight_tracker, print_details=False, shuffle=True)
print(max_grad_before)
print(max_grad_after)
print(zero_state)
print(preds)

weight_tracker.show()