import math

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

from jl.posterior_minimizer import hyper_parameters
from jl.posterior_minimizer import custom_modules as cm
from jl.posterior_minimizer.train import SigmoidBxeTrainer, BoundsAsParam
from jl.posterior_minimizer.regularizer import DirectReg, L1

from jl.posterior_minimizer import dataset_creator

torch.manual_seed(542821)

num_class = 2
patterns_per_class = 2
bits_per_pattern = 2
noisy_d = 200
hidden_layer = 8

n_per_pattern = 50
correct_per_pattern = 40
test_n_per_pattern = 100

distinct_inputs_problem = dataset_creator.DistinctInputsForFeatures(num_class, patterns_per_class, bits_per_pattern, noisy_d)
x, y, _ = distinct_inputs_problem.generate_dataset(n_per_pattern, correct_per_pattern, shuffle=True)
# We want to have 100% correct patterns in the test set, so we pass test_n_per_pattern twice
x_test, y_test, _ = distinct_inputs_problem.generate_dataset(test_n_per_pattern, test_n_per_pattern, shuffle=True)
n = x.shape[0]
d = x.shape[1]
n_test = x_test.shape[0]


hp = hyper_parameters.HyperParameters(batch_size=n,
                                      epochs=200, # 300
                                      learning_rate=0.03, # 0.01
                                      momentum=0.2,
                                      weight_decay=0.0,
                                      post_constant=0.0,
                                      print_epoch=True,
                                      print_batch=False)

train_set = TensorDataset(x, y)
train_loader = DataLoader(train_set, hp.batch_size)
test_set = TensorDataset(x_test, y_test)
test_loader = DataLoader(test_set, n_test)

model = cm.Mlp([d, hidden_layer, 1], is_bias=False)


trainer = SigmoidBxeTrainer()
trainer.run(model, train_loader, test_loader, hp, direct_reg_constructor=L1)
print(model.linears[0].weight)
print(torch.norm(model.linears[0].weight))