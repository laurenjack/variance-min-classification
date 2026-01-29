import math

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

from jl.posterior_minimizer import hyper_parameters
from jl.posterior_minimizer import custom_modules as cm
from jl.posterior_minimizer.train import SigmoidBxeTrainer, BoundsAsParam
from jl.posterior_minimizer.regularizer import DirectReg, L1
from jl.posterior_minimizer import weight_tracker as wt

from jl.posterior_minimizer import dataset_creator

torch.manual_seed(341111) # 73462

n = 800
n_test = 100
real_d = 1
noisy_d = n // 10 # n - real_d
d = real_d + noisy_d
root_d = math.ceil(d ** 0.5)

bra = dataset_creator.BinaryRandomAssigned(2, real_d, noisy_d=noisy_d, scale_by_root_d=False)
x, y, _ = bra.generate_dataset(n, percent_correct=0.8, shuffle=True)
x_test, y_test, _ = bra.generate_dataset(n_test, percent_correct=1.0, shuffle=True)


hp = hyper_parameters.HyperParameters(batch_size=n,
                                      epochs=300, # 300
                                      learning_rate= 0.03,
                                      momentum=0.0,
                                      weight_decay=0.0,
                                      post_constants=0.25,
                                      gamma=0.95,
                                      all_linear=True,
                                      print_epoch=True,
                                      print_batch=True)

train_set = TensorDataset(x, y)
train_loader = DataLoader(train_set, hp.batch_size)
test_set = TensorDataset(x_test, y_test)
test_loader = DataLoader(test_set, n_test)


# model = cm.Mlp([d, 1], is_bias=False)
# model = cm.HiddenLayerFixed(d, 0.125 * torch.ones(1, 8, dtype=torch.float32))
# inv_root = 1.0 / 8.0 ** 0.5
# fixed_w = -inv_root + 2 * inv_root * torch.rand((1, 8))
# print(fixed_w)
# model = cm.Mlp([d, 32, 32, 32, 1], is_bias=False)
model = cm.Mlp([d, 1, 1], is_bias=False)
# model = cm.HiddenLayersFixed([d, 8, 1])

trainer = SigmoidBxeTrainer()
trainer.run(model, train_loader, test_loader, hp, direct_reg_constructor=L1, weight_tracker=wt.SingleDimWeightTracker())
# print(model.linears[1].weight)
# print(model.linears[0].weight[:, 0])
# print(torch.norm(model.linears[0].weight))