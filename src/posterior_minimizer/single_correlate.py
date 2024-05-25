import math

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

from src import hyper_parameters
from src import custom_modules as cm
from src.train import SigmoidBxeTrainer, BoundsAsParam, DirectReg, L1

from src import dataset_creator

torch.manual_seed(73462) # 83274665

n = 200
n_test = 100
real_d = 1
noisy_d = 2999
d = real_d + noisy_d

bra = dataset_creator.BinaryRandomAssigned(2, real_d, noisy_d=noisy_d)
x, y = bra.generate_dataset(n, percent_correct=0.8, shuffle=True)
x_test, y_test = bra.generate_dataset(n_test, percent_correct=1.0, shuffle=True)




hp = hyper_parameters.HyperParameters(batch_size=n,
                                      epochs=1000, # 300
                                      learning_rate=0.00003,
                                      momentum=0.2,
                                      weight_decay=0.0,
                                      post_constant=1.0 / 256.0 ** 0.5,
                                      print_epoch=True,
                                      print_batch=False)

train_set = TensorDataset(x, y)
train_loader = DataLoader(train_set, hp.batch_size)
test_set = TensorDataset(x_test, y_test)
test_loader = DataLoader(test_set, n_test)


# model = cm.Mlp([d, 1], is_bias=False)
# model = cm.HiddenLayerFixed(d, 0.125 * torch.ones(1, 8, dtype=torch.float32))
# inv_root = 1.0 / 8.0 ** 0.5
# fixed_w = -inv_root + 2 * inv_root * torch.rand((1, 8))
# print(fixed_w)
# model = cm.HiddenLayerFixed(d, fixed_w)
model = cm.Mlp([d, 256, 1], is_bias=False)

trainer = SigmoidBxeTrainer()
trainer.run(model, train_loader, test_loader, hp, direct_reg_constructor=L1)
print(model.linears[1].weight)
print(model.linears[0].weight[:, 0])
# print(torch.norm(model.linears[0].weight))