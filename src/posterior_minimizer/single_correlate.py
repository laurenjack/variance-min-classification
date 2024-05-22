import math

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

from src import hyper_parameters
from src import custom_modules as cm
from src.train import SigmoidBxeTrainer, BoundsAsParam, DirectReg, L1

from src import dataset_creator

torch.manual_seed(83274665)

n = 200
n_test = 100
real_d = 1
noisy_d = 2999
d = real_d + noisy_d

bra = dataset_creator.BinaryRandomAssigned(2, real_d, noisy_d=noisy_d)
x, y = bra.generate_dataset(n, percent_correct=0.8, shuffle=True)
x_test, y_test = bra.generate_dataset(n_test, percent_correct=1.0, shuffle=True)




hp = hyper_parameters.HyperParameters(batch_size=n,
                                      epochs=500, # 300
                                      learning_rate=0.015, # 0.01
                                      momentum=0.2,
                                      weight_decay=0.0,
                                      post_constant=1.0,
                                      print_epoch=True,
                                      print_batch=False)

train_set = TensorDataset(x, y)
train_loader = DataLoader(train_set, hp.batch_size)
test_set = TensorDataset(x_test, y_test)
test_loader = DataLoader(test_set, n_test)


model = nn.Linear(d, 1, bias=False)

trainer = SigmoidBxeTrainer()
trainer.run(model, train_loader, test_loader, hp, direct_reg_constructor=L1)
print(model.weight)
print(torch.norm(model.weight))