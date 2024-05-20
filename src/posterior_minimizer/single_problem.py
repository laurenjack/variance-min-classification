import math

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

from src import hyper_parameters
from src import custom_modules as cm
from src.train import SigmoidBxeTrainer, BoundsAsParam, DirectReg, L1
from src import gradient_grapher
from src.posterior_minimizer import data_generator
from src import dataset_creator

torch.manual_seed(83274665)


def repeat(x, y, r):
    n = x.shape[0]
    num_false_signal = r // 2 + math.ceil(r ** 0.5)
    x = x.repeat(r, 1)
    for k in range(num_false_signal):
        x[k * n + 2, 1] = 2.0
    y = y.repeat(r)
    return x, y


def duplicate_noisy(x, noisy_d):
    copied_d = x[:, 1].unsqueeze(1).repeat(1, noisy_d)
    x = torch.cat((x[:, 0].unsqueeze(1), copied_d), dim=1)
    return x



r = 1
noisy_d = 1
x = torch.tensor([[1.0, 1.0, -1.0, -1.0, -1.0], [1.0, -1.0, -2.0, 1.0, -1.0]]).transpose(0, 1)
y = torch.tensor([1, 1, 1, 0, 0])
x_test = x.detach().clone()
# x_test[2, 1] = -2.0
x_test[2, 0] = 1.0
y_test = y.detach().clone()
x, y = repeat(x, y, r)
x = duplicate_noisy(x, noisy_d)
x_test = duplicate_noisy(x_test, noisy_d)

n = x.shape[0]
d = x.shape[1]


hp = hyper_parameters.HyperParameters(batch_size=n,
                                      epochs=30, # 300
                                      learning_rate=0.1, # 0.01
                                      momentum=0.2,
                                      weight_decay=0.0,
                                      post_constant=1.0,
                                      print_epoch=True,
                                      print_batch=False)


train_set = TensorDataset(x, y)
train_loader = DataLoader(train_set, hp.batch_size)
test_set = TensorDataset(x_test, y_test)
test_loader = DataLoader(test_set, hp.batch_size)

model = nn.Linear(d, 1, bias=False)

trainer = SigmoidBxeTrainer()
trainer.run(model, train_loader, test_loader, hp, direct_reg_constructor=L1)
print(model.weight)
print(torch.norm(model.weight))

# sigmoid = nn.Sigmoid()
# print(sigmoid(model(x)))







