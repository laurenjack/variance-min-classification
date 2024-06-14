import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from scipy.stats import binom

from src import hyper_parameters
from src import custom_modules as cm
from src.train import SigmoidBxeTrainer, BoundsAsParam, DirectReg, L1

from src import dataset_creator
from src.posterior_minimizer import weight_tracker as wt

torch.manual_seed(769)  # 392841 769

n = 100
n_test = 100
d = 1
desired_success_rate = 0.8
bp = (1 - desired_success_rate) / 2
threshold_success = binom.ppf(1 - bp, n, 0.5) - 1
post_constant = (threshold_success - 0.5 * n

all_noise = dataset_creator.AllNoise(num_class=2, d=d)
x, y = all_noise.generate_dataset(n, shuffle=True)
x[1,0] = 1.0
x[2,0] = -1.0
x_test, y_test = all_noise.generate_dataset(n_test, shuffle=True)


hp = hyper_parameters.HyperParameters(batch_size=n,
                                      epochs=400,
                                      learning_rate= 0.1,
                                      momentum=0.0,
                                      weight_decay=0.0,
                                      post_constant=0.5,
                                      gamma=0.9,
                                      is_adam=True,
                                      all_linear=False,
                                      print_epoch=False,
                                      print_batch=False)

train_set = TensorDataset(x, y)
train_loader = DataLoader(train_set, hp.batch_size)
test_set = TensorDataset(x_test, y_test)
test_loader = DataLoader(test_set, n_test)

model = cm.Mlp([d, 1], is_bias=False)
# with torch.no_grad():
#     model.linears[0].weight.copy_(1.0 * torch.ones([1, 1], dtype=torch.float32))

trainer = SigmoidBxeTrainer()
weight_tracker = wt.AllWeightsLinear()
trainer.run(model, train_loader, test_loader, hp, direct_reg_constructor=L1, weight_tracker=weight_tracker)
y_shift = y * 2 - 1
x0 = x[:,0]
prod = y_shift * x0
is_same = prod == 1
print(is_same.sum().item())
print(model.linears[0].weight)
sigmoid = nn.Sigmoid()
print(sigmoid(model(x_test[0])))
weight_tracker.show()