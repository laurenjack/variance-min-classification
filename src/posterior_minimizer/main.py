import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

from src import hyper_parameters
from src import custom_modules as cm
from src.train import SigmoidBxeTrainer
from src.posterior_minimizer import data_generator

torch.manual_seed(43181)

n_per_class = 5
test_n_per_class = 100
d = 3
p = 1.0


hp = hyper_parameters.HyperParameters(batch_size=n_per_class * 2,
                                      epochs=30, # 300
                                      learning_rate=0.2, # 0.01
                                      momentum=0.2,
                                      weight_decay=0.0,
                                      print_epoch=True,
                                      print_batch=False)


x, y = data_generator.uniform_one_true_dim(n_per_class, d, p)
print('Class 0')
print(x[(1 - y).bool()])
print('Class 1')
print(x[y.bool()])

train_set = TensorDataset(x, y)
train_loader = DataLoader(train_set, hp.batch_size)
# The test set will allow us to evaluate whether the model is using spurious dimensions
x, y = data_generator.uniform_one_true_dim(test_n_per_class, d, 1.0)
test_set = TensorDataset(x, y)
test_loader = DataLoader(test_set, hp.batch_size)
# model = nn.Linear(d, 1, bias=False)
model = cm.Linear(d, 1, bias=False)
trainer = SigmoidBxeTrainer()

trainer.run(model, train_loader, test_loader, hp)
print(model.weight)
