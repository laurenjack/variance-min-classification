import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

from src import hyper_parameters
from src import custom_modules as cm
from src.train import SigmoidBxeTrainer
from src import gradient_grapher
from src.posterior_minimizer import data_generator

torch.manual_seed(43181)


def off_axis_basis(x):
    vectors = [torch.tensor([1., 1., 1.]), torch.tensor([-2., 1., 1.]), torch.tensor([1., 1., -2.])]
    basis_vectors = tuple([v / torch.norm(v) for v in vectors])
    basis = torch.stack(basis_vectors, dim=1)
    return torch.matmul(x, torch.inverse(basis))


n_per_class = 5
test_n_per_class = 100
d = 3
p = 1.0


hp = hyper_parameters.HyperParameters(batch_size=n_per_class * 2,
                                      epochs=1000, # 300
                                      learning_rate=0.4, # 0.01
                                      momentum=0.2,
                                      weight_decay=0.0,
                                      post_constant=0.13,
                                      print_epoch=False,
                                      print_batch=False)


x, y = data_generator.uniform_one_true_dim(n_per_class, d, p)

print('Class 0')
print(x[(1 - y).bool()])
print('Class 1')
print(x[y.bool()])
# x = off_axis_basis(x)
# print('Class 0')
# print(x[(1 - y).bool()])
# print('Class 1')
# print(x[y.bool()])

train_set = TensorDataset(x, y)
train_loader = DataLoader(train_set, hp.batch_size)
# The test set will allow us to evaluate whether the model is using spurious dimensions
x_test, y_test = data_generator.uniform_one_true_dim(test_n_per_class, d, 1.0)
# x = off_axis_basis(x)
test_set = TensorDataset(x_test, y_test)
test_loader = DataLoader(test_set, hp.batch_size)
# model = nn.Linear(d, 1, bias=False)
model = cm.Linear(d, 1, bias=False)

# c_values = [e / 10 - 2 for e in range(1, 41)]
# gradient_grapher.run(model, x, y, hp, c_values)

# print(model.weight)
trainer = SigmoidBxeTrainer()
trainer.run(model, train_loader, test_loader, hp)
print(model.weight)
print(model.weight_mag)
print(model.w)

sigmoid = nn.Sigmoid()
print(sigmoid(model(x)))

c_values = [e / 10 - 2 for e in range(1, 41)]
gradient_grapher.run(model, x, y, hp, c_values)






# print(model.weight)
# print(model.weight_mag)
# print(model.w)