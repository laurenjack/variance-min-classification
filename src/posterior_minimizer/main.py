import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

from src import hyper_parameters
from src import custom_modules as cm
from src.train import SigmoidBxeTrainer, BoundsAsParam, DirectReg
from src import gradient_grapher
from src.posterior_minimizer import data_generator
from src import dataset_creator

torch.manual_seed(43188) # 43181


def off_axis_basis(x):
    vectors = [torch.tensor([1., 1., 1.]), torch.tensor([-2., 1., 1.]), torch.tensor([1., 1., -2.])]
    basis_vectors = tuple([v / torch.norm(v) for v in vectors])
    basis = torch.stack(basis_vectors, dim=1)
    return torch.matmul(x, torch.inverse(basis))


n_per_class = 4
test_n_per_class = 100
real_d = 1
noisy_d = 2
d = real_d + noisy_d
p = 0.75


hp = hyper_parameters.HyperParameters(batch_size=n_per_class * 2,
                                      epochs=30, # 300
                                      learning_rate=0.1, # 0.01
                                      momentum=0.2,
                                      weight_decay=0.0,
                                      post_constant=0.0,
                                      print_epoch=True,
                                      print_batch=False)

# x, y = data_generator.uniform_one_true_dim(n_per_class, d, p)
bra = dataset_creator.BinaryRandomAssigned(2, real_d, noisy_d=noisy_d)
x, y = bra.generate_dataset(n_per_class * 2, percent_correct=p, shuffle=True)
train_set = TensorDataset(x, y)
train_loader = DataLoader(train_set, hp.batch_size)

print('Class 0')
print(x[(1 - y).bool()])
print('Class 1')
print(x[y.bool()])
# x = off_axis_basis(x)
# print('Class 0')
# print(x[(1 - y).bool()])
# print('Class 1')
# print(x[y.bool()])


# The test set will allow us to evaluate whether the model is using spurious dimensions
# x_test, y_test = data_generator.uniform_one_true_dim(test_n_per_class, d, 1.0)
x_test, y_test = bra.generate_dataset(test_n_per_class * 2, percent_correct=1.0, shuffle=True)
# x_test = off_axis_basis(x_test)
test_set = TensorDataset(x_test, y_test)
test_loader = DataLoader(test_set, hp.batch_size)

model = nn.Linear(d, 1, bias=False)
# model = cm.Linear(d, 1, bias=False)

# c_values = [e / 10 - 2 for e in range(1, 41)]
# gradient_grapher.run(model, x, y, hp, c_values)

# print(model.weight)
trainer = SigmoidBxeTrainer()
trainer.run(model, train_loader, test_loader, hp, direct_reg=BoundsAsParam) # BoundsAsParam
print(model.weight)
print(torch.norm(model.weight))
# print( model.weight.grad)
# print(model.weight_mag)
# print(model.w)

sigmoid = nn.Sigmoid()
print(sigmoid(model(x)))

# c_values = [e / 100 for e in range(1, 301)]
# gradient_grapher.run(model, x, y, hp, c_values)






# print(model.weight)
# print(model.weight_mag)
# print(model.w)