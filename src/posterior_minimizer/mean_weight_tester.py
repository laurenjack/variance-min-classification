import torch
from torch.utils.data import TensorDataset, DataLoader

from src import train
from src import dataset_creator as dc
from src import hyper_parameters
from src import custom_modules as cm
from src.posterior_minimizer import regularizer as reg
from src.posterior_minimizer import weight_tracker as wt

# torch.manual_seed(3411129)

n = 100
n_test = 10
d = 10
scale_of_mean = (1.0 / n) ** 0.5


hp = hyper_parameters.HyperParameters(batch_size=n,
                                      epochs=300,
                                      learning_rate= 0.1, # / d ** 0.5,
                                      momentum=0.0,
                                      weight_decay=0.0,
                                      desired_success_rate=0.72,
                                      sizes=[d, 1],
                                      gamma=1.0,
                                      is_adam=True,
                                      all_linear=True,
                                      reg_type="DirectReg",
                                      print_epoch=False,
                                      print_batch=False)

runs = 100

sigmoid = torch.nn.Sigmoid()
prediction_1 = 0
sum_weight = 0
sum_squares = 0
sum_magnintudes = 0
weight_list = []
magnitude_list = []
with_actual_mean = dc.SingleDirectionGaussian(d, scale_of_mean=0.0)
for r in range(runs):
    print(f"Run {r}")
    model = cm.Mlp(hp.sizes, is_bias=False, all_linear=hp.all_linear)
    x, y = with_actual_mean.generate_dataset(n, shuffle=False)
    x_test, y_test = with_actual_mean.generate_dataset(n_test, shuffle=True)
    train_set = TensorDataset(x, y)
    train_loader = DataLoader(train_set, hp.batch_size)
    test_set = TensorDataset(x_test, y_test)
    test_loader = DataLoader(test_set, n_test)
    trainer = train.DirectMeanTrainer()
    trainer.run(model, train_loader, test_loader, hp, reg.DirectReg(0.5), weight_tracker=None)
    bool_index = y.bool()
    x1 = x[bool_index]
    a = sigmoid(model(x1))
    prediction_1 += torch.mean(a)
    w = model.linears[0].weight # .item()
    sum_weight += w
    sum_square = torch.sum(w ** 2.0).item()
    sum_squares += sum_square
    magnitude = sum_square ** 0.5
    sum_magnintudes += magnitude
    magnitude_list.append(magnitude)
    weight_list.append(w)

print(scale_of_mean)
var = sum_squares / runs
print(var)
print(var ** 0.5)
mean_weight = sum_weight / runs
sum_squares_per_dim = sum([(w - mean_weight) ** 2.0 for w in weight_list]) / runs
print(sum_squares_per_dim ** 0.5)
print(prediction_1 / runs)


# weight_tracker = wt.AllWeights(len(hp.sizes) - 1, node_limit=10)
# model = cm.Mlp(hp.sizes, is_bias=False, all_linear=hp.all_linear)
# with_actual_mean = dc.SingleDirectionGaussian(d, scale_of_mean=0.0)
# x, y = with_actual_mean.generate_dataset(n, shuffle=False)
# x_test, y_test = with_actual_mean.generate_dataset(n_test, shuffle=True)
# train_set = TensorDataset(x, y)
# train_loader = DataLoader(train_set, hp.batch_size)
# test_set = TensorDataset(x_test, y_test)
# test_loader = DataLoader(test_set, n_test)
# trainer = train.SigmoidBxeTrainer()
# trainer.run(model, train_loader, test_loader, hp, reg.DirectReg(0.5), weight_tracker=weight_tracker)
# print(model.linears[0].weight)
# weight_tracker.show()