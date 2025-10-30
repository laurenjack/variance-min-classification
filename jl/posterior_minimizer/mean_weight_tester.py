import torch
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

from jl.posterior_minimizer import train
from jl.posterior_minimizer import dataset_creator as dc
from jl.posterior_minimizer import hyper_parameters
from jl.posterior_minimizer import custom_modules as cm
from jl.posterior_minimizer import regularizer as reg
from jl.posterior_minimizer import variance as v
from jl.posterior_minimizer import weight_tracker as wt

torch.manual_seed(184)


runs = 50
n = 100
n_test = 10
percent_correct = 0.5
d = 100
scale_of_mean = (1.0 / n) ** 0.5

dp = hyper_parameters.DataParameters(percent_correct, n, n_test, d)

hp = hyper_parameters.HyperParameters(batch_size=n,
                                      epochs=250,
                                      learning_rate=1.0,  # 1.0 / d ** 0.5,
                                      momentum=0.9,
                                      weight_decay=0.00,
                                      desired_success_rate=0.5,
                                      sizes=[d, 10, 1],  # 40, 30,
                                      do_train=True,
                                      gamma=1.0,
                                      is_adam=False,
                                      all_linear=False,
                                      reg_type="L1",  # InverseMagnitudeL2
                                      var_type="Analytical",
                                      reg_epsilon=0.0,
                                      print_epoch=False,
                                      print_batch=False)




sigmoid = torch.nn.Sigmoid()
prediction_1 = 0
inital_weight_list = []
weight_list = []
grad_at_zero_list = []
grad_at_zero_after_list = []
with_actual_mean = dc.Gaussian(d, perfect_class_balance=False)
regularizer = reg.create(dp, hp)

for r in range(runs):
    if r % 10 == 0:
        print(f"Run {r}")
    model = cm.Mlp(hp.sizes, is_bias=False, all_linear=hp.all_linear)
    x, y = with_actual_mean.generate_dataset(n, shuffle=False)
    x_test, y_test = with_actual_mean.generate_dataset(n_test, shuffle=True)
    train_set = TensorDataset(x, y)
    train_loader = DataLoader(train_set, hp.batch_size)
    test_set = TensorDataset(x_test, y_test)
    test_loader = DataLoader(test_set, n_test)
    # trainer = train.DirectMeanTrainer()
    trainer = train.SigmoidBxeTrainer()
    # print(f"W0 before {model.linears[0].weight.item()}")
    # print(f"W1 before {model.linears[1].weight.item()}")
    # linear = model.linears[0]
    # b = (2 * 3 / 1) ** 0.5
    # new_weights = torch.empty_like(linear.weight).uniform_(-b, b)
    # linear.weight.data = new_weights
    initial_w = model.linears[0].weight * model.linears[1].weight.t()
    inital_weight_list.append(initial_w)
    grad_at_zero =  v.grad_at_zero(model, x, y)[0] / model.linears[1].weight.t()

    trainer.run(model, train_loader, test_loader, hp, regularizer, weight_tracker=None)
    grad_at_zero_after = v.grad_at_zero(model, x, y)[0] / model.linears[1].weight.t()
    # print(f"W0 after {model.linears[0].weight.item()}")
    # print(f"W1 after {model.linears[1].weight.item()}")
    bool_index = y.bool()
    x1 = x[bool_index]
    a = sigmoid(model(x1))
    prediction_1 += torch.mean(a)
    w = model.linears[0].weight * model.linears[1].weight.t() # .item()
    weight_list.append(w)
    grad_at_zero_list.append(grad_at_zero)
    grad_at_zero_after_list.append(grad_at_zero_after)


# mean_dot = torch.zeros(hp.sizes[1])
# for iw, w in zip(inital_weight_list, weight_list):
#     # iw0 = inital_w[0]
#     # w0 = w[0]
#     # dot = torch.dot(iw0, w0) / (torch.norm(iw0) * torch.norm(w0))
#     prod = torch.sum(iw * w, dim=1) / (torch.sum(iw ** 2, dim=1) * torch.sum(iw ** 2, dim=1)) ** 0.5
#     mean_dot += prod
# mean_dot /= runs
# print(f"Mean Dot Product: {mean_dot}")

# def plot_sds(grads, runs, title):
#     std_devs = []
#     for g in grads:
#         std = g[0,0].item() * n ** 0.5
#         std_devs.append(std)
#     print(f"Mean sd: {sum(std_devs) / runs}")
#     plt.figure(figsize=(10, 6))
#     plt.hist(std_devs, bins='auto', edgecolor='black')
#     plt.title(f'Mean gradients {title}')
#     plt.xlabel('Standard Deviation')
#     plt.ylabel('Frequency')
#     plt.grid(True, alpha=0.3)
#     plt.figure()
#     plt.show()

def report(grad_list, runs):
    mean = sum(grad_list) / runs
    vars = sum([(grad - mean) ** 2 for grad in grad_list]) / runs
    print(f"Mean: {torch.mean(mean, dim=1)}")
    print(f"sd: {torch.mean(vars ** 0.5, dim=1)}")


print(scale_of_mean)
# sum_squares_per_dim = sum([w ** 2 for w in weight_list]) / runs
# sd_per_w = sum_squares_per_dim ** 0.5
# var_grads_per_dim = sum([grad ** 2 for grad in grad_at_zero_list]) / runs
# sd_per_grad = var_grads_per_dim ** 0.5
# var_grads_after_per_dim = sum([grad ** 2 for grad in grad_at_zero_after_list]) / runs
# sd_per_grad_after = var_grads_after_per_dim ** 0.5

report(grad_at_zero_list, runs)
report(grad_at_zero_after_list, runs)

# plot_sds(grad_at_zero_list, runs, 'before')
# plot_sds(grad_at_zero_after_list, runs, 'after')
# print(sd_per_w)
# print(torch.mean(sd_per_w))
# print(torch.mean(sd_per_grad))
# print(torch.mean(sd_per_grad_after))
# print(prediction_1 / runs)


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