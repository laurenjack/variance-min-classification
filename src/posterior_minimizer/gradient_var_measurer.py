import torch
from torch import nn

from src import dataset_creator
from src import custom_modules as cm
from src.train import SigmoidBxeTrainer, BoundsAsParam
from src.posterior_minimizer.regularizer import DirectReg, L1, reg_grad, variance_grad
from src.posterior_minimizer.variance import manual_grad_calc, grad_at_zero

runs = 100
n = 1000
d = 11
sizes = [d, 7, 5, 3, 1] # 7, 3,


def calc_sd(grads):
    mean = torch.mean(grads, dim=0, keepdim=True)
    return torch.sum((grads - mean) ** 2, dim=0) ** 0.5


problem = dataset_creator.AllNoise(num_class=2, d=d)
x, y = problem.generate_dataset(n, shuffle=True)
model = cm.Mlp(sizes, is_bias=False, all_linear=True)

# Get the gradients
manual_grads = grad_at_zero(model, x, y, point_level=True)
var_grads = variance_grad(model, 0.5 / n ** 0.5)
reg_grads = reg_grad(model, 0.5 / n ** 0.5)
weights = [linear.weight for linear in model.linears]
sigmoid = nn.Sigmoid()
z = model(x)
a = sigmoid(z)
actual_grads = manual_grad_calc(weights, x.t(), (y.view(n, 1) - a) / n, is_var=False, point_level=True)

# loss = SigmoidBxeTrainer().loss(z, y)
# loss.backward()
# torch_grads = [linear.weight.grad for linear in model.linears]
# actual_grads_summed = manual_grad_calc(weights, x.t(), (y.view(n, 1) - a) / n, is_var=False, point_level=False)


# Report on each layer
for l in range(len(sizes) - 1):
    print(f"Layer {l}:")
    sd = calc_sd(manual_grads[l])
    actual_sd = calc_sd(actual_grads[l])
    print(f"  sd: {sd}")
    print("")
    print(f"  var_grad: {var_grads[l]}")
    print("")
    print(f"  reg_grad: {reg_grads[l]}")
    print("")
    print(f"  actual_sd: {actual_sd}")
    # print(f"  torch_grad: {torch_grads[l]}")
    # print("")
    # print(f"  actual_grad: {actual_grads_summed[l]}")
    print("")
    print("")






