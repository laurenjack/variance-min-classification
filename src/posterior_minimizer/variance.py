import math

import torch
from torch import nn

from src.hyper_parameters import HyperParameters


def manual_grad_calc(model, forward_product, backward_product, point_level=False):
    relu = nn.ReLU()
    weights = [linear.weight for linear in model.linears]
    num_layers = len(weights)
    is_actives = []
    forward_products = []
    for l, W in enumerate(weights):
        W = W.detach()
        forward_products.append(forward_product)
        forward_product = W @ forward_product
        if not model.all_linear and l < num_layers - 1:
            forward_product = relu(forward_product)
        is_active = (forward_product != 0)
        is_active = is_active.float().t()
        is_actives.append(is_active)

    backward_products = []
    for W, is_active in zip(reversed(weights), reversed(is_actives)):
        W = W.detach()
        backward_product = is_active * backward_product
        backward_products.append(backward_product)
        backward_product = backward_product @ W

    # Now for each weight, combine the forward and backward product at that point to form the regularizer.
    grads = []
    num_layers = len(weights)
    for l in range(num_layers):
        fp = forward_products[l]
        bp = backward_products[-(l + 1)]
        dl, n = fp.shape
        _, dl1 = bp.shape
        if point_level:
            grad = bp.view(1, n, dl1) * fp.view(dl, n, 1)
            grad = grad.permute(1, 2, 0)  # Put the data dim first
        else:
            grad = bp.t() @ fp.t()
        grads.append(grad)
    return grads


def variance_grad_calc(model, forward_product, backward_product):
    weights = [linear.weight for linear in model.linears]
    forward_products = []
    for W in weights:
        W = W.detach()
        forward_products.append(forward_product)
        forward_product = W @ forward_product

    backward_products = []
    for W in reversed(weights):
        W = W.detach()
        backward_products.append(backward_product)
        backward_product = backward_product @ W

    # Now for each weight, combine the forward and backward product at that point to form the regularizer.
    grads = []
    num_layers = len(weights)
    for l in range(num_layers):
        fp = forward_products[l]
        bp = backward_products[-(l + 1)]
        bp = bp ** 2
        fp = fp ** 2
        fp = torch.sum(fp, dim=1, keepdim=True)
        grad = bp.t() @ fp.t()
        grads.append(grad)
    return grads


def grad_at_zero(model, x, y, percent_correct, point_level=False):
    n, d = x.shape
    y_shift = y.view(n, 1) * 2.0 - 1
    forward_product = x.t()
    backward_product = - (percent_correct * (1 - percent_correct)) ** 0.5 * y_shift / n  #
    backward_product = backward_product.view(n, 1)
    # The two commented out lines below apply to the DirectMeanTrainer
    # forward_product = -(2 * x * y_shift).t() / n  # (d, n)
    # backward_product = torch.ones(n, 1)
    return manual_grad_calc(model, forward_product, backward_product, point_level=point_level)


class Variance:

    def calculate(self, model, x, y):
        pass


class Empirical(Variance):

    def calculate(self, model, x, y):
        sigmoid = nn.Sigmoid()
        n, d = x.shape
        # y_shift = y * 2.0 - 1
        # y_shift = y_shift.view(n, 1)
        # mean = model(torch.eye(d))
        # target = y_shift.view(n, 1) @ mean.t()
        # fp = (target - 2 * x).t() / n  # (d, n)
        # bp = y_shift  # (n, 1)
        z = model(x)
        a = sigmoid(z)
        bp = -(y.view(n, 1) - a) / n
        point_level_grads = manual_grad_calc(model, x.t(), bp, point_level=True)  # list[(n, dl+1, dl]
        mean_grads = [torch.sum(plg, dim=0, keepdim=True) for plg in point_level_grads]
        mgs = [linear.weight.grad for linear in model.linears]
        return [torch.mean((n * plg - mg) ** 2, dim=0) / (n - 1) for plg, mg in zip(point_level_grads, mean_grads)]


class EmpiricalAtFlat(Variance):

    def calculate(self, model, x, y):
        n, d = x.shape
        point_level_grads = grad_at_zero(model, x, y, point_level=True)
        mean_grads = [torch.sum(plg, dim=0, keepdim=True) for plg in point_level_grads]
        return [torch.mean((n * plg - mg) ** 2, dim=0) / (n - 1) for plg, mg in zip(point_level_grads, mean_grads)]


class Analytical(Variance):

    # def calculate(self, model, x, y):
    #     n, d = x.shape
    #     forward_product = torch.eye(d)
    #     # prop_product = dp.percent_correct * (1 - dp.percent_correct)
    #     sd_scale = 0.5
    #     if not model.all_linear:
    #         sd_scale *= (0.5 - 0.5 / (d * math.pi)) ** 0.5
    #     backward_product = sd_scale / n ** 0.5 * torch.ones(1, 1)  # 2
    #     return variance_grad_calc(model, forward_product, backward_product)

    def calculate(self, model, x, y):
        n, d = x.shape
        forward_product = torch.eye(d)
        backward_product = torch.ones(1, 1)  # 2
        return variance_grad_calc(model, forward_product, backward_product)


def create_variance(hp: HyperParameters):
    if hp.var_type == 'Empirical':
        return Empirical()
    elif hp.var_type == 'EmpiricalAtFlat':
        return EmpiricalAtFlat()
    elif hp.var_type == 'Analytical':
        return Analytical()
    raise ValueError(f"Unrecognized variance type {hp.var_type}")
