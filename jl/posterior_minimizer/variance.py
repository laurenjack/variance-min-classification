from scipy.stats import norm, chi2

import torch
from torch import nn

from jl.posterior_minimizer.hyper_parameters import DataParameters, HyperParameters


def _propagate_gradients(model, forward_product, backward_product):
    relu = nn.ReLU()
    weights = [(linear.weight, linear.bias) for linear in model.linears]
    num_layers = len(weights)
    is_actives = []
    forward_products = []
    for l, (W, b) in enumerate(weights):
        W = W.detach()
        if b is None:
            b = 0.0
        else:
            b = b.detach()
        forward_products.append(forward_product)
        forward_product = W @ forward_product + b
        if not model.all_linear and l < num_layers - 1:
            forward_product = relu(forward_product)
        is_active = (forward_product != 0)
        is_active = is_active.float().t()
        is_actives.append(is_active)

    backward_products = []
    for (W, b), is_active in zip(reversed(weights), reversed(is_actives)):
        W = W.detach()
        backward_product = is_active * backward_product
        backward_products.append(backward_product)
        backward_product = backward_product @ W

    return forward_products, backward_products


def manual_grad_calc(model, forward_product, backward_product, point_level=False):
    forward_products, backward_products = _propagate_gradients(model, forward_product, backward_product)
    # Now for each weight, combine the forward and backward product at that point to form the regularizer.
    grads = []
    bias_grads = []
    num_layers = len(model.linears)
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
        bias_grads.append(torch.sum(bp, dim=0))
        grads.append(grad)
    return grads, bias_grads


def _propagate_variance(model, forward_product, backward_product):
    weights = [(linear.weight, linear.bias) for linear in model.linears]
    forward_products = []
    for W, b in weights:
        W = W.detach()
        forward_products.append(forward_product)
        if b is None:
            b = 0.0
        else:
            b = b.detach()
        forward_product = W @ forward_product + b

    backward_products = []
    for W, b in reversed(weights):
        W = W.detach()
        backward_products.append(backward_product)
        backward_product = backward_product @ W
    return forward_products, backward_products


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

    def __init__(self, dp, hp):
        self.sizes = hp.sizes
        self.all_linear = hp.all_linear
        self.is_bias = hp.is_bias
        self.n = dp.n
        self.percent_correct = dp.percent_correct
        self.desired_success_rate = hp.desired_success_rate
        self.relu_bound = hp.relu_bound

    def calculate(self, model, x, y, z_scaled=True):
        n, d = x.shape
        forward_product = torch.eye(d)
        backward_product = torch.ones(1, 1)
        return self.variance_grad_calc(model, forward_product, backward_product, z_scaled=z_scaled)

    def scale_calc(self):
        pass

    def variance_grad_calc(self, model, forward_product, backward_product, z_scaled=True):
        pass


class L1Style(Variance):

    def scale_calc(self):
        d = self.sizes[0]
        if self.all_linear:
            num_in = d - 1 # sum(self.sizes[:-1])
            if self.is_bias:
                num_in += 1
        else:
            num_in = self.sizes[0] * self.sizes[1]
        bp = (1 - self.desired_success_rate ** (1 / num_in)) / 2
        z = norm.ppf(1 - bp)

        prop_product = self.percent_correct * (1 - self.percent_correct)
        sd_scale = (prop_product / self.n) ** 0.5
        if not self.all_linear:
            sd_scale *= self.relu_bound ** 0.5 # (0.5 - 0.5 / (d * math.pi)) ** 0.5
        z *= sd_scale
        return z

    def variance_grad_calc(self, model, forward_product, backward_product, z_scaled=True):
        forward_products, backward_products = _propagate_variance(model, forward_product, backward_product)
        # Now for each weight, combine the forward and backward product at that point to form the regularizer.
        grads = []
        bias_grads = []
        if z_scaled:
            z = self.scale_calc()
        else:
            z = 1.0
        num_layers = len(model.linears)
        for l in range(num_layers):
            fp = forward_products[l]
            bp = backward_products[-(l + 1)]
            bp = bp ** 2
            fp = fp ** 2
            fp = torch.sum(fp, dim=1, keepdim=True)
            grad = bp.t() @ fp.t()
            # To standard deviation
            grad = z * grad ** 0.5
            grads.append(grad)
            bias_grads.append(bp[0] ** 0.5)  # TODO(Jack) check this index
        return grads, bias_grads


class L2Style(Variance):

    def scale_calc(self):
        #TODO(Jack) remove this - only works for single layer, with direct mean training
        d = self.sizes[0]
        post_constant = chi2.ppf(self.desired_success_rate, d)
        post_constant *= 4 / self.n
        z = post_constant ** 0.5
        return z

    def variance_grad_calc(self, model, forward_product, backward_product, z_scaled=True):
        forward_products, backward_products = _propagate_variance(model, forward_product, backward_product)
        grads = []
        bias_grads = []
        num_layers = len(model.linears)
        for l in range(num_layers):
            fp = forward_products[l]
            dl, _ = fp.shape
            bp = backward_products[-(l + 1)]
            bp = bp ** 2
            fp = fp ** 2
            # Sum across dim=1 to get variance scaler for each input
            fp = torch.sum(fp, dim=1)
            # TODO(Jack) check this maths
            if model.is_bias:
                fp += 1
            mean = torch.sum(fp)
            variance = torch.sum(fp ** 2)
            # Satterthwaite Approximation of generalized chi-square as chi-squared
            f = mean ** 2 / variance
            scale = variance / mean
            # percentile = (1 - desired_success_rate ** (1 / num_layers))
            # post_constant = chi2.ppf(1 - percentile, f)
            post_constant = chi2.ppf(self.desired_success_rate, f)
            post_constant *= 4 / self.n * scale

            # The inner dimension is always just 1 here
            grad = post_constant * bp.t()
            grad = grad ** 0.5
            bias_grads.append(grad[0])  # TODO(Jack) check this index
            grad = grad.repeat(1, dl)
            grads.append(grad)

            return grads, bias_grads


def create_variance(dp: DataParameters, hp: HyperParameters):
    if hp.reg_type == 'L1' or hp.reg_type == 'NoReg':
        return L1Style(dp, hp)
    elif hp.reg_type == 'L2':
        return L2Style(dp, hp)
    raise ValueError(f"No Variance defined for reg type: {hp.reg_type}")
