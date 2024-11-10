import torch

from src.hyper_parameters import HyperParameters


def manual_grad_calc(model, forward_product, backward_product, is_variance=False, point_level=False):
    assert not (is_variance and point_level)
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
        dl, n = fp.shape
        _, dl1 = bp.shape
        if point_level:
            grad = bp.view(1, n, dl1) * fp.view(dl, n, 1)
            grad = grad.permute(1, 2, 0)  # Put the data dim first
        else:
            if is_variance:
                bp = bp ** 2
                fp = fp ** 2
                fp = torch.sum(fp, dim=1, keepdim=True)
            grad = bp.t() @ fp.t()
        grads.append(grad)
    return grads


def grad_at_zero(model, x, y, point_level=False):
    n, d = x.shape
    y_shift = y.view(n, 1) * 2.0 - 1
    forward_product = x.t()
    backward_product = - 0.5 * y_shift / n
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
        weights = [linear.weight.detach() for linear in model.linears]
        n, d = x.shape
        y_shift = y * 2.0 - 1
        y_shift = y_shift.view(n, 1)
        mean = model(torch.eye(d))
        target = y_shift.view(n, 1) @ mean.t()
        fp = (target - 2 * x).t() / n  # (d, n)
        bp = y_shift  # (n, 1)
        point_level_grads = manual_grad_calc(weights, fp, bp, point_level=True)  # list[(n, dl+1, dl]
        mean_grads = [torch.sum(plg, dim=0, keepdim=True) for plg in point_level_grads]
        return [torch.mean((n * plg - mg) ** 2, dim=0) / (n - 1) for plg, mg in zip(point_level_grads, mean_grads)]


class EmpiricalAtFlat(Variance):

    def calculate(self, model, x, y):
        n, d = x.shape
        point_level_grads = grad_at_zero(model, x, y, point_level=True)
        mean_grads = [torch.sum(plg, dim=0, keepdim=True) for plg in point_level_grads]
        return [torch.mean((n * plg - mg) ** 2, dim=0) / (n - 1) for plg, mg in zip(point_level_grads, mean_grads)]


class Analytical(Variance):

    def calculate(self, model, x, y):
        n, d = x.shape
        forward_product = torch.eye(d)
        # prop_product = dp.percent_correct * (1 - dp.percent_correct)
        backward_product = 0.5 / n ** 0.5 * torch.ones(1, 1)  # 2
        return manual_grad_calc(model, forward_product, backward_product, is_variance=True)


def create_variance(hp: HyperParameters):
    if hp.var_type == 'Empirical':
        return Empirical()
    elif hp.var_type == 'EmpiricalAtFlat':
        return EmpiricalAtFlat()
    elif hp.var_type == 'Analytical':
        return Analytical()
    raise ValueError(f"Unrecognized variance type {hp.var_type}")
