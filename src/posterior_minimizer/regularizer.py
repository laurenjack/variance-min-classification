import torch
from scipy.stats import norm, chi2

from src.hyper_parameters import DataParameters, HyperParameters
from src.posterior_minimizer import variance as v
from src.posterior_minimizer.variance import Variance


def create(dp: DataParameters, hp: HyperParameters):
    variance = v.create_variance(hp)
    if hp.reg_type == "L1":
        reg_constructor = L1
    elif hp.reg_type == "L2":
        reg_constructor = L2
    elif hp.reg_type == "NoReg":
        reg_constructor = NoReg
    else:
        raise ValueError(f"Unsupported Regualrizer: {hp.reg_type}")
    return reg_constructor(variance, dp, hp)


class DirectReg:
    """
    Modifies the gradients of a model directly, to apply a regularizer.
    """

    def __init__(self, variance: Variance, dp: DataParameters, hp: HyperParameters):
        self.variance = variance
        self.sizes = hp.sizes
        self.desired_success_rate = hp.desired_success_rate
        self.n = dp.n

    def apply(self, model, x, y):
        regs = self.reg_grad(model, x, y)
        for l, reg in enumerate(regs):
            model.linears[l].weight.grad += reg

    # def get_max_gradient(self, model, x, y):
    #     n = x.shape[0]
    #     y_shift = y * 2 - 1
    #     means = torch.mean(x * y_shift.view(n, 1), dim=0)
    #     squared_mag = torch.sum((2 * means) ** 2) ** 0.5
    #     return RegularizationState(squared_mag, self.post_constant, "Actual Squared Mean Magnitude")
    #
    # def get_zero_state(self, model, x, y):
    #     d = x.shape[1]
    #     means = model(torch.eye(d))
    #     squared_mag = torch.sum(means ** 2) ** 0.5
    #     return RegularizationState(squared_mag, self.zero_threshold, "Squared Weight Magnitude")


class NoReg(DirectReg):

    def __init__(self, variance: Variance, dp: DataParameters, hp: HyperParameters):
        super().__init__(variance, dp, hp)

    def apply(self, model, x, y):
        pass


class L2(DirectReg):

    def __init__(self, variance: Variance, dp: DataParameters, hp: HyperParameters):
        # zero_threshold = post_constant / (1 + post_constant)
        super().__init__(variance, dp, hp)

    def reg_grad(self, model, x, y):
        n, d = x.shape
        num_out = len(self.sizes) - 1  # sum(hp.sizes[:-1]) # sum(hp.sizes[1:])
        bp = (1 - self.desired_success_rate ** (1 / num_out))
        post_constant = chi2.ppf(1 - bp, d)
        post_constant *= 4 / n
        post_constant = post_constant ** 0.5
        return [post_constant * linear.weight.detach() for linear in model.linears]

    # def get_max_gradient(self, model, x, y):
    #     return self.get_greatest_manual_grad(model, x, y)

    def get_zero_state(self, model, x, y):
        d = x.shape[1]
        means = model(torch.eye(d))
        squared_mag = torch.sum(means ** 2) ** 0.5
        return RegularizationState(squared_mag, self.zero_threshold, "Squared Weight Magnitude")



class L1(DirectReg):

    ZERO_THRESHOLD = 0.0005

    def __init__(self, variance: Variance, dp: DataParameters, hp: HyperParameters):
        super().__init__(variance, dp, hp)

    def reg_grad(self, model, x, y):
        weights = [linear.weight for linear in model.linears]
        variances = self.variance.calculate(model, x, y)
        num_in = sum(self.sizes[:-1])
        bp = (1 - self.desired_success_rate ** (1 / num_in)) / 2
        # prop_product = dp.percent_correct * (1 - dp.percent_correct)
        z = norm.ppf(1 - bp)
        return [torch.sign(W) * z * var ** 0.5 for W, var in zip(weights, variances)]

    def get_zero_state(self, model, x, y):
        d = x.shape[1]
        means = model(torch.eye(d))
        squared_mag = torch.sum(means ** 2) ** 0.5
        return RegularizationState(squared_mag, L1.ZERO_THRESHOLD, "Squared Weight Magnitude")

    def get_max_gradient(self, model, x, y):
        regs = self.reg_grad(model, x, y)
        manual_grads = v.grad_at_zero(model, x, y)
        # if isinstance(self, InverseMagnitudeL2):
        #     reg_grads = [torch.sum(reg_grad ** 2.0, dim=1) ** 0.5 for reg_grad in reg_grads]
        #     manual_grads = [torch.sum(manual_grad ** 2.0, dim=1) ** 0.5 for manual_grad in manual_grads]
        has_gradient_greater_reg = False
        greatest_delta = -10000000
        greatest_deltas_grad = None
        greatest_deltas_reg = None
        for manual_grad, reg in zip(manual_grads, regs):
            abs_manual_grad = torch.abs(manual_grad)
            abs_reg = torch.abs(reg)
            delta = abs_manual_grad - abs_reg
            positive_index = delta > 0
            has_gradient_greater_reg = has_gradient_greater_reg or torch.any(positive_index)
            flattened_delta = delta.view(-1)
            max_index = torch.argmax(flattened_delta)
            next_max_delta = flattened_delta[max_index].item()
            if next_max_delta > greatest_delta:
                greatest_delta = next_max_delta
                greatest_deltas_grad = abs_manual_grad.view(-1)[max_index].item()
                greatest_deltas_reg = abs_reg.view(-1)[max_index].item()
        return RegularizationState(greatest_deltas_grad, greatest_deltas_reg, "Greatest Gradient")


class RegularizationState:
    """
    A number (value) in relation to a threshold, that represents whether a network was 'fully regularized', in the
    sense that random signals were sufficiently suppressed.
    """

    def __init__(self, value, threshold, description):
        self.value = value
        self.threshold = threshold
        self.description = description

    def exceeds_threshold(self):
        return self.value > self.threshold

    def __str__(self):
        return f"{self.description}: ({self.value}, {self.threshold})"
