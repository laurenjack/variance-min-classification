import math
import torch
from scipy.stats import norm, chi2

from src.hyper_parameters import DataParameters, HyperParameters
from src.posterior_minimizer import variance as v
from src.posterior_minimizer.variance import Variance


ZERO_THRESHOLD = 0.0001


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
        self.percent_correct = dp.percent_correct
        self.sigmoid = torch.nn.Sigmoid()

    def apply(self, model, x, y, epoch):
        z, sds = self.reg_grad(model, x, y)
        for l, sd in enumerate(sds):
            W = model.linears[l].weight
            w = W.detach()
            grad = W.grad
            # If the regularizer overpowers the gradient, set it to zero
            # if epoch >= 50:
            #     # if epoch == 50:
            #     #     print(self.get_max_gradient(model, x, y))
            #     #     print('Yeow')
            reg = z * sd
            reg *= torch.sign(w)
            updated_grad = torch.abs(w) * ((grad + reg) / (sd + 1e-8) + 0.03 * w)
            # updated_grad = grad / (sd + 1e-8) + 0.01 * w
            W.grad = updated_grad

            #_, d1 = w.shape
            #index_max_W = torch.argmax(torch.abs(w)).item()
            # row = index_max_W // d1
            # col = index_max_W % d1
            # max_W = W[row, col]
            # sd_of_max = sd[row, col]
            # grad_of_max = updated_grad[row, col]
            # reg = reg * torch.sign(grad)

            # updated_grad = torch.where(torch.abs(grad) > torch.abs(reg), (grad - reg) / sd, 0.0)
            # else:
            #     updated_grad = grad / sd
            # updated_grad += 0.1 * W.detach()

            # model.linears[l].weight.grad += reg

    def get_max_gradient(self, model, x, y, true_d=0):
        z, sds = self.reg_grad(model, x, y)
        regs = [z * sd for sd in sds]
        manual_grads = v.grad_at_zero(model, x, y, self.percent_correct)
        # if isinstance(self, InverseMagnitudeL2):
        #     reg_grads = [torch.sum(reg_grad ** 2.0, dim=1) ** 0.5 for reg_grad in reg_grads]
        #     manual_grads = [torch.sum(manual_grad ** 2.0, dim=1) ** 0.5 for manual_grad in manual_grads]
        has_gradient_greater_reg = False
        greatest_delta = -10000000
        greatest_deltas_grad = None
        greatest_deltas_reg = None
        for manual_grad, reg in zip(manual_grads[:1], regs[:1]):
            abs_manual_grad = torch.abs(manual_grad[:, true_d:])
            abs_reg = torch.abs(reg[:, true_d:])
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

    def get_zero_state(self, model, x, y, true_d=0):
        # Zero out the true dimensions, so we can see the regularizer had the intended effect
        x = x.clone()
        x[:, :true_d] = 0.0
        d = x.shape[1]
        # means = model(torch.eye(d))
        # squared_mag = torch.sum(means ** 2) ** 0.5
        z = model(x)
        squared_mag = (torch.sum(z ** 2) / d) ** 0.5
        return RegularizationState(squared_mag, ZERO_THRESHOLD, "Squared Weight Magnitude")


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

    def reg_grad(self, model, x, y):
        variances = self.variance.calculate(model, x, y)
        return 0.0, [var ** 0.5 for var in variances]


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


first = True

class L1(DirectReg):

    def __init__(self, variance: Variance, dp: DataParameters, hp: HyperParameters):
        super().__init__(variance, dp, hp)

    def reg_grad(self, model, x, y):
        # weights = [linear.weight for linear in model.linears]
        variances = self.variance.calculate(model, x, y)
        d = self.sizes[0]
        if d == 1:
            num_in = 1
        elif model.all_linear:
            num_in = d  # sum(self.sizes[:-1])
        else:
            num_in = self.sizes[0] * self.sizes[1]
            # num_in = 0
            # for sl, sl1 in zip(self.sizes[:-1], self.sizes[1:]):
            #     num_in += sl * sl1
        bp = (1 - self.desired_success_rate ** (1 / num_in)) / 2
        z = norm.ppf(1 - bp)

        prop_product = self.percent_correct * (1 - self.percent_correct)
        sd_scale = (prop_product / self.n) ** 0.5
        if not model.all_linear:
            sd_scale *=(2 / 8) ** 0.5 # (0.5 - 0.5 / (d * math.pi)) ** 0.5
        z *= sd_scale

        # z = norm.ppf(1 - bp ** 0.5)
        global first
        if first:
            print('ZZZZs')
            print(norm.ppf(1 - bp))
            print(norm.ppf(1 - bp ** 0.5))
            first = False
        return z, [var ** 0.5 for var in variances]



class NullRegularizationState:

    def exceeds_threshold(self):
        return False

    def __str__(self):
        return "N/A"


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
