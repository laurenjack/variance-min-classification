import torch

from jl.posterior_minimizer.empirical.reporter import greatest_grad_calc, RegularizationState
from jl.posterior_minimizer.hyper_parameters import DataParameters, HyperParameters
from jl.posterior_minimizer import variance as v
from jl.posterior_minimizer.variance import Variance


ZERO_THRESHOLD = 0.0001


def create(variance: v.Variance, dp: DataParameters, hp: HyperParameters):
    if hp.reg_type == "L1":
        reg_constructor = L1
    elif hp.reg_type == "L2":
        reg_constructor = L2
    elif hp.reg_type == "NoReg":
        reg_constructor = NoReg
    else:
        raise ValueError(f"Unsupported Regualrizer: {hp.reg_type}")
    return reg_constructor(variance, dp, hp)


def get_zero_state(model, x, y, zero_threshold, true_d=0):
    # Zero out the true dimensions, so we can see the regularizer had the intended effect
    x = x.clone()
    x[:, :true_d] = 0.0
    d = x.shape[1]
    # means = model(torch.eye(d))
    # squared_mag = torch.sum(means ** 2) ** 0.5
    z = model(x)
    squared_mag = (torch.sum(z ** 2) / d) ** 0.5
    return RegularizationState(squared_mag, zero_threshold, "Squared Weight Magnitude")



class DirectReg:
    """
    Modifies the gradients of a model directly, to apply a regularizer.
    """

    def __init__(self, zero_threshold, variance: Variance, dp: DataParameters, hp: HyperParameters):
        self.zero_threshold = zero_threshold
        self.variance = variance
        self.sizes = hp.sizes
        self.desired_success_rate = hp.desired_success_rate
        self.relu_bound = hp.relu_bound
        self.n = dp.n
        self.percent_correct = dp.percent_correct
        self.sigmoid = torch.nn.Sigmoid()

    def regularize_param(self, W, reg):
        pass

    def apply(self, model, x, y, epoch):
        regs, bias_regs = self.variance.calculate(model, x, y)
        for l, (reg, bias_reg) in enumerate(zip(regs, bias_regs)):
            W = model.linears[l].weight
            b = model.linears[l].bias
            self.regularize_param(W, reg)
            if b is not None:
                self.regularize_param(b, bias_reg)

    def get_max_gradient(self, model, x, y, true_d=0):
        w_regs, bias_regs = self.variance.calculate(model, x, y)
        w_grads, bias_grads = v.grad_at_zero(model, x, y, self.percent_correct)
        # grads = [w_grads[0][:, true_d:]]
        # regs = [w_regs[0][:, true_d:]]
        # if model.is_bias:
        #     grads += bias_grads[:1]
        #     regs += bias_regs[:1]
        # if isinstance(self, L2):
        #     # TODO(Jack) handle biases
        #     grads = [torch.sum(reg_grad ** 2.0, dim=1) ** 0.5 for reg_grad in grads]
        #     regs = [torch.sum(manual_grad ** 2.0, dim=1) ** 0.5 for manual_grad in regs]
        return greatest_grad_calc(w_grads, w_regs, true_d = true_d)

    def get_zero_state(self, model, x, y, true_d=0):
        return get_zero_state(model, x, y, self.zero_threshold, true_d)


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
        super().__init__(variance.scale_calc(), variance, dp, hp)


class L1(DirectReg):

    def __init__(self, variance: Variance, dp: DataParameters, hp: HyperParameters):
        super().__init__(0.0001, variance, dp, hp)

    def regularize_param(self, W, reg):
        w = W.detach()
        grad = W.grad
        reg *= torch.sign(w)
        updated_grad = torch.abs(w) * (grad + reg) # * #/ (sd + 1e-8) + 0.01 * w) # self.n ** 0.5 *
        W.grad = updated_grad


class L2(DirectReg):

    def __init__(self, variance: Variance, dp: DataParameters, hp: HyperParameters):
        post_constant = variance.scale_calc()
        zero_threshold = post_constant / (1 + post_constant)
        super().__init__(zero_threshold, variance, dp, hp)

    def regularize_param(self, W, reg):
        w = W.detach()
        grad = W.grad
        updated_grad = grad + reg * w
        W.grad = updated_grad

    def get_zero_state(self, model, x, y, true_d=0):
        # Zero out the true dimensions, so we can see the regularizer had the intended effect
        x = x.clone()
        x[:, :true_d] = 0.0
        d = x.shape[1]
        means = model(torch.eye(d))
        squared_mag = torch.sum(means ** 2) ** 0.5
        return RegularizationState(squared_mag, self.zero_threshold, "Squared Weight Magnitude")


