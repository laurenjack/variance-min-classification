import torch
from scipy.stats import norm, chi2

from src.hyper_parameters import DataParameters, HyperParameters


def regularize(model, regularizer, x, y):
        regs = variance_grad(model, regularizer)
        for l, reg in enumerate(regs):
            model.linears[l].weight.grad += reg


def create(dp: DataParameters, hp: HyperParameters):
    if hp.reg_type == "L1":
        num_nodes = sum(hp.sizes[:-1])
        var_scaler = 4
        reg_constructor = L1
    elif hp.reg_type == "InverseMagnitudeL2":
        # num_nodes = 1
        # var_scaler = dp.d
        # reg_constructor = InverseMagnitudeL2
        bp = (1 - hp.desired_success_rate)
        post_constant = chi2.ppf(1 - bp, dp.d) - hp.reg_epsilon
        post_constant *= 4 / dp.n
        post_constant = post_constant ** 0.5
        return InverseMagnitudeL2(post_constant)
    elif hp.reg_type == "L2":
        bp = (1 - hp.desired_success_rate)
        post_constant = chi2.ppf(1 - bp, dp.d) - hp.reg_epsilon
        post_constant *= 4 / dp.n
        post_constant = post_constant ** 0.5
        return L2(post_constant)
    elif hp.reg_type == "DirectReg":
        # num_nodes = 1
        # var_scaler = 2 * dp.d
        # reg_contructor = DirectReg
        bp = (1 - hp.desired_success_rate)
        post_constant = chi2.ppf(1 - bp, dp.d) - hp.reg_epsilon
        post_constant *= 4 / dp.n
        post_constant = post_constant ** 0.5
        return DirectReg(post_constant)
    elif hp.reg_type == "GradientWeighted":
        bp = (1 - hp.desired_success_rate)
        post_constant = chi2.ppf(1 - bp, dp.d) - hp.reg_epsilon
        post_constant *= 4 / dp.n
        post_constant = post_constant ** 0.5
        return GradientWeighted(post_constant)
    elif hp.reg_type == "GradientWeightedNormed":
        bp = (1 - hp.desired_success_rate)
        post_constant = chi2.ppf(1 - bp, dp.d) - hp.reg_epsilon
        post_constant *= 4 / dp.n
        post_constant = post_constant ** 0.5
        return GradientWeightedNormed(post_constant)
    else:
        raise ValueError(f"Unsupported Regualrizer: {hp.reg_type}")
    bp = (1 - hp.desired_success_rate ** (1 / num_nodes)) / 2
    # prop_product = dp.percent_correct * (1 - dp.percent_correct)
    var = var_scaler / dp.n  # * prop_product
    sd = var ** 0.5
    post_constant = norm.ppf(1 - bp, scale=sd) - hp.reg_epsilon
    return reg_constructor(post_constant)


class DirectReg:
    """
    Modify the gradients of a model directly, to apply a regularizer.
    """

    def __init__(self, post_constant, threshold=None):
        self.post_constant = post_constant
        self.threshold = threshold
        if threshold is None:
            self.threshold = post_constant
        self.sigmoid = torch.nn.Sigmoid()

    def weight_scaling(self, W):
        return 0.0 * W.detach()

    def get_max_gradient(self, model, x, y):
        n = x.shape[0]
        y_shift = y * 2 - 1
        means = torch.mean(x * y_shift.view(n, 1), dim=0)
        squared_mag = torch.sum((2 * means) ** 2) ** 0.5
        return RegularizationState(squared_mag, self.post_constant, "Actual Squared Mean Magnitude")

    def get_zero_state(self, model, x, y):
        d = x.shape[1]
        means = model(torch.eye(d))
        squared_mag = torch.sum(means ** 2) ** 0.5
        return RegularizationState(squared_mag, self.threshold, "Squared Weight Magnitude")


class L1(DirectReg):

    ZERO_THRESHOLD = 0.0005

    def __init__(self, post_constant):
        super().__init__(post_constant)

    def weight_scaling(self, W):
        W = W.detach()
        return torch.sign(W)

    def get_max_gradient(self, model, x, y):
        return get_greatest_manual_grad(model, x, y, self)

    def get_zero_state(self, model, x, y):
        logit_sum = get_scaled_logit_sum(model, x, y)
        return RegularizationState(logit_sum, L1.ZERO_THRESHOLD, "Mean Logit Sum")


class InverseMagnitudeL2(DirectReg):

    ZERO_THRESHOLD = 0.005

    # def weight_scaling(self, W):
    #     W = W.detach()
    #     return W / torch.norm(W)

    def weight_scaling(self, W):
        wtf = W.detach() / torch.norm( W.detach())
        optimal_weight = W.detach() - W.grad.detach()
        scaling = optimal_weight / torch.norm(optimal_weight)
        return wtf

    def get_zero_state(self, model, x, y):
        d = x.shape[1]
        means = model(torch.eye(d))
        squared_mag = torch.sum(means ** 2) ** 0.5
        return RegularizationState(squared_mag, InverseMagnitudeL2.ZERO_THRESHOLD, "Squared Weight Magnitude")



class L2(DirectReg):

    def __init__(self, post_constant):
        threshold = post_constant / (1 + post_constant)
        super().__init__(post_constant, threshold)

    def weight_scaling(self, W):
        return W.detach()


class GradientWeighted(DirectReg):

    def __init__(self, post_constant):
        threshold = (1 - post_constant) * post_constant
        if threshold < 0:
            print("WARNING - threshold lower than zero, expect zero weights everytime")
        super().__init__(post_constant, threshold)

    def weight_scaling(self, W):
        w = W.detach()
        scales = w - W.grad
        return torch.sign(w) * torch.abs(scales)


class GradientWeightedNormed(DirectReg):
    ZERO_THRESHOLD = 0.005

    def weight_scaling(self, W):
        w = W.detach()
        scales = w  - W.grad
        scales = torch.sign(w) * torch.abs(scales)
        return scales / torch.norm(scales)

    def get_zero_state(self, model, x, y):
        d = x.shape[1]
        means = model(torch.eye(d))
        squared_mag = torch.sum(means ** 2) ** 0.5
        return RegularizationState(squared_mag, GradientWeightedNormed.ZERO_THRESHOLD, "Squared Weight Magnitude")


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


def grad_at_zero(model, x, y, point_level=False):
    weights = [linear.weight for linear in model.linears]
    y_shift = y * -2 + 1
    n, d = x.shape
    # The three commented out lines below apply to the SigmoidBxeTrainer
    # forward_product = x.t()
    # backward_product = 0.5 * y_shift / n
    # backward_product = backward_product.view(n, 1)
    backward_product = 2 * torch.mean(x * y_shift.view(n, 1), dim=0)
    return [backward_product]
    # return manual_grad_calc(weights, forward_product, backward_product, point_level=point_level)


def variance_grad(model, regularizer):
    weights = [linear.weight for linear in model.linears]
    d = weights[0].shape[1]
    forward_product = torch.eye(d)
    backward_product = regularizer.post_constant * torch.ones(d, 1)
    return manual_grad_calc(weights, forward_product, backward_product, regularizer=regularizer)


def manual_grad_calc(weights, forward_product, backward_product, regularizer=None, point_level=False):
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
    for l in range(len(weights)):
        fp = forward_products[l]
        bp = backward_products[-(l + 1)]
        if point_level:
            dl, n = fp.shape
            _, dl1 = bp.shape
            grad = bp.view(1, n, dl1) * fp.view(dl, n, 1)
            grad = grad.permute(1, 2, 0)  # Put the data dim first
        else:
            if regularizer:
                bp = bp ** 2
                fp = fp ** 2
            grad = bp.t() @ fp.t()
            if regularizer:
                grad = grad ** 0.5 * regularizer.weight_scaling(weights[l])
        grads.append(grad)

    return grads


def get_whole_node_grad(model, x, y, regularizer):
    grad = grad_at_zero(model, x, y)[0]
    reg = variance_grad(model, regularizer)[0]
    return RegularizationState(torch.norm(grad), torch.norm(reg), "Greatest Gradient Norm")



def get_greatest_manual_grad(model, x, y, regularizer):
    regs = variance_grad(model, regularizer)
    manual_grads = grad_at_zero(model, x, y)
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


def get_scaled_logit_sum(model, x, y):
    n = x.shape[0]
    y_shift = y * 2 - 1
    z = x.t()
    for linear in model.linears:
        W = linear.weight
        z = W @ z
    logit_sum = torch.sum(y_shift.view(1, n) * z)
    return logit_sum / n
