import torch
from scipy.stats import norm, chi2

from src.hyper_parameters import DataParameters, HyperParameters


def regularize(model, regularizer, x, y):
        # regs = regularizer.variance_grad(model)
        regs = regularizer.empirical_variance_grad(model, x, y)
        for l, reg in enumerate(regs):
            _, reg_grad = reg
            model.linears[l].weight.grad += reg_grad


def create(dp: DataParameters, hp: HyperParameters):
    if hp.reg_type == "L1":
        num_in = sum(hp.sizes[:-1])
        var_scaler = 4  # 1
        bp = (1 - hp.desired_success_rate ** (1 / num_in)) / 2
        # prop_product = dp.percent_correct * (1 - dp.percent_correct)
        var = var_scaler / dp.n  # * prop_product
        sd = var ** 0.5
        # post_constant = norm.ppf(1 - bp, scale=sd) - hp.reg_epsilon
        post_constant = norm.ppf(1 - bp)
        return L1(post_constant, hp.desired_success_rate, dp.n)
    elif hp.reg_type == "InverseMagnitudeL2":
        reg_constructor = InverseMagnitudeL2
    elif hp.reg_type == "L2":
        reg_constructor = L2
    elif hp.reg_type == "NoReg":
        reg_constructor = NoReg
    elif hp.reg_type == "GradientWeighted":
        reg_constructor = GradientWeighted
    elif hp.reg_type == "GradientWeightedNormed":
        reg_constructor = GradientWeightedNormed
    else:
        raise ValueError(f"Unsupported Regualrizer: {hp.reg_type}")
    # This is the calculation of the posterior constant for all regularizers which regularize over the whole node
    num_out = len(hp.sizes) - 1 # sum(hp.sizes[:-1]) # sum(hp.sizes[1:])
    bp = (1 - hp.desired_success_rate ** (1 / num_out))
    post_constant = chi2.ppf(1 - bp, dp.d) - hp.reg_epsilon
    post_constant *= 4 / dp.n
    post_constant = post_constant ** 0.5
    return reg_constructor(post_constant, hp.desired_success_rate, dp.n)


class DirectReg:
    """
    Modifies the gradients of a model directly, to apply a regularizer.
    """

    def __init__(self, post_constant, zero_threshold, desired_success_rate, n):
        self.post_constant = post_constant
        self.zero_threshold = zero_threshold
        self.desired_success_rate = desired_success_rate
        self.n = n
        self.sigmoid = torch.nn.Sigmoid()

    def weight_scaling(self, W):
        raise NotImplementedError("This is an abstract regularizer, please use a subclass")

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
        return RegularizationState(squared_mag, self.zero_threshold, "Squared Weight Magnitude")

    # def grad_at_zero(self, model, x, y, point_level=False):
    #     weights = [linear.weight for linear in model.linears]
    #     y_shift = y * -2 + 1
    #     n, d = x.shape
    #     # The three commented out lines below apply to the SigmoidBxeTrainer
    #     # forward_product = x.t()
    #     # backward_product = 0.5 * y_shift / n
    #     # backward_product = backward_product.view(n, 1)
    #     backward_product = torch.ones(1, 1)
    #     forward_product = 2 * torch.mean(x * y_shift.view(n, 1), dim=0, keepdim=True).t()
    #     return self.manual_grad_calc(weights, forward_product, backward_product, point_level=point_level)

    def grad_at_zero(self, model, x, y, point_level=False):
        weights = [linear.weight for linear in model.linears]
        # y_shift = y * -2.0 + 1
        n, d = x.shape
        # The three commented out lines below apply to the SigmoidBxeTrainer
        # forward_product = x.t()
        # backward_product = 0.5 * y_shift / n
        # backward_product = backward_product.view(n, 1)
        y_shift = y.view(n, 1) * 2.0 - 1
        forward_product = -(2 * x * y_shift).t() / n  # (d, n)
        backward_product = torch.ones(n, 1)
        return self.manual_grad_calc(weights, forward_product, backward_product, point_level=point_level)


    def variance_grad(self, model):
        weights = [linear.weight for linear in model.linears]
        d = weights[0].shape[1]
        forward_product = torch.eye(d)
        backward_product = torch.ones(1, 1)
        return self.manual_grad_calc(weights, forward_product, backward_product, is_variance=True)

    def empirical_variance_grad(self, model, x, y):
        weights = [linear.weight.detach() for linear in model.linears]
        n, d = x.shape
        y_shift = y * 2.0 - 1
        y_shift = y_shift.view(n, 1)
        mean = model(torch.eye(d))
        target = y_shift.view(n, 1) @ mean.t()
        fp = (target - 2 * x).t() / n # (d, n)
        bp = y_shift  # (n, 1)
        point_level_grads = self.manual_grad_calc(weights, fp, bp, point_level=True)  # list[(n, dl+1, dl]
        # point_level_grads = self.grad_at_zero(model, x, y, point_level=True)
        mean_grads = [torch.sum(plg, dim=0, keepdim=True) for plg in point_level_grads]
        # var_grads = [torch.sum((plg - mg) ** 2, dim=0) / n / (n - 1) for plg, mg in zip(point_level_grads, mean_grads)]
        var_grads = [torch.mean((n * plg - mg) ** 2, dim=0) / (n - 1) for plg, mg in zip(point_level_grads, mean_grads)]
        # var_grads = [torch.sum(plg ** 2, dim=0) for plg, mg in zip(point_level_grads, mean_grads)]
        sd_grads = [torch.sign(W) * self.post_constant * var_grad ** 0.5 for W, var_grad in zip(weights, var_grads)]
        return [(None, sd_grad) for sd_grad in sd_grads]

    def manual_grad_calc(self, weights, forward_product, backward_product, is_variance=False, point_level=False):
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
                    grad = self.reg_at_layer(weights[l], fp, bp, num_layers)
                else:
                    grad = bp.t() @ fp.t()
            grads.append(grad)
        return grads

    def reg_at_layer(self, weight, fp, bp, num_layers):
        dl, _ = fp.shape
        bp = bp ** 2
        fp = fp ** 2
        fp = torch.sum(fp, dim=1, keepdim=True)
        # if isinstance(self, InverseMagnitudeL2):
        #     fp = torch.mean(fp, dim=0, keepdim=True)
        # The inner dimension is always just 1 here
        grad = bp.t() @ fp.t()
        # if isinstance(self, InverseMagnitudeL2):
        #     grad = grad.repeat(1, dl)
        unscaled_grad = grad ** 0.5
        scaled_grad = self.post_constant * unscaled_grad * self.weight_scaling(weight)
        return unscaled_grad, scaled_grad

    def get_greatest_manual_grad(self, model, x, y):
        # regs = self.variance_grad(model)
        regs = self.empirical_variance_grad(model, x, y)
        reg_grads = [scaled_grad for _, scaled_grad in regs]
        manual_grads = self.grad_at_zero(model, x, y)
        if isinstance(self, InverseMagnitudeL2):
            reg_grads = [torch.sum(reg_grad ** 2.0, dim=1) ** 0.5 for reg_grad in reg_grads]
            manual_grads = [torch.sum(manual_grad ** 2.0, dim=1) ** 0.5 for manual_grad in manual_grads]
        has_gradient_greater_reg = False
        greatest_delta = -10000000
        greatest_deltas_grad = None
        greatest_deltas_reg = None
        for manual_grad, reg in zip(manual_grads, reg_grads):
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




class NoReg(DirectReg):

    def __init__(self, post_constant, desired_success_rate, n):
        super().__init__(post_constant, post_constant, desired_success_rate, n)

    def weight_scaling(self, W):
        return 0.0 * W.detach()


class L2(DirectReg):

    def __init__(self, post_constant, desired_success_rate, n):
        zero_threshold = post_constant / (1 + post_constant)
        super().__init__(post_constant, zero_threshold, desired_success_rate, n)

    def weight_scaling(self, W):
        return W.detach()

    # def get_max_gradient(self, model, x, y):
    #     return self.get_greatest_manual_grad(model, x, y)

    def get_zero_state(self, model, x, y):
        d = x.shape[1]
        means = model(torch.eye(d))
        squared_mag = torch.sum(means ** 2) ** 0.5
        return RegularizationState(squared_mag, self.zero_threshold, "Squared Weight Magnitude")


class GradientWeighted(DirectReg):

    def __init__(self, post_constant, desired_success_rate, n):
        zero_threshold = (1 - post_constant) * post_constant
        if zero_threshold < 0:
            print("WARNING - zero_threshold lower than zero, expect zero weights everytime")
        super().__init__(post_constant, zero_threshold, desired_success_rate, n)

    def weight_scaling(self, W):
        w = W.detach()
        scales = w - W.grad
        return torch.sign(w) * torch.abs(scales)


class L1(DirectReg):

    ZERO_THRESHOLD = 0.0005

    def __init__(self, post_constant, desired_success_rate, n):
        super().__init__(post_constant, L1.ZERO_THRESHOLD, desired_success_rate, n)

    def weight_scaling(self, W):
        W = W.detach()
        return torch.sign(W)

    def get_max_gradient(self, model, x, y):
        return self.get_greatest_manual_grad(model, x, y)

    # def get_zero_state(self, model, x, y):
    #     logit_sum = get_scaled_logit_sum(model, x, y)
    #     return RegularizationState(logit_sum, L1.ZERO_THRESHOLD, "Mean Logit Sum")


class InverseMagnitudeL2(DirectReg):

    ZERO_THRESHOLD = 0.005

    def __init__(self, post_constant, desired_success_rate, n):
        super().__init__(post_constant, InverseMagnitudeL2.ZERO_THRESHOLD, desired_success_rate, n)

    def weight_scaling(self, W):
        w = W.detach()
        return w / torch.norm(w)

    def get_max_gradient(self, model, x, y):
        return self.get_greatest_manual_grad(model, x, y)

    def reg_at_layer(self, weight, fp, bp, num_layers):
        dl, _ = fp.shape
        bp = bp ** 2
        fp = fp ** 2
        # Sum across dim=1 to get variance constant for each input
        fp = torch.sum(fp, dim=1)
        # sum across dim=0 to produce the single L2-style regularizer for this layer
        mean = torch.sum(fp)
        variance = torch.sum(fp ** 2)
        # Satterthwaite Approximation of generalized chi-square as chi-squared
        f = mean ** 2 / variance
        scale = variance / mean
        percentile = (1 - self.desired_success_rate ** (1 / num_layers))
        post_constant = chi2.ppf(1 - percentile, f)
        post_constant *= 4 / self.n * scale

        # The inner dimension is always just 1 here
        grad = post_constant * bp.t()
        grad = grad.repeat(1, dl)
        unscaled_grad = grad ** 0.5
        scaled_grad = unscaled_grad * self.weight_scaling(weight)
        return unscaled_grad, scaled_grad


class GradientWeightedNormed(DirectReg):
    ZERO_THRESHOLD = 0.005

    def __init__(self, post_constant, desired_success_rate):
        super().__init__(post_constant, GradientWeightedNormed.ZERO_THRESHOLD, desired_success_rate)

    def weight_scaling(self, W):
        w = W.detach()
        scales = w - W.grad
        scales = torch.sign(w) * torch.abs(scales)
        return scales / torch.norm(scales)


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






# def get_scaled_logit_sum(model, x, y):
#     n = x.shape[0]
#     y_shift = y * 2 - 1
#     z = x.t()
#     for linear in model.linears:
#         W = linear.weight
#         z = W @ z
#     logit_sum = torch.sum(y_shift.view(1, n) * z)
#     return logit_sum / n
