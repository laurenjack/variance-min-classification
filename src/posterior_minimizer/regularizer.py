import torch
from scipy.stats import norm, chi2

from src.hyper_parameters import DataParameters, HyperParameters


def regularize(model, regularizer, x, y):
        regs = regularizer.variance_grad(model)
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
        post_constant = norm.ppf(1 - bp, scale=sd) - hp.reg_epsilon
        return L1(post_constant)
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
    num_out = sum(hp.sizes[1:])
    bp = (1 - hp.desired_success_rate ** (1 / num_out))
    post_constant = chi2.ppf(1 - bp, dp.d) - hp.reg_epsilon
    post_constant *= 4 / dp.n
    post_constant = post_constant ** 0.5
    return reg_constructor(post_constant)


class DirectReg:
    """
    Modifies the gradients of a model directly, to apply a regularizer.
    """

    def __init__(self, post_constant, zero_threshold):
        self.post_constant = post_constant
        self.zero_threshold = zero_threshold
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

    def grad_at_zero(self, model, x, y, point_level=False):
        weights = [linear.weight for linear in model.linears]
        y_shift = y * -2 + 1
        n, d = x.shape
        # The three commented out lines below apply to the SigmoidBxeTrainer
        # forward_product = x.t()
        # backward_product = 0.5 * y_shift / n
        # backward_product = backward_product.view(n, 1)
        backward_product = torch.ones(1, 1)
        forward_product = 2 * torch.mean(x * y_shift.view(n, 1), dim=0, keepdim=True).t()
        return self.manual_grad_calc(weights, forward_product, backward_product, point_level=point_level)


    def variance_grad(self, model):
        weights = [linear.weight for linear in model.linears]
        d = weights[0].shape[1]
        forward_product = torch.eye(d)
        backward_product = self.post_constant * torch.ones(1, 1)
        return self.manual_grad_calc(weights, forward_product, backward_product, is_variance=True)


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
        for l in range(len(weights)):
            fp = forward_products[l]
            bp = backward_products[-(l + 1)]
            dl, n = fp.shape
            _, dl1 = bp.shape
            if point_level:
                grad = bp.view(1, n, dl1) * fp.view(dl, n, 1)
                grad = grad.permute(1, 2, 0)  # Put the data dim first
            else:
                if is_variance:
                    grad = self.reg_at_layer(weights[l], fp, bp)
                else:
                    grad = bp.t() @ fp.t()
            grads.append(grad)
        return grads

    def reg_at_layer(self, weight, fp, bp):
        dl, _ = fp.shape
        bp = bp ** 2
        fp = fp ** 2
        if isinstance(self, L2):
            # L2 - style regularizers are the same w.r.t to the input, at each layer
            fp = torch.sum(fp, dim=0, keepdim=True)
        fp = torch.sum(fp, dim=1, keepdim=True)
        # The inner dimension is always just 1 here
        grad = bp.t() @ fp.t()
        if isinstance(self, L2):
            grad = grad.repeat(1, dl)
        unscaled_grad = grad ** 0.5
        scaled_grad = unscaled_grad * self.weight_scaling(weight)
        return unscaled_grad, scaled_grad

    def get_greatest_manual_grad(self, model, x, y):
        regs = self.variance_grad(model)
        reg_grads = [unscaled_reg_grad for unscaled_reg_grad, _ in regs]
        manual_grads = self.grad_at_zero(model, x, y)
        if isinstance(self, L2):
            regs = [torch.sum(reg ** 2.0, dim=1) ** 0.5 for reg in reg_grads]
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

    def __init__(self, post_constant):
        super().__init__(post_constant, post_constant)

    def weight_scaling(self, W):
        return 0.0 * W.detach()


class L2(DirectReg):

    def __init__(self, post_constant):
        zero_threshold = post_constant / (1 + post_constant)
        super().__init__(post_constant, zero_threshold)

    def weight_scaling(self, W):
        return W.detach()

    def get_max_gradient(self, model, x, y):
        return self.get_greatest_manual_grad(model, x, y)

    def get_zero_state(self, model, x, y):
        d = x.shape[1]
        means = model(torch.eye(d))
        squared_mag = torch.sum(means ** 2) ** 0.5
        return RegularizationState(squared_mag, self.zero_threshold, "Squared Weight Magnitude")


class GradientWeighted(DirectReg):

    def __init__(self, post_constant):
        zero_threshold = (1 - post_constant) * post_constant
        if zero_threshold < 0:
            print("WARNING - zero_threshold lower than zero, expect zero weights everytime")
        super().__init__(post_constant, zero_threshold)

    def weight_scaling(self, W):
        w = W.detach()
        scales = w - W.grad
        return torch.sign(w) * torch.abs(scales)


class L1(DirectReg):

    ZERO_THRESHOLD = 0.0005

    def __init__(self, post_constant):
        super().__init__(post_constant, L1.ZERO_THRESHOLD)

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

    def __init__(self, post_constant):
        super().__init__(post_constant, InverseMagnitudeL2.ZERO_THRESHOLD)

    def weight_scaling(self, W):
        w = W.detach()
        return w / torch.norm(w)


class GradientWeightedNormed(DirectReg):
    ZERO_THRESHOLD = 0.005

    def __init__(self, post_constant):
        super().__init__(post_constant, InverseMagnitudeL2.ZERO_THRESHOLD)

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
