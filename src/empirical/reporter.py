import torch


def show_greatest_grad(model, x, y, dp, hp):
    grads_at_zero, regs = grads_at_zero_and_regs(model, x, y, dp, hp)
    return greatest_grad_calc(grads_at_zero, regs, true_d=dp.true_d)


def grads_at_zero_and_regs(model, x, y, dp, hp):
    assign_expected_delta(model, x, y, dp, hp)
    model.gradient_box['report'] = True
    model.gradient_box['grads_at_zero'] = []
    model.gradient_box['regs'] = []
    output = model(x)
    n = x.shape[0]
    # The actual back-propagated signal does not matter because we are just collecting the grads at zero and regs
    output.backward(gradient = y.float().view(n, 1))
    model.gradient_box['report'] = False
    grads_at_zero = model.gradient_box['grads_at_zero']
    grads_at_zero.reverse()
    regs = model.gradient_box['regs']
    regs.reverse()
    return grads_at_zero, regs



def greatest_grad_calc(grads_at_zero, regs, true_d=0):
    # Ignore the non-noisy dimensions
    grads_at_zero = [grads_at_zero[0][:, true_d:]]
    regs = [regs[0][:, true_d:]]
    has_gradient_greater_reg = False
    greatest_delta = -10000000
    greatest_deltas_grad = None
    greatest_deltas_reg = None
    for manual_grad, reg in zip(grads_at_zero, regs):
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


def assign_expected_delta(model, x, y, dp, hp, epoch=0):
    # The delta in gradient box is used to keep track of the back-propagated gradient, by assuming that it's not set
    # everytime we call backwards. So we need to always reset it to None
    model.gradient_box['epoch'] = epoch
    model.gradient_box['delta'] = None
    n = x.shape[0]
    y_shift = y.view(n, 1) * 2.0 - 1
    prop_product = dp.percent_correct * (1 - dp.percent_correct)
    sd_scale = prop_product ** 0.5 # / dp.n  # TODO(Jack) consider batch_size
    if not hp.all_linear:
        sd_scale *= hp.relu_bound ** 0.5  # (0.5 - 0.5 / (d * math.pi)) ** 0.5
    model.gradient_box['expected_delta'] = sd_scale * y_shift
