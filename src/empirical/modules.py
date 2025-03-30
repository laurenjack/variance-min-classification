import torch
import torch.nn as nn
from scipy.stats import norm

from src.hyper_parameters import DataParameters, HyperParameters


def calculate_z(dp: DataParameters, hp: HyperParameters):
    d = hp.sizes[0]
    if hp.all_linear:
        num_in = d  - 1 # - 1 # sum(self.sizes[:-1])
        if hp.is_bias:
            num_in += 1
    else:
        num_in = hp.sizes[0] * hp.sizes[1]
    bp = (1 - hp.desired_success_rate ** (1 / num_in)) / 2
    z = norm.ppf(1 - bp)
    # z *= sd_scale
    return z


class CustomLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, gradient_box):
        """
        Forward pass for a linear layer (without bias).
        """
        ctx.save_for_backward(x, weight)
        ctx.gradient_box = gradient_box
        # Standard linear forward: output = input @ weight^T
        output = x.matmul(weight.t())
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass:
          Computes grad_input and a regularized grad_weight.
        """
        input, weight = ctx.saved_tensors
        gradient_box = ctx.gradient_box
        z = gradient_box['z']
        # Save the SigmoidBCE gradient, if we are at the final layer. We will back-propagate ones instead,
        # which will allow us to calculate both the grad and the variance grad.
        if gradient_box['delta'] is None:
            gradient_box['delta'] = grad_output
            grad_output = torch.ones_like(grad_output)
        delta = gradient_box['delta']
        expected_delta = gradient_box['expected_delta']

        # Gradient with respect to input (for further backprop)
        grad_input = grad_output.matmul(weight)
        n = grad_output.shape[0]

        # Gradient for weight
        # grad_weight = grad_output.t().matmul(input)
        g = (delta * grad_output).t().matmul(input)

        # Expected gradient at zero
        out_dim = grad_output.shape[1]
        in_dim = input.shape[1]
        elementwise = grad_output.view(n, out_dim, 1) * input.view(n, 1, in_dim)
        # TODO(Jack) deal with the zero case
        n_star = torch.sum(elementwise != 0, dim=0)
        eg = (expected_delta * grad_output).t().matmul(input) / (n + 1)
        eg2 = ((expected_delta * grad_output) ** 2).t().matmul(input ** 2) / (n + 1)
        v = (eg2 - eg ** 2) / (n + 1) # (n - 1)
        sd = torch.sqrt(v)
        reg = 1.0 * torch.sign(weight) * sd
        if gradient_box['report']:
            gradient_box['grads_at_zero'].append(torch.abs(eg))
            gradient_box['regs'].append(sd)
            # When reporting we shouldn't return a gradient for the weights, just backpropagate for reporting the other
            # layers
            return grad_input, None, None
        grad_weight = g + reg
        # do_update = ((torch.abs(g) - sd) > 0).float()
        # grad_weight = do_update * g
        # grad_weight = (g + reg) / (sd + 1e-8)
        # grad_weight = torch.abs(weight) * ((g + reg) / (sd * n ** 0.5 + + 1e-8) + 0.03 * weight)
        # grad_weight = g / (sd * n ** 0.5 + 1e-8)
        # thresh = 2.0 if gradient_box['epoch'] >= 100 else 0
        # epoch = gradient_box['epoch']
        # confidence = torch.abs(eg) / (sd + 1e-8)
        # mask = nn.functional.softmax(confidence) * out_dim * in_dim
        # mask = (torch.abs(eg) > thresh * sd).float()
        # grad_weight = mask * g
        # if epoch > 200:
        #     grad_weight = g * mask
        # else:
        #     grad_weight = g

        # No gradient for the gradient box, which is just used to store tensors
        return grad_input, grad_weight, None


class StandardLinearFunction(torch.autograd.Function):
    """
    This is basically identical to nn.Linear's forward and backward functions. It exists so that both the new and
    old implementations can share the exact sa,e CustomLinear class with the same initialization logic, so that we can
    compare them with the exact same initialization and training (when keeping the seed constant)
    """
    @staticmethod
    def forward(ctx, x, weight, gradient_box):
        """
        Forward pass for a linear layer (without bias).
        """
        ctx.save_for_backward(x, weight)
        ctx.gradient_box = gradient_box
        # Standard linear forward: output = input @ weight^T
        output = x.matmul(weight.t())
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass:
          Computes grad_input and a regularized grad_weight.
        """
        input, weight = ctx.saved_tensors
        grad_input = grad_output.matmul(weight)
        grad_weight = grad_output.t().matmul(input)
        # No gradient for the gradient box, which is just used to store tensors
        return grad_input, grad_weight, None


class CustomLinear(nn.Module):
    def __init__(self, in_features, out_features, reg_type, implementation, gradient_box):
        super().__init__()
        self.reg_type = reg_type
        self.implementation = implementation
        self.in_features = in_features
        self.out_features = out_features
        self.gradient_box = gradient_box
        # Initialize the weight (similar to nn.Linear)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = None

    def forward(self, x):
        if self.implementation == 'new' and (self.reg_type == 'L1' or self.gradient_box['report']):
            return CustomLinearFunction.apply(x, self.weight, self.gradient_box)
        else:
            return StandardLinearFunction.apply(x, self.weight, self.gradient_box)


class Mlp(nn.Module):
    def __init__(self, dp, hp):
        """
        The MLP is built from a sequence of Linear layers (no bias) with ReLU
        activations (unless all_linear is True).

        If hp.reg_type == "L1", custom units (with our modified backward) are used;
        if hp.reg_type == "NoReg", standard nn.Linear units (with bias=False) are used.
        Any other value for hp.reg_type raises a ValueError.
        """
        super().__init__()
        self.n = dp.n
        self.sizes = hp.sizes
        self.reg_type = hp.reg_type
        self.implementation = hp.implementation
        self.all_linear = hp.all_linear
        self.gradient_box = {'z': calculate_z(dp, hp), 'report': False}

        self.layers = nn.ModuleList()
        self.linears = []  # to store the raw linear modules
        self.activations = []  # to optionally record activations

        # Build the network: for each adjacent pair in sizes, create a layer.
        for num_input, num_output in zip(hp.sizes[:-2], hp.sizes[1:-1]):
            self._append_to_layer(num_input, num_output, is_relu=not hp.all_linear)
        self._append_to_layer(hp.sizes[-2], hp.sizes[-1], is_relu=False)

        # Initialize weights for each layer using kaiming_uniform.
        for o, layer in enumerate(self.layers):
            # If the layer is a Sequential, its first element is the Linear module.
            linear_module = layer[0] if isinstance(layer, nn.Sequential) else layer
            # Use 'linear' non-linearity for the first layer or if all_linear
            non_linearity = 'relu'
            if o == 0 or hp.all_linear:
                non_linearity = 'linear'
            nn.init.kaiming_uniform_(linear_module.weight, nonlinearity=non_linearity)


    def _append_to_layer(self, num_input, num_output, is_relu: bool):
        """
        Append a new layer to the network. If there is a regularizer defined, use the
        custom linear module; otherwise, use the standard nn.Linear (with bias=False).
        """

        linear = CustomLinear(num_input, num_output, self.reg_type, self.implementation, self.gradient_box)
         # linear = nn.Linear(num_input, num_output, bias=False)
        # else:
        #     raise ValueError(f"Unrecognized reg_type {self.reg_type}")
        self.linears.append(linear)
        if is_relu:
            self.layers.append(nn.Sequential(linear, nn.ReLU()))
        else:
            self.layers.append(linear)

    def forward(self, x):
        a = x
        self.activations = []
        for layer in self.layers:
            a = layer(a)
            self.activations.append(a)
        return a