from abc import ABC, abstractmethod

import torch
from torch import nn
from torch.nn import functional as F

NORM_EPSILON = 0.00000001 # Stops dividing by zero when normalizing the weights and biases.


class MultiModule(nn.Module, ABC):
    """Represents a neural network module with multiple sub models, in the context of variance minimization"""

    def __init__(self):
        super().__init__()

    @abstractmethod
    def get_params(self):
        return None


class SharedMagnitude(MultiModule):

    def __init__(self, is_bias):
        super().__init__()
        self.is_bias = is_bias

    def get_params(self):
        return self.weight, self.bias

    @property
    def weight(self):
        return get_applied(self.w, self.weight_mag)

    @property
    def bias(self):
        if self.is_bias:
            return get_applied(self.b, self.bias_mag)


def norm2(tensor):
    return torch.sum(tensor ** 2) ** 0.5


def create_normed_params(weight, bias):
    weight_mag = norm2(weight)
    bias_mag = norm2(bias)
    return nn.Parameter(weight), nn.Parameter(bias), nn.Parameter(weight_mag), nn.Parameter(bias_mag)


def create_normed_param(tensor):
    tensor_mag = norm2(tensor)
    log_mag = torch.log(tensor_mag)
    return nn.Parameter(tensor), nn.Parameter(log_mag)


def create_batch_norm_params(num_features):
    weight = torch.ones(num_features)
    bias = torch.zeros(num_features)
    return create_normed_params(weight, bias)


def create_conv_params(c_in, out, k):
    scaler = 1 / (c_in * k ** 2) ** 0.5
    weight = 2 * (torch.rand(out, c_in, k, k, requires_grad=True) - 0.5) * scaler
    bias = 2 * (torch.rand(out, requires_grad=True) - 0.5) * scaler
    return create_normed_params(weight, bias)


def create_linear_params(in_dim, out):
    weight = create_weight(in_dim, out)
    bias = create_bias(out)
    return create_normed_params(weight, bias)


def create_weight(in_dim, out):
    weight = 2 * (torch.rand(out, in_dim, requires_grad=True) - 0.5) / in_dim ** 0.5
    return create_normed_param(weight)


def create_bias(out):
    bias = 2 * (torch.rand(out, requires_grad=True) - 0.5) / out ** 0.5
    return create_normed_param(bias)


def get_applied(param, param_magnitude):
    norm = torch.norm(param).detach()
    normed_param = param / (norm + NORM_EPSILON)
    return normed_param * torch.exp(param_magnitude)


class Linear(SharedMagnitude):

    def __init__(self, c_in, out, bias=True):
        super().__init__(bias)
        self.w, self.weight_mag = create_weight(c_in, out)
        if bias:
            self.b, self.bias_mag = create_bias(out)

    def forward(self, x):
        return F.linear(x, weight=self.weight, bias=self.bias)


class BatchNorm2d(SharedMagnitude):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.weight, self.bias, self.weight_mag, self.bias_mag = create_batch_norm_params(num_features)
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        w, b = self.get_params()
        y = F.batch_norm(x, self.running_mean, self.running_var, weight=w, bias=b, training=self.training,
                         momentum=self.momentum, eps=self.eps)
        return y
