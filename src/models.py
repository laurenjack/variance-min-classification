import torch
from torch import nn
import torch.nn.functional as F

from src.custom_modules import NORM_EPSILON, MultiModule, SharedMagnitude, create_conv_params, Linear, BatchNorm2d

IS_PROP_VAR = True # when true, use regularization in proportion to variance, when False, use L2 regularization


def get_reg(tensor, base_tensor):
    w_mean = torch.mean(tensor, axis=0, keepdim=True)
    var = torch.mean((tensor - w_mean) ** 2, axis=0)
    mean = base_tensor + w_mean[0]
    sd = (var / (mean ** 2 + NORM_EPSILON)) ** 0.5
    return sd.detach() * base_tensor ** 2


class Sequential(nn.Sequential):

    def forward(self, x, j, is_variance):
        for module in self:
            x = module(x, j, is_variance)
        return x


# TODO(Jack) remove m, j and is_variance from this point on.
class ConvBlock(SharedMagnitude):
    def __init__(self, m, c_in, out, k, stride, padding=0, with_relu=False):
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.weight, self.bias, self.weight_mag, self.bias_mag = create_conv_params(m, c_in, out, k)
        # self.base_weight, self.base_bias, self.weight, self.bias = create_conv_params(c_in, out, k)
        self.batch_norm = BatchNorm2d(m, out)
        self.with_relu = with_relu

    def forward(self, x, j, is_variance):
        w, b = self.get_params(j, is_variance)
        a = F.conv2d(x, weight=w, bias=b, stride=self.stride, padding=self.padding)
        a = self.batch_norm(a, j, is_variance)
        if self.with_relu:
            a = F.relu(a)
        return a

    def reg_loss(self):
        return super().reg_loss() + self.batch_norm.reg_loss()

class ResidualBlock(nn.Module):
    def __init__(self, m, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = ConvBlock(m, in_channels, out_channels, 3, stride, padding=1, with_relu=True)
        self.conv2 = ConvBlock(m, out_channels, out_channels, 3, 1, padding=1, with_relu=False)
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels

    def forward(self, x, j, is_variance):
        residual = x
        out = self.conv1(x, j, is_variance)
        out = self.conv2(out, j, is_variance)
        if self.downsample:
            residual = self.downsample(x, j, is_variance)
        out += residual
        out = self.relu(out)
        return out

    def reg_loss(self):
        total = self.conv1.reg_loss() + self.conv2.reg_loss()
        if self.downsample:
            total += self.downsample.reg_loss()
        return total


class ResNet(nn.Module):

    def __init__(self, m, layers, num_classes=10):
        super().__init__()
        self.all_residual_blocks = []
        self.inplanes = 64
        k = 7
        c_in = 3
        self.conv1 = ConvBlock(m, c_in, self.inplanes, k, 2, 3, True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer0 = self._make_layer(m, 64, layers[0], stride=1)
        self.layer1 = self._make_layer(m, 128, layers[1], stride=2)
        self.layer2 = self._make_layer(m, 256, layers[2], stride=2)
        self.layer3 = self._make_layer(m, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = Linear(m, 512, num_classes)


    def _make_layer(self, m, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = ConvBlock(m, self.inplanes, planes, 1, stride, with_relu=False)
        layers = [ResidualBlock(m, self.inplanes, planes, stride, downsample)]
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(ResidualBlock(m, self.inplanes, planes))
        self.all_residual_blocks.extend(layers)
        return Sequential(*layers)

    def forward(self, x, j, is_variance):
        x = self.conv1(x, j, is_variance)
        x = self.maxpool(x)
        x = self.layer0(x, j, is_variance)
        x = self.layer1(x, j, is_variance)
        x = self.layer2(x, j, is_variance)
        x = self.layer3(x, j, is_variance)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x, j, is_variance)

        return x

    def reg_loss(self):
        return self.conv1.reg_loss() + sum([r.reg_loss() for r in self.all_residual_blocks]) + self.fc.reg_loss()



class MultiMlp(nn.Module):

    def __init__(self, m, num_input, num_hidden, num_classes):
        super().__init__()
        self.hidden_layer = Linear(m, num_input, num_hidden)
        self.relu = nn.ReLU()
        self.output_layer = Linear(m, num_hidden, num_classes)

    def forward(self, x, j, is_variance):
        a = self.hidden_layer(x, j, is_variance)
        a = self.relu(a)
        return self.output_layer(a, j, is_variance)

    def reg_loss(self):
        return self.hidden_layer.reg_loss() + self.output_layer.reg_loss()


class SharedBase(MultiModule):

    def __init__(self):
        super().__init__()

    def get_params(self, j, is_variance=None): # is_variance never used for this subclass, purely for SharedMagnitude
        w = self.base_weight
        b = self.base_bias
        if j is not None:
            w = w.detach() + self.weight[j]
            b = b.detach() + self.bias[j]
        return w, b

    def reg_loss(self):
        w_reg = torch.sum(self.base_weight ** 2)
        b_reg = torch.sum(self.base_bias ** 2)
        if IS_PROP_VAR:
            w_reg = get_reg(self.weight, self.base_weight)
            b_reg = get_reg(self.bias, self.base_bias)
        return torch.sum(w_reg) + torch.sum(b_reg)


# def create_params(m, base_weight, base_bias):
#     """Take a base_weight and a base_bias tensor, create multi-model versions of the same shape, and then return the
#     original tensors plus the multi-model tensors as parameters"""
#     weight_shape = [m] + list(base_weight.shape)
#     bias_shape = [m] + list(base_bias.shape)
#     weight = torch.zeros(*weight_shape)
#     bias = torch.zeros(*bias_shape)
#     return nn.Parameter(base_weight), nn.Parameter(base_bias), nn.Parameter(weight), nn.Parameter(bias)
#
# def create_batch_norm_params(m, num_features):
#     base_weight = torch.ones(num_features)
#     base_bias = torch.zeros(num_features)
#     return create_params(m, base_weight, base_bias)
#
# def create_conv_params(m, c_in, out, k):
#     scaler = 1 / (c_in * k ** 2) ** 0.5
#     base_weight = 2 * (torch.rand(out, c_in, k, k, requires_grad=True) - 0.5) * scaler
#     base_bias = 2 * (torch.rand(out, requires_grad=True) - 0.5) * scaler
#     return create_params(m, base_weight, base_bias)
#
# def create_linear_params(m, in_dim, out):
#     scaler = 1 / in_dim ** 0.5
#     base_weight = 2 * (torch.rand(out, in_dim, requires_grad=True) - 0.5)  * scaler
#     base_bias = 2 * (torch.rand(out, requires_grad=True) - 0.5) * scaler
#     return create_params(m, base_weight, base_bias)


class RetainedSequential(nn.Sequential):

    def __init__(self, *args):
        super().__init__(*args)
        self.recent_input = None
        self.recent_activations = None

    def forward(self, x):
        self.recent_input = x
        self.recent_activations = []
        for module in self:
            x = module(x)
            # The first condition makes sure we only track the linear outputs, the second stops that from happening
            # when the network is in evaluation mode.
            if isinstance(module, nn.Linear) and x.requires_grad:
                x.retain_grad()  # So that we may recompute the gradient to the weight at that layer.
                self.recent_activations.append(x)
        return x


class Mlp(nn.Module):

    def __init__(self, sizes, num_class, is_bias: bool):
        super().__init__()
        self.num_input = sizes[0]
        ops = []
        num_output = sizes[0] # For the case where there are no hidden layers
        for num_input, num_output in zip(sizes[:-1], sizes[1:]):
            self._append_to_layer(num_input, num_output, ops, is_bias)
            ops.append(nn.ReLU())
        self._append_to_layer(num_output, num_class, ops, is_bias)
        self.layers = RetainedSequential(*ops)

    def _append_to_layer(self, num_input, num_output, ops, is_bias: bool):
        linear = nn.Linear(num_input, num_output, bias=is_bias)
        ops.append(linear)

    def forward(self, x):
        return self.layers(x)



