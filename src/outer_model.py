import torch
from torch import nn
from torch.autograd import Function


class SingleXeGradient(Function):

    @staticmethod
    def forward(ctx, x, y):
        ctx.save_for_backward(y)
        return x.clone()

    @staticmethod
    def backward(ctx, logit_gradient):
        y, = ctx.saved_tensors
        batch_size = y.shape[0]
        batch_indices = torch.arange(batch_size, dtype=torch.int64)
        single_logit_gradient = torch.zeros(*logit_gradient.shape)
        single_logit_gradient[batch_indices, y] = logit_gradient[batch_indices, y]
        return single_logit_gradient, None


class Xe(nn.Module):

    def __init__(self, model: nn.Module):
        self.model = model

    def forward(self, x):
        return self.model(x)


class SingleXe(Xe):

    def forward(self, x, y):
        a = self.model(x)
        return SingleXeGradient.apply(a, y)