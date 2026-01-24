import torch
import torch.nn as nn
from torch.autograd import Function


class ScaledRegLinearFunction(Function):
    """
    Custom autograd function for Scaled Regularization training.

    Forward: Standard linear transformation z = x @ W^T
    Backward:
        Normal mode: standard gradients (dl_dz @ W for input, dl_dz.T @ x for weight)
        Reg mode: standard input gradient, but weight gradient is the scaled reg term:
            mean(dl_dz^2, dim=0, keepdim=True).T * W * scaled_reg_k
    """
    @staticmethod
    def forward(ctx, x, weight, reg_mode, scaled_reg_k):
        ctx.save_for_backward(x, weight)
        ctx.reg_mode = reg_mode
        ctx.scaled_reg_k = scaled_reg_k
        return x.matmul(weight.t())

    @staticmethod
    def backward(ctx, dl_dz):
        x, weight = ctx.saved_tensors

        # Input gradient: always standard (so gradient flows backward correctly)
        grad_x = dl_dz @ weight

        if ctx.reg_mode:
            # Reg mode: weight gradient is the scaled regularization term
            # mean(dl_dz^2, dim=0) gives [d_out], keepdim gives [1, d_out]
            mse = torch.mean(dl_dz ** 2, dim=0, keepdim=True)  # [1, d_out]
            # mse.T is [d_out, 1], broadcast multiply with weight [d_out, d_in]
            grad_w = mse.t() * weight * ctx.scaled_reg_k  # [d_out, d_in]
        else:
            # Normal mode: standard weight gradient
            grad_w = dl_dz.t() @ x

        return grad_x, grad_w, None, None


class ScaledRegLinear(nn.Module):
    """
    Linear layer with custom Scaled Regularization gradient computation.

    In normal mode, behaves exactly like nn.Linear(bias=False).
    In reg mode, the weight gradient becomes the scaled regularization term.

    Args:
        in_features: Size of each input sample
        out_features: Size of each output sample
        scaled_reg_k: Regularization coefficient
    """
    def __init__(self, in_features: int, out_features: int, scaled_reg_k: float):
        super(ScaledRegLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scaled_reg_k = scaled_reg_k
        self.reg_mode = False
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)

    def forward(self, x):
        return ScaledRegLinearFunction.apply(x, self.weight, self.reg_mode, self.scaled_reg_k)

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, scaled_reg_k={self.scaled_reg_k}'


def set_reg_mode(model, mode: bool):
    """Set reg_mode on all ScaledRegLinear layers in the model."""
    for module in model.modules():
        if isinstance(module, ScaledRegLinear):
            module.reg_mode = mode
