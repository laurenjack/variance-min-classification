import torch
import torch.nn as nn
from torch.autograd import Function


class RMSNormLPFunction(Function):
    """
    Custom autograd function for RMSNorm in Logit Prior training.
    
    Forward: Standard RMSNorm: y = x / rms where rms = sqrt(mean(x^2) + eps)
    Backward: Standard RMSNorm gradients (for now - to be modified for logit prior)
    
    Note: This implementation assumes non-learnable weights (weight=1).
    """
    @staticmethod
    def forward(ctx, x, eps):
        # Compute RMS
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + eps)
        # Normalize
        y = x / rms
        # Save for backward
        ctx.save_for_backward(x, rms)
        ctx.eps = eps
        return y

    @staticmethod
    def backward(ctx, grad_output):
        x, rms = ctx.saved_tensors
        d = x.shape[-1]
        
        # RMSNorm gradient (multiplied by unstable denominator):
        grad_x = grad_output - x * (grad_output * x).sum(dim=-1, keepdim=True) / (d * rms ** 2)
        
        # No gradient for eps (it's a constant)
        return grad_x, None


class RMSNormLP(nn.Module):
    """
    RMSNorm with custom Logit Prior gradient computation.
    
    This is similar to the standard RMSNorm but uses a custom autograd function
    for the backward pass. Only supports non-learnable weights.
    
    Args:
        dim: Feature dimension for normalization
        eps: Small constant for numerical stability (default 1e-6)
    """
    def __init__(self, dim, eps=1e-6):
        super(RMSNormLP, self).__init__()
        self.dim = dim
        self.eps = eps
        # Non-learnable weight (buffer, not parameter)
        self.register_buffer('weight', torch.ones(dim))
    
    def forward(self, x):
        # Apply custom autograd function (weight is always 1, so just return normalized)
        return RMSNormLPFunction.apply(x, self.eps)
    
    def extra_repr(self) -> str:
        return f'dim={self.dim}, eps={self.eps}'


class LinearLPFunction(Function):
    """
    Custom autograd function for Logit Prior training.
    
    Forward: Standard linear transformation z = x @ W^T
    Backward: 
        - Gradient w.r.t. input: standard dF_dz @ W (passed to previous layer as normal)
        - Gradient w.r.t. weight: computed using for_weight = dL_dz / (dZ_dz + 1e-8)
          where dL_dz and dZ_dz are the first and second halves of the batch gradient
    """
    @staticmethod
    def forward(ctx, x, weight):
        ctx.save_for_backward(x, weight)
        output = x.matmul(weight.t())
        return output

    @staticmethod
    def backward(ctx, dF_dz):
        x, weight = ctx.saved_tensors
        m = dF_dz.shape[0]
        # Split gradients along batch dimension
        dL_dz = dF_dz[:m // 2]  # Gradient from loss L (cross entropy / BCE)
        dZ_dz = dF_dz[m // 2:]  # Gradient from loss Z (sum of squared logits)
        # Compute modified weight gradient
        for_weight = dL_dz / (torch.abs(dZ_dz) + 1e-8)
        # Use corresponding half of x for weight gradient
        grad_w = for_weight.t() @ x[:m // 2]
        # Return standard gradient for x (backpropagated to previous layer as normal)
        return dF_dz @ weight, grad_w


class LinearLP(nn.Module):
    """
    Linear layer with custom Logit Prior gradient computation.
    
    This class extends PyTorch's Linear functionality but uses a custom autograd
    function (LinearLPFunction) for the backward pass. It does not support biases.
    
    Args:
        in_features: Size of each input sample
        out_features: Size of each output sample
        bias: Must be False (biases not supported)
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super(LinearLP, self).__init__()
        if bias:
            raise ValueError("LinearLP does not support biases. Set bias=False.")
        self.in_features = in_features
        self.out_features = out_features
        # Initialize weight matrix similar to nn.Linear
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
    
    def forward(self, x):
        return LinearLPFunction.apply(x, self.weight)
    
    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias=False'
