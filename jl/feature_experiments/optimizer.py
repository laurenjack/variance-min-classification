"""
RegAdamW optimizer implementation.

This module provides:
1. RegAdamW - A variant of AdamW with modified second moment calculation and update rule
2. Helper functions to register hooks on Linear modules for computing per-sample gradient statistics
"""

import torch
import torch.nn as nn
from torch.optim import Optimizer
from typing import List, Tuple


def _forward_hook(module: nn.Linear, input: Tuple[torch.Tensor, ...], output: torch.Tensor):
    """
    Forward hook to store the input tensor on the module for use in backward hook.
    """
    # input is a tuple, first element is the actual input tensor
    module._reg_input = input[0].detach()


def _backward_hook(module: nn.Linear, grad_input: Tuple[torch.Tensor, ...], grad_output: Tuple[torch.Tensor, ...]):
    """
    Backward hook to compute g2_per_n for RegAdamW.
    
    g_out [batch_size, d1] is the gradient w.r.t. the output of the Linear module.
    x [batch_size, d0] is the input stored during forward pass.
    
    For weight of shape [d1, d0]:
    - g2_per_n = batch_size * g_out.t() ** 2 @ x ** 2
    """
    g_out = grad_output[0]  # [batch_size, d1]
    x = module._reg_input    # [batch_size, d0]
    batch_size = g_out.shape[0]
    
    # Compute g2_per_n: batch_size * sum of per-sample gradient squared
    # g_out.t() is [d1, batch_size], x is [batch_size, d0]
    # Result is [d1, d0] matching weight shape
    g2_per_n = batch_size * (g_out.t() ** 2) @ (x ** 2)
    
    # Store on the weight parameter
    module.weight._g2_per_n = g2_per_n
    
    # Clean up stored input
    del module._reg_input


def register_reg_adam_w_hooks(model: nn.Module) -> List[torch.utils.hooks.RemovableHandle]:
    """
    Register forward and backward hooks on all Linear modules in the model.
    
    Args:
        model: The neural network model
        
    Returns:
        List of hook handles (can be used to remove hooks later if needed)
    """
    handles = []
    for module in model.modules():
        if isinstance(module, nn.Linear):
            handles.append(module.register_forward_hook(_forward_hook))
            handles.append(module.register_full_backward_hook(_backward_hook))
    return handles


class RegAdamW(Optimizer):
    """
    RegAdamW optimizer - AdamW with modified second moment calculation and update rule.
    
    The second moment uses the sum of per-sample gradient squared (g2_per_n) instead
    of the squared gradient. The update rule scales both the first moment and weight
    decay term by the denominator: W - lr*(m + wd*W) / (sqrt(v) + eps)
    
    Args:
        params: Iterable of parameters to optimize
        lr: Learning rate (default: 1e-3)
        betas: Coefficients for computing running averages (default: (0.9, 0.999))
        eps: Term added to denominator for numerical stability (default: 1e-8)
        weight_decay: Weight decay coefficient (default: 0.01)
    """
    
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
            
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(RegAdamW, self).__init__(params, defaults)
    
    @torch.no_grad()
    def step(self, closure=None):
        """
        Performs a single optimization step.
        
        Args:
            closure: A closure that reevaluates the model and returns the loss (optional)
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            lr = group['lr']
            eps = group['eps']
            weight_decay = group['weight_decay']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                
                # Get state for this parameter
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                state['step'] += 1
                
                # Get g2_per_n computed by backward hook
                g2_per_n = p._g2_per_n
                
                # First moment: use standard gradient (no scaling)
                first_moment = grad
                
                # Second moment: g2_per_n
                second_moment = g2_per_n
                
                # Clean up stored statistics
                del p._g2_per_n
                
                # Update biased first and second moment estimates
                exp_avg.mul_(beta1).add_(first_moment, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).add_(second_moment, alpha=1 - beta2)
                
                # Bias correction
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                # Compute bias-corrected estimates
                exp_avg_corrected = exp_avg / bias_correction1
                exp_avg_sq_corrected = exp_avg_sq / bias_correction2
                
                # Compute the denominator
                denom = exp_avg_sq_corrected.sqrt().add_(eps)
                
                # RegAdamW update rule: W - lr*(m + wd*W) / (sqrt(v) + eps)
                p.sub_((exp_avg_corrected + weight_decay * p) / denom * lr)
        
        return loss
