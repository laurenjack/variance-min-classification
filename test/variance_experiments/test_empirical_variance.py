#!/usr/bin/env python3
"""
Pytest tests to verify width-mask functionality (h, d_model, down_rank_dim).
The masked models should behave like smaller base Resnet models with the
corresponding parameter equal to the mask length, when weights are aligned.
"""

import torch

from jl.model_creator import create_resnet, create_mlp
from jl.config import Config


def test_h_mask_forward_pass():
    """h-mask reduces effective hidden dimension in forward pass to match base Resnet(h*)."""
    d = 4
    h_max = 8
    h_star = 4
    c_full = Config(model_type='resnet', d=d, n_val=100, n=1000, batch_size=32, lr=0.001, epochs=1, weight_decay=0.0,
                    num_layers=2, num_class=2, h=h_max, d_model=None, width_varyer="h")
    c_reduced = Config(model_type='resnet', d=d, n_val=100, n=1000, batch_size=32, lr=0.001, epochs=1, weight_decay=0.0,
                       num_layers=2, num_class=2, h=h_star, d_model=None)

    model_masked = create_resnet(c_full)
    model_reduced = create_resnet(c_reduced)

    # Align weights for overlapping hidden dims
    with torch.no_grad():
        for full_blk, red_blk in zip(model_masked.blocks, model_reduced.blocks):
            red_blk.weight_in.weight.data = full_blk.weight_in.weight.data[:h_star, :]
            red_blk.weight_out.weight.data = full_blk.weight_out.weight.data[:, :h_star]
        model_reduced.final_layer.weight.data = model_masked.final_layer.weight.data

    h_mask = torch.zeros(h_max)
    h_mask[:h_star] = 1.0

    x = torch.randn(5, d)
    out_masked = model_masked(x, width_mask=h_mask)
    out_reduced = model_reduced(x)
    assert torch.allclose(out_masked, out_reduced, atol=1e-5)


def test_h_mask_gradients():
    """Gradients match between masked h model and base smaller-h model when weights align."""
    d = 3
    h_max = 6
    h_star = 3
    c_full = Config(model_type='resnet', d=d, n_val=100, n=1000, batch_size=32, lr=0.001, epochs=1, weight_decay=0.0,
                    num_layers=1, num_class=2, h=h_max, d_model=None, width_varyer="h")
    c_reduced = Config(model_type='resnet', d=d, n_val=100, n=1000, batch_size=32, lr=0.001, epochs=1, weight_decay=0.0,
                       num_layers=1, num_class=2, h=h_star, d_model=None)

    model_masked = create_resnet(c_full)
    model_reduced = create_resnet(c_reduced)

    with torch.no_grad():
        model_reduced.blocks[0].weight_in.weight.data = model_masked.blocks[0].weight_in.weight.data[:h_star, :]
        model_reduced.blocks[0].weight_out.weight.data = model_masked.blocks[0].weight_out.weight.data[:, :h_star]
        model_reduced.final_layer.weight.data = model_masked.final_layer.weight.data

    h_mask = torch.zeros(h_max)
    h_mask[:h_star] = 1.0

    x_masked = torch.randn(2, d, requires_grad=True)
    x_reduced = x_masked.clone().detach().requires_grad_(True)
    target = torch.randn(2)

    out_masked = model_masked(x_masked, width_mask=h_mask)
    out_reduced = model_reduced(x_reduced)
    loss_masked = torch.nn.functional.mse_loss(out_masked, target)
    loss_reduced = torch.nn.functional.mse_loss(out_reduced, target)

    loss_masked.backward()
    loss_reduced.backward()

    assert torch.allclose(out_masked, out_reduced, atol=1e-5)
    assert torch.allclose(loss_masked, loss_reduced, atol=1e-5)
    assert torch.allclose(x_masked.grad, x_reduced.grad, atol=1e-5)

    full_in_grad = model_masked.blocks[0].weight_in.weight.grad[:h_star, :]
    red_in_grad = model_reduced.blocks[0].weight_in.weight.grad
    assert torch.allclose(full_in_grad, red_in_grad, atol=1e-5)

    full_out_grad = model_masked.blocks[0].weight_out.weight.grad[:, :h_star]
    red_out_grad = model_reduced.blocks[0].weight_out.weight.grad
    assert torch.allclose(full_out_grad, red_out_grad, atol=1e-5)

    assert torch.allclose(model_masked.final_layer.weight.grad, model_reduced.final_layer.weight.grad, atol=1e-5)


def test_d_model_mask_forward_pass():
    """d_model mask simulates a base Resnet with d_model* (no down-rank)."""
    d = 5
    d_model_max = 12
    d_model_star = 7
    h = 6
    c_masked = Config(model_type='resnet', d=d, n_val=10, n=10, batch_size=2, lr=1e-3, epochs=1, weight_decay=0.0,
                      num_layers=2, num_class=2, h=h, d_model=d_model_max, width_varyer="d_model")
    c_reduced = Config(model_type='resnet', d=d, n_val=10, n=10, batch_size=2, lr=1e-3, epochs=1, weight_decay=0.0,
                       num_layers=2, num_class=2, h=h, d_model=d_model_star)

    model_masked = create_resnet(c_masked)
    model_reduced = create_resnet(c_reduced)

    with torch.no_grad():
        # Align input projection
        model_reduced.input_projection.weight.data = model_masked.input_projection.weight.data[:d_model_star, :]
        # Align block weights (match d_model* columns)
        for full_blk, red_blk in zip(model_masked.blocks, model_reduced.blocks):
            red_blk.weight_in.weight.data = full_blk.weight_in.weight.data[:, :d_model_star]
            red_blk.weight_out.weight.data = full_blk.weight_out.weight.data[:d_model_star, :]
        # Align final layer (first d_model* columns)
        model_reduced.final_layer.weight.data = model_masked.final_layer.weight.data[:, :d_model_star]

    d_mask = torch.zeros(d_model_max)
    d_mask[:d_model_star] = 1.0

    x = torch.randn(3, d)
    out_masked = model_masked(x, width_mask=d_mask)
    out_reduced = model_reduced(x)
    assert torch.allclose(out_masked, out_reduced, atol=1e-5)


def test_down_rank_dim_mask_forward_pass():
    """down_rank_dim mask simulates base Resnet with down_rank_dim*."""
    d = 4
    d_model = None
    h = 6
    dr_max = 8
    dr_star = 5
    c_masked = Config(model_type='resnet', d=d, n_val=10, n=10, batch_size=2, lr=1e-3, epochs=1, weight_decay=0.0,
                      num_layers=2, num_class=2, h=h, d_model=d_model, down_rank_dim=dr_max, width_varyer="down_rank_dim")
    c_reduced = Config(model_type='resnet', d=d, n_val=10, n=10, batch_size=2, lr=1e-3, epochs=1, weight_decay=0.0,
                       num_layers=2, num_class=2, h=h, d_model=d_model, down_rank_dim=dr_star)

    model_masked = create_resnet(c_masked)
    model_reduced = create_resnet(c_reduced)

    with torch.no_grad():
        # Align all pre-down-rank weights (blocks are same shape)
        for full_blk, red_blk in zip(model_masked.blocks, model_reduced.blocks):
            red_blk.weight_in.weight.data = full_blk.weight_in.weight.data
            red_blk.weight_out.weight.data = full_blk.weight_out.weight.data
        # Align down-rank layer and final
        model_reduced.down_rank_layer.weight.data = model_masked.down_rank_layer.weight.data[:dr_star, :]
        model_reduced.final_layer.weight.data = model_masked.final_layer.weight.data[:, :dr_star]

    dr_mask = torch.zeros(dr_max)
    dr_mask[:dr_star] = 1.0

    x = torch.randn(3, d)
    out_masked = model_masked(x, width_mask=dr_mask)
    out_reduced = model_reduced(x)
    assert torch.allclose(out_masked, out_reduced, atol=1e-5)


def test_h_mask_forward_pass_mlp():
    """h mask simulates a base MLP with h* (no down-rank)."""
    d = 5
    h_max = 12
    h_star = 7
    c_masked = Config(model_type='mlp', d=d, n_val=10, n=10, batch_size=2, lr=1e-3, epochs=1, weight_decay=0.0,
                      num_layers=2, num_class=2, h=h_max, width_varyer="h")
    c_reduced = Config(model_type='mlp', d=d, n_val=10, n=10, batch_size=2, lr=1e-3, epochs=1, weight_decay=0.0,
                       num_layers=2, num_class=2, h=h_star)

    model_masked = create_mlp(c_masked)
    model_reduced = create_mlp(c_reduced)

    with torch.no_grad():
        # Align input layer
        model_reduced.layers[0].weight.data = model_masked.input_layer.weight.data[:h_star, :]
        
        # Align first norm (after input layer)
        model_reduced.layers[1].weight.data = model_masked.input_norm.weight.data[:h_star]
        
        # Align hidden layers
        # model_reduced has Sequential with pattern: Linear, RMSNorm, ReLU for each layer
        # For num_layers=2: [Linear(input), RMSNorm, ReLU, Linear(hidden), RMSNorm, ReLU]
        # So hidden layer starts at index 3
        if len(model_masked.hidden_layers) > 0:
            model_reduced.layers[3].weight.data = model_masked.hidden_layers[0].weight.data[:h_star, :h_star]
            model_reduced.layers[4].weight.data = model_masked.hidden_norms[0].weight.data[:h_star]
        
        # Align final layer (first h* columns)
        model_reduced.final_layer.weight.data = model_masked.final_layer.weight.data[:, :h_star]

    h_mask = torch.zeros(h_max)
    h_mask[:h_star] = 1.0

    x = torch.randn(3, d)
    out_masked = model_masked(x, width_mask=h_mask)
    out_reduced = model_reduced(x)
    assert torch.allclose(out_masked, out_reduced, atol=1e-5)


def test_down_rank_dim_mask_forward_pass_mlp():
    """down_rank_dim mask simulates base MLP with down_rank_dim*."""
    d = 4
    h = 8
    dr_max = 10
    dr_star = 6
    c_masked = Config(model_type='mlp', d=d, n_val=10, n=10, batch_size=2, lr=1e-3, epochs=1, weight_decay=0.0,
                      num_layers=2, num_class=2, h=h, down_rank_dim=dr_max, width_varyer="down_rank_dim")
    c_reduced = Config(model_type='mlp', d=d, n_val=10, n=10, batch_size=2, lr=1e-3, epochs=1, weight_decay=0.0,
                       num_layers=2, num_class=2, h=h, down_rank_dim=dr_star)

    model_masked = create_mlp(c_masked)
    model_reduced = create_mlp(c_reduced)

    with torch.no_grad():
        # Align all pre-down-rank layers (they have same shape)
        # Both models have Sequential: [Linear, RMSNorm, ReLU, Linear, RMSNorm, ReLU, ...]
        # Copy layer by layer for the sequential part
        linear_idx = 0
        for i, layer in enumerate(model_reduced.layers):
            if isinstance(layer, torch.nn.Linear):
                model_reduced.layers[i].weight.data = model_masked.layers[i].weight.data
            elif hasattr(layer, 'weight'):  # RMSNorm
                model_reduced.layers[i].weight.data = model_masked.layers[i].weight.data
        
        # Align down-rank layer and final
        model_reduced.down_rank_layer.weight.data = model_masked.down_rank_layer.weight.data[:dr_star, :]
        model_reduced.final_layer.weight.data = model_masked.final_layer.weight.data[:, :dr_star]

    dr_mask = torch.zeros(dr_max)
    dr_mask[:dr_star] = 1.0

    x = torch.randn(3, d)
    out_masked = model_masked(x, width_mask=dr_mask)
    out_reduced = model_reduced(x)
    assert torch.allclose(out_masked, out_reduced, atol=1e-5)
