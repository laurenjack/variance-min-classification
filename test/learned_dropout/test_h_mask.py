#!/usr/bin/env python3
"""
Pytest tests to verify h_mask functionality works as expected.
"""

import torch
import pytest
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))

from src.learned_dropout.models_standard import ResNetStandard
from src.learned_dropout.config import ModelConfig


def test_h_mask_forward_pass():
    """Test that h_mask reduces effective hidden dimension correctly in forward pass."""
    
    # Create a simple configuration
    config = ModelConfig(
        d=4,
        n_val=100,
        n=1000,
        batch_size=32,
        layer_norm="layer_norm",
        lr=0.001,
        epochs=10,
        weight_decay=0.0,
        hidden_sizes=[8, 8],  # Two blocks with hidden size 8
        d_model=None
    )
    
    # Create model with h_list = [8, 8]
    model_full = ResNetStandard(config, h_list=[8, 8])
    
    # Create model with h_list = [4, 4] (reduced)
    model_reduced = ResNetStandard(config, h_list=[4, 4])
    
    # Copy weights from full model to reduced model (first 4 dimensions)
    with torch.no_grad():
        # Copy input weights (first 4 columns)
        for i, (full_block, reduced_block) in enumerate(zip(model_full.blocks, model_reduced.blocks)):
            reduced_block.weight_in.weight.data = full_block.weight_in.weight.data[:4, :]
            reduced_block.weight_out.weight.data = full_block.weight_out.weight.data[:, :4]
            
        # Copy final layer weights  
        model_reduced.final_layer.weight.data = model_full.final_layer.weight.data
    
    # Create h_mask with first 4 elements as 1.0, rest as 0.0
    h_mask = torch.zeros(8)
    h_mask[:4] = 1.0
    
    # Test with same input
    x = torch.randn(5, 4)  # batch_size=5, input_dim=4
    
    # Forward pass with full model and mask
    output_masked = model_full(x, h_mask=h_mask)
    
    # Forward pass with reduced model
    output_reduced = model_reduced(x)
    
    # They should be very close (small numerical differences expected)
    assert torch.allclose(output_masked, output_reduced, atol=1e-5), "Outputs should be nearly identical"


def test_h_mask_gradients():
    """Test that gradients are identical between masked and reduced models."""
    
    # Create configuration
    config = ModelConfig(
        d=3,
        n_val=100,
        n=1000,
        batch_size=32,
        layer_norm="rms_norm",
        lr=0.001,
        epochs=10,
        weight_decay=0.0,
        hidden_sizes=[6],  # One block with hidden size 6
        d_model=None
    )
    
    # Create models
    model_full = ResNetStandard(config, h_list=[6])
    model_reduced = ResNetStandard(config, h_list=[3])
    
    # Copy weights from full model to reduced model (first 3 dimensions)
    with torch.no_grad():
        model_reduced.blocks[0].weight_in.weight.data = model_full.blocks[0].weight_in.weight.data[:3, :]
        model_reduced.blocks[0].weight_out.weight.data = model_full.blocks[0].weight_out.weight.data[:, :3]
        model_reduced.final_layer.weight.data = model_full.final_layer.weight.data
    
    # Create h_mask with first 3 elements as 1.0, rest as 0.0
    h_mask = torch.zeros(6)
    h_mask[:3] = 1.0
    
    # Test input and target (need separate inputs for separate backward passes)
    x_masked = torch.randn(2, 3, requires_grad=True)
    x_reduced = x_masked.clone().detach().requires_grad_(True)
    target = torch.randn(2)
    
    # Forward pass with masked model
    output_masked = model_full(x_masked, h_mask=h_mask)
    loss_masked = torch.nn.functional.mse_loss(output_masked, target)
    
    # Forward pass with reduced model
    output_reduced = model_reduced(x_reduced)
    loss_reduced = torch.nn.functional.mse_loss(output_reduced, target)
    
    # Compute gradients
    loss_masked.backward()
    loss_reduced.backward()
    
    # Check that outputs are nearly identical
    assert torch.allclose(output_masked, output_reduced, atol=1e-5), "Outputs should be nearly identical"
    
    # Check that losses are nearly identical
    assert torch.allclose(loss_masked, loss_reduced, atol=1e-5), "Losses should be nearly identical"
    
    # Check gradients for input
    assert torch.allclose(x_masked.grad, x_reduced.grad, atol=1e-5), "Input gradients should be identical"
    
    # Check gradients for corresponding weights in both models
    # Input layer weights (first 3 rows should match)
    full_input_grad = model_full.blocks[0].weight_in.weight.grad[:3, :]
    reduced_input_grad = model_reduced.blocks[0].weight_in.weight.grad
    assert torch.allclose(full_input_grad, reduced_input_grad, atol=1e-5), "Input weight gradients should match"
    
    # Output layer weights (first 3 columns should match)
    full_output_grad = model_full.blocks[0].weight_out.weight.grad[:, :3]
    reduced_output_grad = model_reduced.blocks[0].weight_out.weight.grad
    assert torch.allclose(full_output_grad, reduced_output_grad, atol=1e-5), "Output weight gradients should match"
    
    # Final layer weights should match
    assert torch.allclose(model_full.final_layer.weight.grad, model_reduced.final_layer.weight.grad, atol=1e-5), "Final layer gradients should match"
