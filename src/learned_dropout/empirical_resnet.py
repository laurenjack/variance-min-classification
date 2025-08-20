import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from src import dataset_creator
from src.learned_dropout.models_standard import ResNetStandard
import src.learned_dropout.empirical_variance as ev
from src.learned_dropout.config import EmpiricalConfig


def build_single_resnet(c: EmpiricalConfig, h: int, device: torch.device) -> nn.Module:
    """Build a ResNet with a single residual block: [d] -> ResidualBlock(d, h) -> [d] -> Linear(d, 1)"""
    # Single residual block with hidden dimension h
    h_list = [h]
    return ResNetStandard(c=c, h_list=h_list).to(device)


def build_double_resnet(c: EmpiricalConfig, h: int, device: torch.device) -> nn.Module:
    """Build a ResNet with two residual blocks: [d] -> ResidualBlock(d, h) -> ResidualBlock(d, h) -> [d] -> Linear(d, 1)"""
    # Two residual blocks with hidden dimension h
    h_list = [h, h]
    return ResNetStandard(c=c, h_list=h_list).to(device)


def build_triple_resnet(c: EmpiricalConfig, h: int, device: torch.device) -> nn.Module:
    """Build a ResNet with three residual blocks: [d] -> ResidualBlock(d, h) -> ResidualBlock(d, h) -> ResidualBlock(d, h) -> [d] -> Linear(d, 1)"""
    # Three residual blocks with hidden dimension h
    h_list = [h, h, h]
    return ResNetStandard(c=c, h_list=h_list).to(device)


def build_quadruple_resnet(c: EmpiricalConfig, h: int, device: torch.device) -> nn.Module:
    """Build a ResNet with four residual blocks: [d] -> ResidualBlock(d, h) -> ResidualBlock(d, h) -> ResidualBlock(d, h) -> ResidualBlock(d, h) -> [d] -> Linear(d, 1)"""
    # Four residual blocks with hidden dimension h
    h_list = [h, h, h, h]
    return ResNetStandard(c=c, h_list=h_list).to(device)


def run_resnet_experiment(device, problem, validation_set, c: EmpiricalConfig):
    """Train and display variance quantities for single-layer ResNet."""
    print("Running Single-Layer ResNet Experiment")
    model_vars, model_losses, model_val_accuracies, model_val_losses = ev.run_experiment(
        build_single_resnet, device, validation_set, problem, c
    )

    # plot results with dual y-axes
    fig, ax1 = plt.subplots(figsize=(12, 8))
    
    # Plot variance on left y-axis
    color = 'tab:blue'
    ax1.set_xlabel("Hidden size parameter h")
    ax1.set_ylabel("Mean variance of validation logits", color=color)
    ax1.plot(c.h_range, model_vars, marker="o", label="Variance", color='blue', linestyle='-')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)
    
    # Create second y-axis for training loss
    ax2 = ax1.twinx()
    color = 'tab:orange'
    ax2.set_ylabel("Mean training loss", color=color)
    ax2.plot(c.h_range, model_losses, marker="s", label="Training Loss", color='orange', linestyle='--')
    ax2.tick_params(axis='y', labelcolor=color)
    
    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    plt.title(f"Single-Layer ResNet: Variance and training loss vs. hidden size (d = {c.d}, n = {c.n})")
    plt.tight_layout()
    plt.show()


def three_resnet_experiment(device, problem, validation_set, c: EmpiricalConfig):
    """Train and compare three ResNet models: double vs triple vs quadruple residual blocks."""
    print("Running ResNet 1 (double residual blocks)")
    r1_vars, r1_losses, r1_val_accuracies, r1_val_losses = ev.run_experiment(
        build_double_resnet, device, validation_set, problem, c
    )

    print("\nRunning ResNet 2 (triple residual blocks)")
    r2_vars, r2_losses, r2_val_accuracies, r2_val_losses = ev.run_experiment(
        build_triple_resnet, device, validation_set, problem, c
    )

    print("\nRunning ResNet 3 (quadruple residual blocks)")
    r3_vars, r3_losses, r3_val_accuracies, r3_val_losses = ev.run_experiment(
        build_quadruple_resnet, device, validation_set, problem, c
    )

    # plot results with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))
    
    # Colors for the three models
    color_double = 'tab:blue'
    color_triple = 'tab:green'
    color_quadruple = 'tab:red'
    
    # Subplot 1: Variance and Validation Loss (dual y-axis)
    ax1.set_ylabel("Mean variance of validation logits", color='black')
    ax1.plot(c.h_range, r1_vars, marker="o", label="Double ResNet", color=color_double, linestyle='-')
    ax1.plot(c.h_range, r2_vars, marker="s", label="Triple ResNet", color=color_triple, linestyle='-')
    ax1.plot(c.h_range, r3_vars, marker="^", label="Quadruple ResNet", color=color_quadruple, linestyle='-')
    ax1.grid(True, alpha=0.3)
    
    # Create second y-axis for validation loss
    ax1_twin = ax1.twinx()
    color_val = 'tab:orange'
    ax1_twin.set_ylabel("Mean validation loss", color=color_val)
    ax1_twin.plot(c.h_range, r1_val_losses, marker="v", label="Double ResNet (val)", color=color_double, linestyle='--', alpha=0.7)
    ax1_twin.plot(c.h_range, r2_val_losses, marker="<", label="Triple ResNet (val)", color=color_triple, linestyle='--', alpha=0.7)
    ax1_twin.plot(c.h_range, r3_val_losses, marker=">", label="Quadruple ResNet (val)", color=color_quadruple, linestyle='--', alpha=0.7)
    ax1_twin.tick_params(axis='y', labelcolor=color_val)
    
    # Combine legends for first subplot
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    # Subplot 2: Training Loss
    ax2.plot(c.h_range, r1_losses, marker="o", label="Double ResNet", color=color_double, linestyle='-')
    ax2.plot(c.h_range, r2_losses, marker="s", label="Triple ResNet", color=color_triple, linestyle='-')
    ax2.plot(c.h_range, r3_losses, marker="^", label="Quadruple ResNet", color=color_quadruple, linestyle='-')
    ax2.set_ylabel("Mean training loss", color='black')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Subplot 3: Validation Accuracy
    ax3.plot(c.h_range, r1_val_accuracies, marker="o", label="Double ResNet", color=color_double, linestyle='-')
    ax3.plot(c.h_range, r2_val_accuracies, marker="s", label="Triple ResNet", color=color_triple, linestyle='-')
    ax3.plot(c.h_range, r3_val_accuracies, marker="^", label="Quadruple ResNet", color=color_quadruple, linestyle='-')
    ax3.set_xlabel("Hidden size parameter h")
    ax3.set_ylabel("Mean validation accuracy", color='black')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    plt.suptitle(f"ResNet Comparison: Double vs Triple vs Quadruple Residual Blocks (d = {c.d}, n = {c.n})")
    plt.tight_layout()
    plt.show()


def build_down_rank_resnet(c: EmpiricalConfig, down_rank_dim: int, device: torch.device, h_list: list[int]) -> nn.Module:
    """Build a ResNet with specified h_list and variable down_rank_dim."""
    return ResNetStandard(c=c, h_list=h_list, down_rank_dim=down_rank_dim).to(device)


def build_d_model_resnet(c: EmpiricalConfig, d_model: int, device: torch.device, h_list: list[int]) -> nn.Module:
    """Build a ResNet with specified h_list and variable d_model."""
    # Create a copy of the config with the new d_model
    from copy import copy
    c_modified = copy(c)
    c_modified.d_model = d_model
    return ResNetStandard(c=c_modified, h_list=h_list).to(device)


def run_down_rank_experiment(device, problem, validation_set, c: EmpiricalConfig, h_list: list[int]):
    """Train and display variance quantities for ResNet with varying down_rank_dim."""
    print("Running Down-Rank ResNet Experiment")
    
    # Create a wrapper function that captures h_list
    def build_model_with_h_list(c: EmpiricalConfig, down_rank_dim: int, device: torch.device):
        return build_down_rank_resnet(c, down_rank_dim, device, h_list)
    
    model_vars, model_losses, model_val_accuracies, model_val_losses = ev.run_experiment(
        build_model_with_h_list, device, validation_set, problem, c
    )

    # plot results with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))
    
    # Subplot 1: Variance and Validation Loss (dual y-axis)
    color1 = 'tab:blue'
    ax1.set_ylabel("Mean variance of validation logits", color=color1)
    ax1.plot(c.h_range, model_vars, marker="o", label="Variance", color=color1, linestyle='-')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, alpha=0.3)
    
    # Create second y-axis for validation loss
    ax1_twin = ax1.twinx()
    color2 = 'tab:red'
    ax1_twin.set_ylabel("Mean validation loss", color=color2)
    ax1_twin.plot(c.h_range, model_val_losses, marker="s", label="Validation Loss", color=color2, linestyle='--')
    ax1_twin.tick_params(axis='y', labelcolor=color2)
    
    # Combine legends for first subplot
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    # Subplot 2: Training Loss
    ax2.plot(c.h_range, model_losses, marker="^", label="Training Loss", color='orange', linestyle='-')
    ax2.set_ylabel("Mean training loss", color='orange')
    ax2.tick_params(axis='y', labelcolor='orange')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Subplot 3: Validation Accuracy
    ax3.plot(c.h_range, model_val_accuracies, marker="v", label="Validation Accuracy", color='green', linestyle='-')
    ax3.set_xlabel("Down-rank dimension")
    ax3.set_ylabel("Mean validation accuracy", color='green')
    ax3.tick_params(axis='y', labelcolor='green')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    plt.suptitle(f"ResNet Down-Rank Experiment: Variance, validation loss, training loss, and validation accuracy vs. down-rank dim (d = {c.d}, h_list = {h_list})")
    plt.tight_layout()
    plt.show()


def run_d_model_experiment(device, problem, validation_set, c: EmpiricalConfig, h_list: list[int]):
    """Train and display variance quantities for ResNet with varying d_model."""
    print("Running D-Model ResNet Experiment")
    
    # Create a wrapper function that captures h_list
    def build_model_with_h_list(c: EmpiricalConfig, d_model: int, device: torch.device):
        return build_d_model_resnet(c, d_model, device, h_list)
    
    model_vars, model_losses, model_val_accuracies, model_val_losses = ev.run_experiment(
        build_model_with_h_list, device, validation_set, problem, c
    )

    # plot results with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))
    
    # Subplot 1: Variance and Validation Loss (dual y-axis)
    color1 = 'tab:blue'
    ax1.set_ylabel("Mean variance of validation logits", color=color1)
    ax1.plot(c.h_range, model_vars, marker="o", label="Variance", color=color1, linestyle='-')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, alpha=0.3)
    
    # Create second y-axis for validation loss
    ax1_twin = ax1.twinx()
    color2 = 'tab:red'
    ax1_twin.set_ylabel("Mean validation loss", color=color2)
    ax1_twin.plot(c.h_range, model_val_losses, marker="s", label="Validation Loss", color=color2, linestyle='--')
    ax1_twin.tick_params(axis='y', labelcolor=color2)
    
    # Combine legends for first subplot
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    # Subplot 2: Training Loss
    ax2.plot(c.h_range, model_losses, marker="^", label="Training Loss", color='orange', linestyle='-')
    ax2.set_ylabel("Mean training loss", color='orange')
    ax2.tick_params(axis='y', labelcolor='orange')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Subplot 3: Validation Accuracy
    ax3.plot(c.h_range, model_val_accuracies, marker="v", label="Validation Accuracy", color='green', linestyle='-')
    ax3.set_xlabel("Model dimension (d_model)")
    ax3.set_ylabel("Mean validation accuracy", color='green')
    ax3.tick_params(axis='y', labelcolor='green')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    plt.suptitle(f"ResNet D-Model Experiment: Variance, validation loss, training loss, and validation accuracy vs. d_model (d = {c.d}, h_list = {h_list})")
    plt.tight_layout()
    plt.show()

