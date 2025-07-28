from dataclasses import dataclass

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from src import dataset_creator
from src.learned_dropout.models_standard import ResNetStandard
import src.learned_dropout.empirical_variance as ev


@dataclass
class Config:
    """Configuration parameters for ResNet experiments."""
    d: int
    n_val: int
    n: int
    batch_size: int
    layer_norm: str
    h_range: list[int]
    num_runs: int
    lr: float
    epochs: int


def build_single_resnet(d: int, h: int, device: torch.device, layer_norm: str) -> nn.Module:
    """Build a ResNet with a single residual block: [d] -> ResidualBlock(d, h) -> [d] -> Linear(d, 1)"""
    # Single residual block with hidden dimension h
    h_list = [h]
    return ResNetStandard(d=d, h_list=h_list, relus=True, layer_norm=layer_norm).to(device)


def build_double_resnet(d: int, h: int, device: torch.device, layer_norm: str) -> nn.Module:
    """Build a ResNet with two residual blocks: [d] -> ResidualBlock(d, h) -> ResidualBlock(d, h) -> [d] -> Linear(d, 1)"""
    # Two residual blocks with hidden dimension h
    h_list = [h, h]
    return ResNetStandard(d=d, h_list=h_list, relus=True, layer_norm=layer_norm).to(device)


def build_triple_resnet(d: int, h: int, device: torch.device, layer_norm: str) -> nn.Module:
    """Build a ResNet with three residual blocks: [d] -> ResidualBlock(d, h) -> ResidualBlock(d, h) -> ResidualBlock(d, h) -> [d] -> Linear(d, 1)"""
    # Three residual blocks with hidden dimension h
    h_list = [h, h, h]
    return ResNetStandard(d=d, h_list=h_list, relus=True, layer_norm=layer_norm).to(device)


def build_quadruple_resnet(d: int, h: int, device: torch.device, layer_norm: str) -> nn.Module:
    """Build a ResNet with four residual blocks: [d] -> ResidualBlock(d, h) -> ResidualBlock(d, h) -> ResidualBlock(d, h) -> ResidualBlock(d, h) -> [d] -> Linear(d, 1)"""
    # Four residual blocks with hidden dimension h
    h_list = [h, h, h, h]
    return ResNetStandard(d=d, h_list=h_list, relus=True, layer_norm=layer_norm).to(device)


def run_resnet_experiment(device, problem, validation_set, c: Config):
    """Train and display variance quantities for single-layer ResNet."""
    print("Running Single-Layer ResNet Experiment")
    model_vars, model_losses = ev.run_experiment(
        build_single_resnet, c.n, c.d, c.h_range, c.num_runs, c.lr, c.epochs, c.batch_size, device, c.layer_norm, validation_set, problem
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


def three_resnet_experiment(device, problem, validation_set, c: Config):
    """Train and compare three ResNet models: double vs triple vs quadruple residual blocks."""
    print("Running ResNet 1 (double residual blocks)")
    r1_vars, r1_losses = ev.run_experiment(
        build_double_resnet, c.n, c.d, c.h_range, c.num_runs, c.lr, c.epochs, c.batch_size, device, c.layer_norm, validation_set, problem
    )

    print("\nRunning ResNet 2 (triple residual blocks)")
    r2_vars, r2_losses = ev.run_experiment(
        build_triple_resnet, c.n, c.d, c.h_range, c.num_runs, c.lr, c.epochs, c.batch_size, device, c.layer_norm, validation_set, problem
    )

    print("\nRunning ResNet 3 (quadruple residual blocks)")
    r3_vars, r3_losses = ev.run_experiment(
        build_quadruple_resnet, c.n, c.d, c.h_range, c.num_runs, c.lr, c.epochs, c.batch_size, device, c.layer_norm, validation_set, problem
    )

    # plot results with dual y-axes
    fig, ax1 = plt.subplots(figsize=(12, 8))
    
    # Plot variance on left y-axis
    color_double = 'tab:blue'
    color_triple = 'tab:green'
    color_quadruple = 'tab:red'
    ax1.set_xlabel("Hidden size parameter h")
    ax1.set_ylabel("Mean variance of validation logits", color='black')
    ax1.plot(c.h_range, r1_vars, marker="o", label="Double ResNet", color=color_double, linestyle='-')
    ax1.plot(c.h_range, r2_vars, marker="s", label="Triple ResNet", color=color_triple, linestyle='-')
    ax1.plot(c.h_range, r3_vars, marker="^", label="Quadruple ResNet", color=color_quadruple, linestyle='-')
    ax1.grid(True, alpha=0.3)
    
    # Create second y-axis for training loss
    ax2 = ax1.twinx()
    color_train = 'tab:orange'
    ax2.set_ylabel("Mean training loss", color=color_train)
    ax2.plot(c.h_range, r1_losses, marker="v", label="Double ResNet (train)", color=color_double, linestyle='--', alpha=0.7)
    ax2.plot(c.h_range, r2_losses, marker="<", label="Triple ResNet (train)", color=color_triple, linestyle='--', alpha=0.7)
    ax2.plot(c.h_range, r3_losses, marker=">", label="Quadruple ResNet (train)", color=color_quadruple, linestyle='--', alpha=0.7)
    ax2.tick_params(axis='y', labelcolor=color_train)
    
    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    plt.title(f"ResNet Comparison: Double vs Triple vs Quadruple Residual Blocks (d = {c.d}, n = {c.n})")
    plt.tight_layout()
    plt.show()


# def main():
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     
#     # Configuration parameters - all explicitly defined here
#     c = Config(
#         d=10,
#         n_val=1000,
#         n=100,
#         batch_size=100,
#         layer_norm="layer_norm",
#         h_range=list(range(4, 101, 4)),  # 4, 8, 12, ..., 100
#         num_runs=20,
#         lr=3e-3,
#         epochs=100
#     )
#     
#     # prepare validation set (same across runs)
#     problem = dataset_creator.Gaussian(d=c.d, perfect_class_balance=False)
#     x_val, y_val = problem.generate_dataset(c.n_val, shuffle=True)
#     validation_set = x_val.to(device), y_val.to(device)
# 
#     # Run three ResNet comparison experiment
#     three_resnet_experiment(device, problem, validation_set, c)
#     
#     # Run single ResNet experiment (commented out since we're focusing on comparison)
#     # run_resnet_experiment(device, problem, validation_set, c)


def build_down_rank_resnet(d: int, down_rank_dim: int, device: torch.device, layer_norm: str) -> nn.Module:
    """Build a ResNet with fixed h_list=[150, 150] and variable down_rank_dim."""
    # Fixed architecture: two residual blocks with hidden dimension 150
    h_list = [150, 150]
    return ResNetStandard(d=d, h_list=h_list, relus=True, layer_norm=layer_norm, down_rank_dim=down_rank_dim).to(device)


def run_down_rank_experiment(device, problem, validation_set, c: Config):
    """Train and display variance quantities for ResNet with varying down_rank_dim."""
    print("Running Down-Rank ResNet Experiment")
    model_vars, model_losses = ev.run_experiment(
        build_down_rank_resnet, c.n, c.d, c.h_range, c.num_runs, c.lr, c.epochs, c.batch_size, device, c.layer_norm, validation_set, problem
    )

    # plot results with dual y-axes
    fig, ax1 = plt.subplots(figsize=(12, 8))
    
    # Plot variance on left y-axis
    color = 'tab:blue'
    ax1.set_xlabel("Down-rank dimension")
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
    
    plt.title(f"ResNet Down-Rank Experiment: Variance and training loss vs. down-rank dim (d = {c.d}, h_list = [150, 150])")
    plt.tight_layout()
    plt.show()


def main():
    torch.manual_seed(38173)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Configuration parameters for down-rank experiment
    c = Config(
        d=10,
        n_val=1000,
        n=100,
        batch_size=100,
        layer_norm="layer_norm",
        h_range=list(range(20, 31)),  # 10, 11, 12, ..., 20 (down_rank_dim values)
        num_runs=20,
        lr=3e-3,
        epochs=100
    )
    
    # prepare validation set (same across runs)
    problem = dataset_creator.Gaussian(d=c.d, perfect_class_balance=False)
    x_val, y_val = problem.generate_dataset(c.n_val, shuffle=True)
    validation_set = x_val.to(device), y_val.to(device)

    # Run down-rank experiment
    run_down_rank_experiment(device, problem, validation_set, c)


if __name__ == "__main__":
    main() 