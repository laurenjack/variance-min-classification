import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from src import dataset_creator
from src.learned_dropout.models_standard import MLPStandard
import src.learned_dropout.empirical_variance as ev


def build_model_m1(d: int, h: int, device: torch.device, layer_norm: bool) -> nn.Module:
    """Model 1: single-hidden-layer MLPStandard: [d, h, 1] with LayerNorm and ReLU"""
    # hidden dims list for MLPStandard
    h_list = [h, d]
    return MLPStandard(d=d, h_list=h_list, relus=True, layer_norm=layer_norm).to(device)


def build_model_m2(d: int, h: int, device: torch.device, layer_norm: bool) -> nn.Module:
    """Model 2: two-hidden-layer MLPStandard: [d, h//2, d, 1] with LayerNorm and ReLU"""
    # hidden dims list for MLPStandard
    h_list = [h // 2, d, h // 2, d]
    return MLPStandard(d=d, h_list=h_list, relus=True, layer_norm=layer_norm).to(device)

def build_model_m3(d: int, h: int, device: torch.device, layer_norm: bool) -> nn.Module:
    """Model 3: four-hidden-layer MLPStandard: [d, h//4, d, h//4, d, h//4, d, h//4, d, 1] with LayerNorm and ReLU"""
    # hidden dims list for MLPStandard
    h_list = [h // 4, d, h // 4, d, h // 4, d, h // 4, d]
    return MLPStandard(d=d, h_list=h_list, relus=True, layer_norm=layer_norm).to(device)

def build_single_layer_mlp(d, h, device: torch.device, layer_norm: bool) -> nn.Module:
    return MLPStandard(d=d, h_list=[h], relus=True, layer_norm=layer_norm).to(device)

def build_single_layer_mlp_down_rank(d, h, device: torch.device, layer_norm: bool) -> nn.Module:
    return MLPStandard(d=d, h_list=[h], relus=True, down_rank_dim=10, layer_norm=layer_norm).to(device)


def build_single_layer_mlp_down_rank_up_proj(d, h, device: torch.device, layer_norm: bool) -> nn.Module:
    return MLPStandard(d=d, h_list=[h], relus=True, down_rank_dim=10, up_proj=True, layer_norm=layer_norm).to(device)


def three_model_experiment(device, problem, validation_set, d: int = 40, layer_norm: bool = True):
    # configuration
    h_range = range(4, 81, 4)  # 4, 6, 8, ..., 20
    num_runs = 20
    lr = 3e-3
    # lr_base = 3e-4
    epochs = 1
    n = 3000
    batch_size = 30

    # run experiments for different model architectures
    print("Running Model 1 (single layer)")
    m1_vars, m1_losses = ev.run_experiment(
        build_model_m1, n, d, h_range, num_runs, lr, epochs, batch_size, device,
        layer_norm, validation_set, problem
    )

    print("\nRunning Model 2 (two layers)")
    m2_vars, m2_losses = ev.run_experiment(
        build_model_m2, n, d, h_range, num_runs, lr, epochs, batch_size, device, layer_norm,
        validation_set, problem
    )

    print("\nRunning Model 3 (four layers)")
    m3_vars, m3_losses = ev.run_experiment(
        build_model_m3, n, d, h_range, num_runs, lr, epochs, batch_size, device, layer_norm,
        validation_set, problem
    )

    # plot results
    fig, ax1 = plt.subplots(figsize=(12, 8))
    
    # Plot variance
    ax1.set_xlabel("Hidden size parameter h")
    ax1.set_ylabel("Mean variance of validation logits")
    ax1.plot(list(h_range), m1_vars, marker="o", label="Model 1 Variance", color='blue', linestyle='-')
    ax1.plot(list(h_range), m2_vars, marker="s", label="Model 2 Variance", color='green', linestyle='-')
    ax1.plot(list(h_range), m3_vars, marker="^", label="Model 3 Variance", color='red', linestyle='-')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right')
    
    plt.title(f"Model comparison: variance vs. hidden size (d = {d}, n = {n})")
    plt.tight_layout()
    plt.show()


def train_single_model(build_model_func, device, problem, validation_set, n: int, d: int, batch_size: int, layer_norm: bool = True):
    """Train and display variance quantities for a given model."""
    # configuration
    h_range = range(4, 41, 4)  # 4, 6, 8, ..., 20
    num_runs = 20
    lr = 3e-3
    epochs = 100

    print("Running Model Experiment")
    model_vars, model_losses = ev.run_experiment(
        build_model_func, n, d, h_range, num_runs, lr, epochs, batch_size, device, layer_norm, validation_set, problem
    )

    # plot results with dual y-axes
    fig, ax1 = plt.subplots(figsize=(12, 8))
    
    # Plot variance on left y-axis
    color = 'tab:blue'
    ax1.set_xlabel("Hidden size parameter h")
    ax1.set_ylabel("Mean variance of validation logits", color=color)
    ax1.plot(list(h_range), model_vars, marker="o", label="Variance", color='blue', linestyle='-')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)
    
    # Create second y-axis for training loss
    ax2 = ax1.twinx()
    color = 'tab:orange'
    ax2.set_ylabel("Mean training loss", color=color)
    ax2.plot(list(h_range), model_losses, marker="s", label="Training Loss", color='orange', linestyle='--')
    ax2.tick_params(axis='y', labelcolor=color)
    
    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    plt.title(f"Model: -Variance and training loss vs. hidden size (d = {d}, n = {n})")
    plt.tight_layout()
    plt.show()


def three_mlp_experiment(device, problem, validation_set, n: int, d: int, batch_size: int, layer_norm: bool = True):
    """Train and compare three single-layer MLP models: standard vs down-rank vs down-rank with up-proj."""
    # configuration
    h_range = range(4, 201, 8)  # 4, 6, 8, ..., 40
    num_runs = 20
    lr = 3e-3
    epochs = 100

    print("Running Model 1 (standard single layer)")
    m1_vars, m1_losses = ev.run_experiment(
        build_single_layer_mlp, n, d, h_range, num_runs, lr, epochs, batch_size, device, layer_norm, validation_set, problem
    )

    print("\nRunning Model 2 (down-rank single layer)")
    m2_vars, m2_losses = ev.run_experiment(
        build_single_layer_mlp_down_rank, n, d, h_range, num_runs, lr, epochs, batch_size, device, layer_norm, validation_set, problem
    )

    print("\nRunning Model 3 (down-rank with up-proj single layer)")
    m3_vars, m3_losses = ev.run_experiment(
        build_single_layer_mlp_down_rank_up_proj, n, d, h_range, num_runs, lr, epochs, batch_size, device, layer_norm, validation_set, problem
    )

    # plot results with dual y-axes
    fig, ax1 = plt.subplots(figsize=(12, 8))
    
    # Plot variance on left y-axis
    color_std = 'tab:blue'
    color_down = 'tab:green'
    color_up = 'tab:red'
    ax1.set_xlabel("Hidden size parameter h")
    ax1.set_ylabel("Mean validation loss", color='black')
    ax1.plot(list(h_range), m1_vars, marker="o", label="Standard MLP", color=color_std, linestyle='-')
    ax1.plot(list(h_range), m2_vars, marker="s", label="Down-rank MLP", color=color_down, linestyle='-')
    ax1.plot(list(h_range), m3_vars, marker="^", label="Down-rank + Up-proj MLP", color=color_up, linestyle='-')
    ax1.grid(True, alpha=0.3)
    
    # Create second y-axis for training loss
    ax2 = ax1.twinx()
    color_train = 'tab:orange'
    ax2.set_ylabel("Mean training loss", color=color_train)
    ax2.plot(list(h_range), m1_losses, marker="v", label="Standard MLP (train)", color=color_std, linestyle='--', alpha=0.7)
    ax2.plot(list(h_range), m2_losses, marker="<", label="Down-rank MLP (train)", color=color_down, linestyle='--', alpha=0.7)
    ax2.plot(list(h_range), m3_losses, marker=">", label="Down-rank + Up-proj MLP (train)", color=color_up, linestyle='--', alpha=0.7)
    ax2.tick_params(axis='y', labelcolor=color_train)
    
    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    plt.title(f"MLP Comparison: Standard vs Down-rank vs Down-rank+Up-proj (d = {d}, n = {n})")
    plt.tight_layout()
    plt.show()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # prepare validation set (same across runs)
    d = 10
    n_val = 1000
    problem = dataset_creator.Gaussian(d=d, perfect_class_balance=False)
    x_val, y_val = problem.generate_dataset(n_val, shuffle=True)
    validation_set = x_val.to(device), y_val.to(device)

    # Run three model comparison experiment
    three_mlp_experiment(device, problem, validation_set, n=100, d=d, batch_size=100, layer_norm=True)
    
    # Run single model experiment
    # train_single_model(build_single_layer_mlp, device, problem, validation_set, n=100, d=d, batch_size=100, layer_norm=True)
    
    # Run three model experiment
    # three_model_experiment(device, problem, validation_set, d=d, layer_norm=True)


if __name__ == "__main__":
    main() 