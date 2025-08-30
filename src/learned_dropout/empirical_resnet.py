from typing import Optional
from copy import deepcopy

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from src import dataset_creator
from src.learned_dropout.models_standard import ResNetStandard
import src.learned_dropout.empirical_variance as ev
from src.learned_dropout.config import EmpiricalConfig


class ResNetBuilder:
    """A parameterized ResNet builder that can create ResNets with different numbers of layers."""
    
    def __init__(self, num_layers: int, d_model: Optional[int] = None, down_rank_dim: Optional[int] = None):
        """
        Args:
            num_layers: Number of residual blocks to include
        """
        self.num_layers = num_layers
        self.d_model = d_model
        self.down_rank_dim = down_rank_dim

    def build(self, c: EmpiricalConfig, h: int, device: torch.device) -> nn.Module:
        """Build a ResNet with the configured number of residual blocks.
        
        Args:
            c: Configuration object
            h: Hidden dimension for each residual block
            device: Device to place the model on
            
        Returns:
            ResNet model with the specified architecture
        """
        h_list = [h] * self.num_layers
        if self.d_model is not None:
            c = deepcopy(c)
            c.d_model = self.d_model
        return ResNetStandard(c=c, h_list=h_list, down_rank_dim=self.down_rank_dim).to(device)



def run_list_resnet_experiment(device, problem, validation_set, c: EmpiricalConfig, percent_correct: float, resnet_builders: list):
    """Train and compare multiple ResNet models using provided ResNetBuilder objects."""
    
    # Run experiments for each builder
    results = []
    for i, builder in enumerate(resnet_builders):
        print(f"Running ResNet {i + 1}")
        vars_, losses, val_accuracies, val_losses = ev.run_experiment_parallel(
            builder, device, validation_set, problem, c, percent_correct
        )
        results.append((vars_, losses, val_accuracies, val_losses))
        print()
    
    # Create dynamic color palette
    colors = plt.cm.tab10(range(len(resnet_builders)))
    markers = ['o', 's', '^', 'v', '<', '>', 'D', 'p', '*', 'h']
    
    # plot results with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))
    
    # Subplot 1: Variance and Validation Loss (dual y-axis)
    ax1.set_ylabel("Mean variance of validation logits", color='black')
    for i, (vars_, _, _, val_losses) in enumerate(results):
        color = colors[i]
        marker = markers[i % len(markers)]
        ax1.plot(c.h_range, vars_, marker=marker, label=f"ResNet {i+1}", color=color, linestyle='-')
    ax1.grid(True, alpha=0.3)
    
    # Create second y-axis for validation loss
    ax1_twin = ax1.twinx()
    color_val = 'tab:orange'
    ax1_twin.set_ylabel("Mean validation loss", color=color_val)
    for i, (_, _, _, val_losses) in enumerate(results):
        color = colors[i]
        marker = markers[i % len(markers)]
        ax1_twin.plot(c.h_range, val_losses, marker=marker, label=f"ResNet {i+1} (val)", 
                     color=color, linestyle='--', alpha=0.7)
    ax1_twin.tick_params(axis='y', labelcolor=color_val)
    
    # Combine legends for first subplot
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    # Subplot 2: Training Loss
    for i, (_, losses, _, _) in enumerate(results):
        color = colors[i]
        marker = markers[i % len(markers)]
        ax2.plot(c.h_range, losses, marker=marker, label=f"ResNet {i+1}", color=color, linestyle='-')
    ax2.set_ylabel("Mean training loss", color='black')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Subplot 3: Validation Accuracy
    for i, (_, _, val_accuracies, _) in enumerate(results):
        color = colors[i]
        marker = markers[i % len(markers)]
        ax3.plot(c.h_range, val_accuracies, marker=marker, label=f"ResNet {i+1}", color=color, linestyle='-')
    ax3.set_xlabel("Hidden size parameter h")
    ax3.set_ylabel("Mean validation accuracy", color='black')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    resnet_names = " vs ".join([f"ResNet {i+1}" for i in range(len(resnet_builders))])
    plt.suptitle(f"ResNet Comparison: {resnet_names} (d = {c.d}, n = {c.n})")
    plt.tight_layout()
    plt.show()
