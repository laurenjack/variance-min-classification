import torch
import matplotlib.pyplot as plt

from src import dataset_creator
import src.learned_dropout.empirical_variance as ev
from src.learned_dropout.config import Config



def run_list_experiment(
    device,
    problem,
    validation_set,
    configs: list[Config],
    width_range: list[int],
    num_runs: int,
    use_percent_correct: bool,
):
    """Train and compare models (Resnet or MLP) using parallel experiment."""
    
    # Run experiments for each config
    results = []
    for i, c in enumerate(configs):
        print(f"Running experiment {i + 1} with {c.model_type}")
        vars_, losses, val_accuracies, val_losses = ev.run_experiment_parallel(
            device,
            validation_set,
            problem,
            c,
            width_range,
            num_runs,
            use_percent_correct,
        )
        results.append((vars_, losses, val_accuracies, val_losses))
        print()
    
    # Create dynamic color palette
    colors = plt.cm.tab10(range(len(configs)))
    markers = ['o', 's', '^', 'v', '<', '>', 'D', 'p', '*', 'h']
    
    # plot results with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))
    
    # Subplot 1: Variance and Validation Loss (dual y-axis)
    ax1.set_ylabel("Mean variance of validation logits", color='black')
    for i, (vars_, _, _, val_losses) in enumerate(results):
        color = colors[i]
        marker = markers[i % len(markers)]
        ax1.plot(width_range, vars_, marker=marker, label=f"Config {i+1}", color=color, linestyle='-')
    ax1.grid(True, alpha=0.3)
    
    # Create second y-axis for validation loss
    ax1_twin = ax1.twinx()
    color_val = 'tab:orange'
    ax1_twin.set_ylabel("Mean validation loss", color=color_val)
    for i, (_, _, _, val_losses) in enumerate(results):
        color = colors[i]
        marker = markers[i % len(markers)]
        ax1_twin.plot(width_range, val_losses, marker=marker, label=f"Config {i+1} (val)", 
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
        ax2.plot(width_range, losses, marker=marker, label=f"Config {i+1}", color=color, linestyle='-')
    ax2.set_ylabel("Mean training loss", color='black')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Subplot 3: Validation Accuracy
    for i, (_, _, val_accuracies, _) in enumerate(results):
        color = colors[i]
        marker = markers[i % len(markers)]
        ax3.plot(width_range, val_accuracies, marker=marker, label=f"Config {i+1}", color=color, linestyle='-')
    
    # Determine x-axis label based on width_varyer
    width_param = configs[0].width_varyer if configs[0].width_varyer else "width"
    ax3.set_xlabel(f"Width parameter: {width_param}")
    ax3.set_ylabel("Mean validation accuracy", color='black')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    config_names = " vs ".join([f"Config {i+1} ({configs[i].model_type})" for i in range(len(configs))])
    plt.suptitle(f"Model Comparison: {config_names} (d = {configs[0].d}, n = {configs[0].n})")
    plt.tight_layout()
    plt.show()
