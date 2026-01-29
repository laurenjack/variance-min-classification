from dataclasses import dataclass
from typing import Tuple, Optional

import torch
from torch import Tensor
import matplotlib.pyplot as plt
import matplotlib as mpl

from jl.config import Config
from jl.multi_runner import train_and_compute_metrics


@dataclass
class GraphConfig:
    constant_name: Optional[str] = None
    constant_value: Optional[float] = None
    show_validation_loss: bool = True
    show_training_loss: bool = True
    show_validation_error: bool = True
    include_variance: bool = False
    show_line: bool = True


def run_list_experiment(
    device,
    problem,
    validation_set: Tuple[Tensor, Tensor, Tensor],
    configs: list[Config],
    width_range: list[int],
    num_runs: int,
    clean_mode: bool,
    graph_config: Optional[GraphConfig] = None,
):
    """
    Train and compare models, plotting loss, error, and optionally variance.

    Args:
        device: torch device
        problem: Dataset generator with generate_dataset method
        validation_set: Tuple of (x_val, y_val, center_indices)
        configs: List of Config objects to compare
        width_range: List of widths to vary
        num_runs: Number of runs per width
        clean_mode: Whether to generate clean data
        graph_config: Configuration for graph display options
    """
    if graph_config is None:
        graph_config = GraphConfig()

    results = []
    for i, c in enumerate(configs):
        print(f"Running experiment {i + 1} with {c.model_type}")
        result = _run_experiments(
            device,
            validation_set,
            problem,
            c,
            num_runs,
            clean_mode,
            width_range,
            graph_config.include_variance,
        )
        results.append(result)
        print()

    _plot_results(results, width_range, configs, graph_config)


def _run_experiments(
    device: torch.device,
    validation_set: Tuple[Tensor, Tensor, Tensor],
    problem,
    c: Config,
    num_runs: int,
    clean_mode: bool,
    width_range: list[int],
    include_variance: bool,
) -> tuple:
    """Internal: Run experiments and return metrics (with or without variance)."""
    x_val, y_val, center_indices_val = validation_set

    train_loss_per_w, val_acc_per_w, val_loss_per_w, val_logits_all = train_and_compute_metrics(
        device=device,
        problem=problem,
        c=c,
        num_runs=num_runs,
        clean_mode=clean_mode,
        x_val=x_val,
        y_val=y_val,
        width_range=width_range,
    )

    mean_train_losses = [v.item() for v in train_loss_per_w]
    mean_val_errors = [1.0 - v.item() for v in val_acc_per_w]
    mean_val_losses = [v.item() for v in val_loss_per_w]

    if include_variance:
        # Variance of validation logits across runs, grouped by center
        num_widths = val_logits_all.shape[0]
        num_runs_actual = val_logits_all.shape[1]
        unique_centers = torch.unique(center_indices_val)
        num_centers = len(unique_centers)

        center_variances = []
        for center_idx in range(num_centers):
            center_mask = (center_indices_val == center_idx)
            if c.num_class == 2:
                center_logits = val_logits_all[:, :, center_mask]
                center_var = center_logits.var(dim=1, unbiased=False)
                center_variances.append(center_var.mean(dim=1))
            else:
                center_logits = val_logits_all[:, :, center_mask, :]
                center_var = center_logits.var(dim=1, unbiased=False)
                center_variances.append(center_var.mean(dim=(1, 2)))

        mean_var_per_w = torch.stack(center_variances, dim=1).mean(dim=1)
        mean_vars = [v.item() for v in mean_var_per_w]

        if c.width_varyer is None:
            print(f"mean variance = {mean_vars[0]:.4e} | mean train loss = {mean_train_losses[0]:.4f} | mean val error = {mean_val_errors[0]:.4f} | mean val loss = {mean_val_losses[0]:.4f}")
        else:
            for i, w in enumerate(width_range):
                print(f"width = {w:2d} | mean variance = {mean_vars[i]:.4e} | mean train loss = {mean_train_losses[i]:.4f} | mean val error = {mean_val_errors[i]:.4f} | mean val loss = {mean_val_losses[i]:.4f}")

        return (mean_vars, mean_train_losses, mean_val_errors, mean_val_losses)
    else:
        if c.width_varyer is None:
            print(f"mean train loss = {mean_train_losses[0]:.4f} | mean val error = {mean_val_errors[0]:.4f} | mean val loss = {mean_val_losses[0]:.4f}")
        else:
            for i, w in enumerate(width_range):
                print(f"width = {w:2d} | mean train loss = {mean_train_losses[i]:.4f} | mean val error = {mean_val_errors[i]:.4f} | mean val loss = {mean_val_losses[i]:.4f}")

        return (mean_train_losses, mean_val_errors, mean_val_losses)


def _setup_academic_style():
    """Configure matplotlib for academic paper aesthetics."""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 12,
        'axes.linewidth': 0.8,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'lines.linewidth': 1.5,
        'lines.markersize': 6,
    })
    

def _plot_results(
    results: list,
    width_range: list[int],
    configs: list[Config],
    graph_config: GraphConfig,
):
    """Plot experiment results with configurable subplots."""
    _setup_academic_style()

    num_configs = len(configs)
    show_legend = num_configs > 1

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']

    linestyle = '-' if graph_config.show_line else 'none'

    # Determine which plots to show
    plots_to_show = []
    if graph_config.include_variance:
        plots_to_show.append('variance')
    if graph_config.show_validation_loss:
        plots_to_show.append('val_loss')
    if graph_config.show_training_loss:
        plots_to_show.append('train_loss')
    if graph_config.show_validation_error:
        plots_to_show.append('val_error')

    num_plots = len(plots_to_show)
    if num_plots == 0:
        return

    fig, axes = plt.subplots(num_plots, 1, figsize=(8, 3 * num_plots))
    if num_plots == 1:
        axes = [axes]

    width_param = configs[0].width_varyer if configs[0].width_varyer else "width"

    for ax_idx, plot_type in enumerate(plots_to_show):
        ax = axes[ax_idx]

        if plot_type == 'variance':
            ax.set_ylabel("Variance of validation logits")
            for i, result in enumerate(results):
                vars_ = result[0]
                color = colors[i % len(colors)]
                marker = markers[i % len(markers)]
                label = f"Config {i+1}" if show_legend else None
                ax.plot(width_range, vars_, marker=marker, label=label, color=color,
                       linestyle=linestyle, markerfacecolor='white', markeredgewidth=1.5)
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
            if show_legend:
                ax.legend(frameon=True, fancybox=False, edgecolor='#cccccc')

        elif plot_type == 'val_loss':
            ax.set_ylabel("Validation loss")
            for i, result in enumerate(results):
                if graph_config.include_variance:
                    val_losses = result[3]
                else:
                    val_losses = result[2]
                color = colors[i % len(colors)]
                marker = markers[i % len(markers)]
                label = f"Config {i+1}" if show_legend else None
                ax.plot(width_range, val_losses, marker=marker, label=label, color=color,
                       linestyle=linestyle, markerfacecolor='white', markeredgewidth=1.5)
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
            if show_legend:
                ax.legend(frameon=True, fancybox=False, edgecolor='#cccccc')

        elif plot_type == 'train_loss':
            ax.set_ylabel("Training loss")
            for i, result in enumerate(results):
                if graph_config.include_variance:
                    losses = result[1]
                else:
                    losses = result[0]
                color = colors[i % len(colors)]
                marker = markers[i % len(markers)]
                label = f"Config {i+1}" if show_legend else None
                ax.plot(width_range, losses, marker=marker, label=label, color=color,
                       linestyle=linestyle, markerfacecolor='white', markeredgewidth=1.5)
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
            if show_legend:
                ax.legend(frameon=True, fancybox=False, edgecolor='#cccccc')

        elif plot_type == 'val_error':
            ax.set_ylabel("Validation error")
            for i, result in enumerate(results):
                if graph_config.include_variance:
                    val_errors = result[2]
                else:
                    val_errors = result[1]
                color = colors[i % len(colors)]
                marker = markers[i % len(markers)]
                label = f"Config {i+1}" if show_legend else None
                ax.plot(width_range, val_errors, marker=marker, label=label, color=color,
                       linestyle=linestyle, markerfacecolor='white', markeredgewidth=1.5)
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

            if graph_config.constant_name is not None and graph_config.constant_value is not None:
                ax.axhline(y=graph_config.constant_value, color='red', linestyle='--',
                          linewidth=1.5, label=graph_config.constant_name)

            if show_legend or (graph_config.constant_name is not None and graph_config.constant_value is not None):
                ax.legend(frameon=True, fancybox=False, edgecolor='#cccccc')

    axes[-1].set_xlabel(f"Width parameter: {width_param}")

    plt.tight_layout()
    plt.show()
