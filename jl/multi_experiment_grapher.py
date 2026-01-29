from typing import Tuple, Optional

import torch
from torch import Tensor
import matplotlib.pyplot as plt

from jl.config import Config
from jl.multi_runner import train_and_compute_metrics


def run_list_experiment(
    device,
    problem,
    validation_set: Tuple[Tensor, Tensor, Tensor],
    configs: list[Config],
    width_range: list[int],
    num_runs: int,
    clean_mode: bool,
    include_variance: bool = False,
):
    """
    Train and compare models, plotting loss, accuracy, and optionally variance.

    Args:
        device: torch device
        problem: Dataset generator with generate_dataset method
        validation_set: Tuple of (x_val, y_val, center_indices)
        configs: List of Config objects to compare
        width_range: List of widths to vary
        num_runs: Number of runs per width
        clean_mode: Whether to generate clean data
        include_variance: If True, compute and plot variance of validation logits
    """
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
            include_variance,
        )
        results.append(result)
        print()

    _plot_results(results, width_range, configs, include_variance)


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
    mean_val_accuracies = [v.item() for v in val_acc_per_w]
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
            print(f"mean variance = {mean_vars[0]:.4e} | mean train loss = {mean_train_losses[0]:.4f} | mean val acc = {mean_val_accuracies[0]:.4f} | mean val loss = {mean_val_losses[0]:.4f}")
        else:
            for i, w in enumerate(width_range):
                print(f"width = {w:2d} | mean variance = {mean_vars[i]:.4e} | mean train loss = {mean_train_losses[i]:.4f} | mean val acc = {mean_val_accuracies[i]:.4f} | mean val loss = {mean_val_losses[i]:.4f}")

        return (mean_vars, mean_train_losses, mean_val_accuracies, mean_val_losses)
    else:
        if c.width_varyer is None:
            print(f"mean train loss = {mean_train_losses[0]:.4f} | mean val acc = {mean_val_accuracies[0]:.4f} | mean val loss = {mean_val_losses[0]:.4f}")
        else:
            for i, w in enumerate(width_range):
                print(f"width = {w:2d} | mean train loss = {mean_train_losses[i]:.4f} | mean val acc = {mean_val_accuracies[i]:.4f} | mean val loss = {mean_val_losses[i]:.4f}")

        return (mean_train_losses, mean_val_accuracies, mean_val_losses)


def _build_title(configs: list[Config]) -> str:
    """Build comprehensive title string from config properties."""
    c = configs[0]
    title_parts = [
        f"model_type={c.model_type}",
        f"d={c.d}",
        f"n_val={c.n_val}",
        f"n={c.n}",
        f"batch_size={c.batch_size}",
        f"lr={c.lr}",
        f"epochs={c.epochs}",
        f"weight_decay={c.weight_decay}",
        f"num_layers={c.num_layers}",
    ]

    if c.h is not None:
        title_parts.append(f"h={c.h}")
    if c.d_model is not None:
        title_parts.append(f"d_model={c.d_model}")
    if c.width_varyer is not None:
        title_parts.append(f"width_varyer={c.width_varyer}")
    if c.c is not None:
        title_parts.append(f"c={c.c}")

    title_parts.append(f"is_norm={c.is_norm}")

    return " | ".join(title_parts)


def _plot_results(
    results: list,
    width_range: list[int],
    configs: list[Config],
    include_variance: bool,
):
    """Plot experiment results with configurable variance subplot."""
    colors = plt.cm.tab10(range(len(configs)))
    markers = ['o', 's', '^', 'v', '<', '>', 'D', 'p', '*', 'h']

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))

    if include_variance:
        # Subplot 1: Variance and Validation Loss (dual y-axis)
        ax1.set_ylabel("Mean variance of validation logits", color='black')
        for i, (vars_, _, _, val_losses) in enumerate(results):
            color = colors[i]
            marker = markers[i % len(markers)]
            ax1.plot(width_range, vars_, marker=marker, label=f"Config {i+1}", color=color, linestyle='-')
        ax1.grid(True, alpha=0.3)

        ax1_twin = ax1.twinx()
        ax1_twin.set_ylabel("Mean validation loss", color='tab:orange')
        for i, (_, _, _, val_losses) in enumerate(results):
            color = colors[i]
            marker = markers[i % len(markers)]
            ax1_twin.plot(width_range, val_losses, marker=marker, label=f"Config {i+1} (val)",
                         color=color, linestyle='--', alpha=0.7)
        ax1_twin.tick_params(axis='y', labelcolor='tab:orange')

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
    else:
        # Subplot 1: Validation Loss
        for i, (losses, val_accuracies, val_losses) in enumerate(results):
            color = colors[i]
            marker = markers[i % len(markers)]
            ax1.plot(width_range, val_losses, marker=marker, label=f"Config {i+1}", color=color, linestyle='-')
        ax1.set_ylabel("Mean validation loss", color='black')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Subplot 2: Training Loss
        for i, (losses, val_accuracies, val_losses) in enumerate(results):
            color = colors[i]
            marker = markers[i % len(markers)]
            ax2.plot(width_range, losses, marker=marker, label=f"Config {i+1}", color=color, linestyle='-')
        ax2.set_ylabel("Mean training loss", color='black')
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        # Subplot 3: Validation Accuracy
        for i, (losses, val_accuracies, val_losses) in enumerate(results):
            color = colors[i]
            marker = markers[i % len(markers)]
            ax3.plot(width_range, val_accuracies, marker=marker, label=f"Config {i+1}", color=color, linestyle='-')

    width_param = configs[0].width_varyer if configs[0].width_varyer else "width"
    ax3.set_xlabel(f"Width parameter: {width_param}")
    ax3.set_ylabel("Mean validation accuracy", color='black')
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    plt.suptitle(_build_title(configs), fontsize=9)
    plt.tight_layout()
    plt.show()
