from typing import Tuple, List, Union

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from torch.func import vmap, stack_module_state, functional_call, grad_and_value

from src import dataset_creator
from src.learned_dropout.config import Config
from src.learned_dropout.models import create_model


def _generate_training_sets(problem, c: Config, num_runs: int, device: torch.device, clean_mode: bool) -> List[Tuple[Tensor, Tensor]]:
    # Generate all training sets once, outside the loops
    training_sets = []
    for _ in range(num_runs):
        x_train, y_train, _ = problem.generate_dataset(c.n, shuffle=True, clean_mode=clean_mode)
        x_train, y_train = x_train.to(device), y_train.to(device)
        training_sets.append((x_train, y_train))
    return training_sets

def _build_models(c: Config, width_range: list[int], num_runs: int, device: torch.device) -> List[List[nn.Module]]:
    # A [num_widths, num_runs] List of Lists of models
    num_widths = len(width_range)
    width_max = max(width_range)
    model_lists = []

    for _ in range(num_widths):
        models = []
        for _ in range(num_runs):
            model = create_model(c).to(device)
            models.append(model)
        model_lists.append(models)
    return model_lists

def run_experiment_parallel(
    device: torch.device,
    validation_set: Tuple[Tensor, Tensor, Tensor],
    problem,
    c: Config,
    width_range: list[int],
    num_runs: int,
    clean_mode: bool,
) -> tuple[list, list, list, list]:
    """
    Run num_widths * num_runs experiments in parallel for models (Resnet or MLP).

    We are running experiments at different neural network widths (e.g. h, d_model, or down_rank_dim values).
    """
    x_val, y_val, center_indices_val = validation_set
    training_sets = _generate_training_sets(problem, c, num_runs, device, clean_mode)
    model_lists = _build_models(c, width_range, num_runs, device)
    
    num_widths = len(width_range)
    width_max = max(width_range)
    n = c.n
    d = c.d
    batch_size = c.batch_size

    # Create width_mask: first width elements are 1.0, rest are 0.0
    width_mask = torch.zeros(num_widths, num_runs, width_max, device=device)
    for w_index, w in enumerate(width_range):
        width_mask[w_index, :, :w] = 1.0
    width_mask = width_mask.reshape(num_widths * num_runs, width_max)
    
    # Flatten model_lists to a list of num_widths * num_runs models
    models_flattened = []
    for w_models in model_lists:
        models_flattened.extend(w_models)
    template = models_flattened[0]
    params, buffers = stack_module_state(models_flattened)
    # Wrap stacked params as nn.Parameter so the optimizer owns them
    params = {name: nn.Parameter(p) for name, p in params.items()}

    # Define the functional model list (num_h * num_runs models combined)
    def all_loss(params, buffers, x, y, width_mask):
        z = functional_call(template, (params, buffers), (x, width_mask))
        loss = F.binary_cross_entropy_with_logits(z, y)
        
        # Add logit regularization if c is specified
        if c.c is not None:
            logit_reg = c.c * torch.mean(z ** 2)
            loss = loss + logit_reg
        
        return loss
    loss_and_grad = grad_and_value(all_loss)
    vectorized_models = vmap(loss_and_grad, in_dims=(0, 0, 0, 0, 0))
    # Pure forward for evaluation (no grad computation)
    def all_forward(params, buffers, x, width_mask):
        return functional_call(template, (params, buffers), (x, width_mask))
    vectorized_forward = vmap(all_forward, in_dims=(0, 0, 0, 0))

    # Helper utilities to avoid repeated boilerplate
    def materialize_run_batches(x_full_list, y_full_list, idx_list):
        x_list = [x_full.index_select(0, idx) for x_full, idx in zip(x_full_list, idx_list)]
        y_list = [y_full.index_select(0, idx) for y_full, idx in zip(y_full_list, idx_list)]
        return x_list, y_list

    def stack_and_broadcast_runs(x_list, y_list, num_widths):
        # x_list: list[num_runs] of [m, d]; y_list: list[num_runs] of [m]
        x = torch.stack(x_list, dim=0)  # [num_runs, m, d]
        y = torch.stack(y_list, dim=0)  # [num_runs, m]
        m_local = x.shape[1]
        d_local = x.shape[2]
        x = x.unsqueeze(0).expand(num_widths, -1, -1, -1).reshape(num_widths * len(x_list), m_local, d_local)
        y = y.unsqueeze(0).expand(num_widths, -1, -1).reshape(num_widths * len(y_list), m_local)
        return x, y
        

    # Training
    x_full_list, y_full_list = zip(*training_sets)
    # Single optimizer for all parameters
    opt = torch.optim.AdamW(params.values(), lr=c.lr, weight_decay=c.weight_decay, eps=c.adam_eps)
    # Ensure template is in training mode for the training loop
    template.train()
    for epoch in range(c.epochs):
        # Generate per-run permutations
        perms = [torch.randperm(n, device=device) for _ in range(num_runs)]
        
        # Track epoch losses for reporting
        epoch_loss_sum_per_model = torch.zeros(num_widths * num_runs, device=device)
        epoch_sample_count = 0
        
        for b in range(0, n, batch_size):
            m = min(batch_size, n - b)
            # Get next batch indices for each run and prepare tensors
            idx_list = [perm[b:b+m] for perm in perms]
            x_list, y_list = materialize_run_batches(x_full_list, y_full_list, idx_list)
            x, y = stack_and_broadcast_runs(x_list, y_list, num_widths)
            grads, losses = vectorized_models(params, buffers, x, y, width_mask)

            # Accumulate losses for epoch reporting
            epoch_loss_sum_per_model += losses * m
            epoch_sample_count += m

            # Set gradients on the optimizer-owned Parameters
            for name, param in params.items():
                param.grad = grads[name]
            opt.step()
            opt.zero_grad(set_to_none=True)
        
        # Compute and report average training loss per width for this epoch
        epoch_loss_mean_per_model = epoch_loss_sum_per_model / epoch_sample_count
        epoch_loss_mean_per_w = epoch_loss_mean_per_model.view(num_widths, num_runs).mean(dim=1)
        
        print(f"Epoch {epoch + 1:3d}/{c.epochs}: ", end="")
        for w_idx, w in enumerate(width_range):
            print(f"width={w:2d}: {epoch_loss_mean_per_w[w_idx]:.4f}", end="  ")
        print()

    # Evaluation
    num_models = num_widths * num_runs
    n_val = x_val.shape[0]

    # Track per-model sums to aggregate per-h metrics
    train_loss_sum_per_model = torch.zeros(num_models, device=device)
    train_count_per_model = torch.zeros(num_models, device=device)

    val_loss_sum_per_model = torch.zeros(num_models, device=device)
    val_correct_sum_per_model = torch.zeros(num_models, device=device)
    val_count = 0

    # For variance across runs of validation logits per width
    val_logits_all = torch.empty(num_widths, num_runs, n_val, device=device)

    template.eval()
    with torch.no_grad():
        # Evaluate training loss per model on its corresponding training set (per run)
        for b in range(0, n, batch_size):
            m = min(batch_size, n - b)
            idx_list = [torch.arange(b, b + m, device=device) for _ in range(num_runs)]
            x_list, y_list = materialize_run_batches(x_full_list, y_full_list, idx_list)
            x_batch, y_batch = stack_and_broadcast_runs(x_list, y_list, num_widths)
            # shapes now [num_models, m, d] and [num_models, m]

            z = vectorized_forward(params, buffers, x_batch, width_mask)  # [num_models, m]
            batch_losses = F.binary_cross_entropy_with_logits(z, y_batch.float(), reduction='none').sum(dim=1)

            train_loss_sum_per_model += batch_losses
            train_count_per_model += m

        # Evaluate validation metrics for all models on the shared validation set
        write_start = 0
        for b in range(0, n_val, batch_size):
            m = min(batch_size, n_val - b)
            xb = x_val[b:b + m]
            yb = y_val[b:b + m]

            # Broadcast the same batch to all models
            xb_models = xb.unsqueeze(0).expand(num_models, -1, -1)  # [num_models, m, d]
            yb_models = yb.unsqueeze(0).expand(num_models, -1)      # [num_models, m]

            z = vectorized_forward(params, buffers, xb_models, width_mask)  # [num_models, m]

            # Accumulate per-model validation loss and accuracy
            val_loss_sum_per_model += F.binary_cross_entropy_with_logits(z, yb_models.float(), reduction='none').sum(dim=1)
            val_correct_sum_per_model += ((z > 0).float() == yb_models.float()).sum(dim=1)
            val_count += m

            # Store logits per h and run to compute variance across runs
            z_hr = z.reshape(num_widths, num_runs, m)  # [num_widths, num_runs, m]
            val_logits_all[:, :, write_start:write_start + m] = z_hr
            write_start += m

    # Aggregate metrics per h
    train_loss_mean_per_model = train_loss_sum_per_model / torch.clamp_min(train_count_per_model, 1)
    val_loss_mean_per_model = val_loss_sum_per_model / max(val_count, 1)
    val_acc_mean_per_model = val_correct_sum_per_model / max(val_count, 1)

    train_loss_mean_per_w = train_loss_mean_per_model.view(num_widths, num_runs).mean(dim=1)
    val_loss_mean_per_w = val_loss_mean_per_model.view(num_widths, num_runs).mean(dim=1)
    val_acc_mean_per_w = val_acc_mean_per_model.view(num_widths, num_runs).mean(dim=1)

    # Variance of validation logits across runs, grouped by center, then mean across centers and samples
    unique_centers = torch.unique(center_indices_val)
    num_centers = len(unique_centers)
    
    # Group by center, compute variance across runs for entire center, then mean across centers
    center_variances = []
    for center_idx in range(num_centers):
        center_mask = (center_indices_val == center_idx)
        center_logits = val_logits_all[:, :, center_mask]  # [num_widths, num_runs, num_samples_in_center]
        # Compute variance across runs for all samples in this center simultaneously
        center_var = center_logits.var(dim=1, unbiased=False)  # [num_widths, num_samples_in_center]
        # Mean across samples in this center
        center_variances.append(center_var.mean(dim=1))  # [num_widths]
    
    # Stack and mean across centers: [num_widths, num_centers] -> [num_widths]  
    mean_var_per_w = torch.stack(center_variances, dim=1).mean(dim=1)

    mean_vars = [v.item() for v in mean_var_per_w]
    mean_train_losses = [v.item() for v in train_loss_mean_per_w]
    mean_val_accuracies = [v.item() for v in val_acc_mean_per_w]
    mean_val_losses = [v.item() for v in val_loss_mean_per_w]

    for i, w in enumerate(width_range):
        print(f"width = {w:2d} | mean variance = {mean_vars[i]:.4e} | mean train loss = {mean_train_losses[i]:.4f} | mean val acc = {mean_val_accuracies[i]:.4f} | mean val loss = {mean_val_losses[i]:.4f}")

    return mean_vars, mean_train_losses, mean_val_accuracies, mean_val_losses