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
from src.learned_dropout.config import EmpiricalConfig


def _generate_training_sets(problem, c: EmpiricalConfig, device: torch.device, percent_correct: float) -> List[Tuple[Tensor, Tensor]]:
    # Generate all training sets once, outside the loops
    training_sets = []
    for _ in range(c.num_runs):
        x_train, y_train, _ = problem.generate_dataset(c.n, shuffle=True, percent_correct=percent_correct)
        x_train, y_train = x_train.to(device), y_train.to(device)
        training_sets.append((x_train, y_train))
    return training_sets

def _build_models(resnet_builder, c: EmpiricalConfig, device: torch.device) -> List[List[nn.Module]]:
    # A [num_h, num_runs] List of Lists of models
    num_h = len(c.h_range)
    h_max = max(c.h_range)
    model_lists = []

    for _ in range(num_h):
        models = []
        for _ in range(c.num_runs):
            model = resnet_builder.build(c, h_max, device)
            models.append(model)
        model_lists.append(models)
    return model_lists

def run_experiment_parallel(
    resnet_builder,
    device: torch.device,
    validation_set: Tuple[Tensor, Tensor, Tensor],
    problem,
    c: EmpiricalConfig,
    percent_correct: float
) -> tuple[list, list, list, list]:
    x_val, y_val, center_indices_val = validation_set
    training_sets = _generate_training_sets(problem, c, device, percent_correct)
    model_lists = _build_models(resnet_builder, c, device)
    
    num_h = len(c.h_range)
    num_runs = c.num_runs
    h_max = max(c.h_range)
    n = c.n
    d = c.d
    batch_size = c.batch_size

    # Create h_mask: first h elements are 1.0, rest are 0.0
    h_mask = torch.zeros(num_h, num_runs, h_max, device=device)
    for h_index, h in enumerate(c.h_range):
        h_mask[h_index, :, :h] = 1.0
    h_mask = h_mask.reshape(num_h * num_runs, h_max)
    
    # Flatten model_lists to a list of num_h * num_runs models
    models_flattened = []
    for h_models in model_lists:
        models_flattened.extend(h_models)
    template = models_flattened[0]
    params, buffers = stack_module_state(models_flattened)
    # Wrap stacked params as nn.Parameter so the optimizer owns them
    params = {name: nn.Parameter(p) for name, p in params.items()}

    # Define the functional model list (num_h * num_runs models combined)
    def all_loss(params, buffers, x, y, h_mask):
        z = functional_call(template, (params, buffers), (x, h_mask))
        return F.binary_cross_entropy_with_logits(z, y)
    loss_and_grad = grad_and_value(all_loss)
    vectorized_models = vmap(loss_and_grad, in_dims=(0, 0, 0, 0, 0))
    # Pure forward for evaluation (no grad computation)
    def all_forward(params, buffers, x, h_mask):
        return functional_call(template, (params, buffers), (x, h_mask))
    vectorized_forward = vmap(all_forward, in_dims=(0, 0, 0, 0))

    # Helper utilities to avoid repeated boilerplate
    def materialize_run_batches(x_full_list, y_full_list, idx_list):
        x_list = [x_full.index_select(0, idx) for x_full, idx in zip(x_full_list, idx_list)]
        y_list = [y_full.index_select(0, idx) for y_full, idx in zip(y_full_list, idx_list)]
        return x_list, y_list

    def stack_and_broadcast_runs(x_list, y_list, num_h):
        # x_list: list[num_runs] of [m, d]; y_list: list[num_runs] of [m]
        x = torch.stack(x_list, dim=0)  # [num_runs, m, d]
        y = torch.stack(y_list, dim=0)  # [num_runs, m]
        m_local = x.shape[1]
        d_local = x.shape[2]
        x = x.unsqueeze(0).expand(num_h, -1, -1, -1).reshape(num_h * len(x_list), m_local, d_local)
        y = y.unsqueeze(0).expand(num_h, -1, -1).reshape(num_h * len(y_list), m_local)
        return x, y
        

    # Training
    x_full_list, y_full_list = zip(*training_sets)
    # Single optimizer for all parameters
    opt = torch.optim.AdamW(params.values(), lr=c.lr, weight_decay=c.weight_decay)
    # Ensure template is in training mode for the training loop
    template.train()
    for epoch in range(c.epochs):
        # Generate per-run permutations
        perms = [torch.randperm(n, device=device) for _ in range(num_runs)]
        
        # Track epoch losses for reporting
        epoch_loss_sum_per_model = torch.zeros(num_h * num_runs, device=device)
        epoch_sample_count = 0
        
        for b in range(0, n, batch_size):
            m = min(batch_size, n - b)
            # Get next batch indices for each run and prepare tensors
            idx_list = [perm[b:b+m] for perm in perms]
            x_list, y_list = materialize_run_batches(x_full_list, y_full_list, idx_list)
            x, y = stack_and_broadcast_runs(x_list, y_list, num_h)
            grads, losses = vectorized_models(params, buffers, x, y, h_mask)

            # Accumulate losses for epoch reporting
            epoch_loss_sum_per_model += losses * m
            epoch_sample_count += m

            # Set gradients on the optimizer-owned Parameters
            for name, param in params.items():
                param.grad = grads[name]
            opt.step()
            opt.zero_grad(set_to_none=True)
        
        # Compute and report average training loss per h for this epoch
        epoch_loss_mean_per_model = epoch_loss_sum_per_model / epoch_sample_count
        epoch_loss_mean_per_h = epoch_loss_mean_per_model.view(num_h, num_runs).mean(dim=1)
        
        print(f"Epoch {epoch + 1:3d}/{c.epochs}: ", end="")
        for h_idx, h in enumerate(c.h_range):
            print(f"h={h:2d}: {epoch_loss_mean_per_h[h_idx]:.4f}", end="  ")
        print()

    # Evaluation
    num_models = num_h * num_runs
    n_val = x_val.shape[0]

    # Track per-model sums to aggregate per-h metrics
    train_loss_sum_per_model = torch.zeros(num_models, device=device)
    train_count_per_model = torch.zeros(num_models, device=device)

    val_loss_sum_per_model = torch.zeros(num_models, device=device)
    val_correct_sum_per_model = torch.zeros(num_models, device=device)
    val_count = 0

    # For variance across runs of validation logits per h
    val_logits_all = torch.empty(num_h, num_runs, n_val, device=device)

    template.eval()
    with torch.no_grad():
        # Evaluate training loss per model on its corresponding training set (per run)
        for b in range(0, n, batch_size):
            m = min(batch_size, n - b)
            idx_list = [torch.arange(b, b + m, device=device) for _ in range(num_runs)]
            x_list, y_list = materialize_run_batches(x_full_list, y_full_list, idx_list)
            x_batch, y_batch = stack_and_broadcast_runs(x_list, y_list, num_h)
            # shapes now [num_models, m, d] and [num_models, m]

            z = vectorized_forward(params, buffers, x_batch, h_mask)  # [num_models, m]
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

            z = vectorized_forward(params, buffers, xb_models, h_mask)  # [num_models, m]

            # Accumulate per-model validation loss and accuracy
            val_loss_sum_per_model += F.binary_cross_entropy_with_logits(z, yb_models.float(), reduction='none').sum(dim=1)
            val_correct_sum_per_model += ((z > 0).float() == yb_models.float()).sum(dim=1)
            val_count += m

            # Store logits per h and run to compute variance across runs
            z_hr = z.reshape(num_h, num_runs, m)  # [num_h, num_runs, m]
            val_logits_all[:, :, write_start:write_start + m] = z_hr
            write_start += m

    # Aggregate metrics per h
    train_loss_mean_per_model = train_loss_sum_per_model / torch.clamp_min(train_count_per_model, 1)
    val_loss_mean_per_model = val_loss_sum_per_model / max(val_count, 1)
    val_acc_mean_per_model = val_correct_sum_per_model / max(val_count, 1)

    train_loss_mean_per_h = train_loss_mean_per_model.view(num_h, num_runs).mean(dim=1)
    val_loss_mean_per_h = val_loss_mean_per_model.view(num_h, num_runs).mean(dim=1)
    val_acc_mean_per_h = val_acc_mean_per_model.view(num_h, num_runs).mean(dim=1)

    # Variance of validation logits across runs, grouped by center, then mean across centers and samples
    unique_centers = torch.unique(center_indices_val)
    num_centers = len(unique_centers)
    
    # Group by center, compute variance across runs for entire center, then mean across centers
    center_variances = []
    for center_idx in range(num_centers):
        center_mask = (center_indices_val == center_idx)
        center_logits = val_logits_all[:, :, center_mask]  # [num_h, num_runs, num_samples_in_center]
        # Compute variance across runs for all samples in this center simultaneously
        center_var = center_logits.var(dim=1, unbiased=False)  # [num_h, num_samples_in_center]
        # Mean across samples in this center
        center_variances.append(center_var.mean(dim=1))  # [num_h]
    
    # Stack and mean across centers: [num_h, num_centers] -> [num_h]  
    mean_var_per_h = torch.stack(center_variances, dim=1).mean(dim=1)

    mean_vars = [v.item() for v in mean_var_per_h]
    mean_train_losses = [v.item() for v in train_loss_mean_per_h]
    mean_val_accuracies = [v.item() for v in val_acc_mean_per_h]
    mean_val_losses = [v.item() for v in val_loss_mean_per_h]

    for i, h in enumerate(c.h_range):
        print(f"h = {h:2d} | mean variance = {mean_vars[i]:.4e} | mean train loss = {mean_train_losses[i]:.4f} | mean val acc = {mean_val_accuracies[i]:.4f} | mean val loss = {mean_val_losses[i]:.4f}")

    return mean_vars, mean_train_losses, mean_val_accuracies, mean_val_losses


def run_experiment(
    resnet_builder,
    device: torch.device,
    validation_set: Tuple[Tensor, Tensor, Tensor],
    problem,
    c: EmpiricalConfig,
    percent_correct: float
) -> tuple[list, list, list, list]:
    """Returns mean variances of validation logits, mean training losses, mean validation accuracies, and mean validation losses for each hidden size h."""
    x_val, y_val, center_indices_val = validation_set
    training_sets = _generate_training_sets(problem, c, device, percent_correct)
    
    mean_vars = []
    mean_train_losses = []
    mean_val_accuracies = []
    mean_val_losses = []
    for h in c.h_range:
        run_preds = []
        run_train_losses = []
        run_val_accuracies = []
        run_val_losses = []
        for run_idx in range(c.num_runs):
            # Use the pre-generated training data for this run
            x_train, y_train = training_sets[run_idx]

            # create data loader for batch training
            train_dataset = TensorDataset(x_train, y_train)
            train_loader = DataLoader(train_dataset, batch_size=c.batch_size, shuffle=True)

            # build model with current hidden parameter
            model = resnet_builder.build(c, h, device)

            # loss and optimizer
            criterion = nn.BCEWithLogitsLoss()
            optimizer = optim.AdamW(model.parameters(), lr=c.lr, weight_decay=c.weight_decay)

            # training loop
            model.train()
            for epoch in range(c.epochs):
                for batch_x, batch_y in train_loader:
                    optimizer.zero_grad()
                    logits = model(batch_x).squeeze()
                    loss = criterion(logits, batch_y.float())
                    
                    # # Add L1 regularization if specified
                    # l1_reg_loss = model.get_l1_regularization_loss()
                    # total_loss = loss + l1_reg_loss
                    
                    loss.backward()
                    optimizer.step()

            # compute total training loss on whole dataset after training
            model.eval()
            with torch.no_grad():
                total_logits = model(x_train).squeeze()
                total_train_loss = criterion(total_logits, y_train.float()).item()
            
            # store total training loss for this run
            run_train_losses.append(total_train_loss)

            # validation predictions and metrics
            model.eval()
            with torch.no_grad():
                z_val = model(x_val).squeeze()  # [n_val]
                val_probs = torch.sigmoid(z_val)
                val_predictions = (val_probs > 0.5).float()
                val_accuracy = (val_predictions == y_val).float().mean().item()
                # Compute validation loss
                criterion = nn.BCEWithLogitsLoss()
                val_loss = criterion(z_val, y_val.float()).item()
            run_preds.append(z_val)
            run_val_accuracies.append(val_accuracy)
            run_val_losses.append(val_loss)

        # compute variance across runs for each center, then mean across centers and samples
        preds_stack = torch.stack(run_preds, dim=0)  # [num_runs, n_val]
        unique_centers = torch.unique(center_indices_val)
        
        center_vars = []
        for center_idx in unique_centers:
            center_mask = (center_indices_val == center_idx)
            center_preds = preds_stack[:, center_mask]  # [num_runs, num_samples_in_center]
            center_var = center_preds.var(dim=0, unbiased=False)  # [num_samples_in_center]
            center_vars.append(center_var.mean())  # Mean variance across samples in this center
        mean_var = torch.stack(center_vars).mean().item()  # Mean across centers
        mean_vars.append(mean_var)
        
        # compute mean training loss across runs
        mean_train_loss = sum(run_train_losses) / len(run_train_losses)
        mean_train_losses.append(mean_train_loss)
        
        # compute mean validation accuracy across runs
        mean_val_accuracy = sum(run_val_accuracies) / len(run_val_accuracies)
        mean_val_accuracies.append(mean_val_accuracy)
        
        # compute mean validation loss across runs
        mean_val_loss = sum(run_val_losses) / len(run_val_losses)
        mean_val_losses.append(mean_val_loss)
        
        print(f"h = {h:2d} | mean variance = {mean_var:.4e} | mean train loss = {mean_train_loss:.4f} | mean val acc = {mean_val_accuracy:.4f} | mean val loss = {mean_val_loss:.4f}")

    return mean_vars, mean_train_losses, mean_val_accuracies, mean_val_losses