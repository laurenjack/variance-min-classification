from typing import Tuple, List, Optional
import math

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.func import vmap, stack_module_state, functional_call, grad_and_value

from jl.config import Config
from jl.model_creator import create_model
from jl.scheduler import create_lr_scheduler


class _VectorizedModel:
    """
    A wrapper for vectorized model inference across multiple trained models.
    
    Stores the stacked parameters, buffers, and template model to enable
    efficient parallel inference using vmap.
    """
    
    def __init__(
        self,
        params: dict,
        buffers: dict,
        template: nn.Module,
        width_mask: Tensor,
        num_widths: int,
        num_runs: int,
        num_class: int,
    ):
        self.params = params
        self.buffers = buffers
        self.template = template
        self.width_mask = width_mask
        self.num_widths = num_widths
        self.num_runs = num_runs
        self.num_models = num_widths * num_runs
        self.num_class = num_class
        
        # Create vectorized forward function
        # randomness='different' ensures independent dropout masks per model if dropout is used
        def _forward(params, buffers, x, width_mask):
            return functional_call(template, (params, buffers), (x, width_mask))
        self._vectorized_forward = vmap(_forward, in_dims=(0, 0, 0, 0), randomness='different')
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through all models.
        
        Args:
            x: Input tensor of shape [num_models, n_samples, d]
            
        Returns:
            Logits tensor of shape [num_models, n_samples] for binary classification,
            or [num_models, n_samples, num_class] for multi-class.
        """
        self.template.eval()
        with torch.no_grad():
            return self._vectorized_forward(self.params, self.buffers, x, self.width_mask)
    
    def get_probabilities(self, x: Tensor) -> Tensor:
        """
        Get probabilities from all models.
        
        Args:
            x: Input tensor of shape [num_models, n_samples, d]
            
        Returns:
            For binary classification (num_class == 2):
                Probabilities tensor of shape [num_models, n_samples]
            For multi-class:
                Probabilities tensor of shape [num_models, n_samples, num_class]
        """
        logits = self.forward(x)
        if self.num_class == 2:
            return torch.sigmoid(logits)
        else:
            return torch.softmax(logits, dim=-1)
    
    def broadcast_input(self, x: Tensor) -> Tensor:
        """
        Broadcast a single validation set to all models.
        
        Args:
            x: Input tensor of shape [n_samples, d]
            
        Returns:
            Broadcasted tensor of shape [num_models, n_samples, d]
        """
        return x.unsqueeze(0).expand(self.num_models, -1, -1)


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
    model_lists = []

    for _ in range(num_widths):
        models = []
        for _ in range(num_runs):
            model = create_model(c).to(device)
            models.append(model)
        model_lists.append(models)
    return model_lists


def materialize_run_batches(
    x_full_list: Tuple[Tensor, ...],
    y_full_list: Tuple[Tensor, ...],
    idx_list: List[Tensor],
) -> Tuple[List[Tensor], List[Tensor]]:
    """Extract batch slices from full training sets for each run."""
    x_list = [x_full.index_select(0, idx) for x_full, idx in zip(x_full_list, idx_list)]
    y_list = [y_full.index_select(0, idx) for y_full, idx in zip(y_full_list, idx_list)]
    return x_list, y_list


def stack_and_broadcast_runs(
    x_list: List[Tensor],
    y_list: List[Tensor],
    num_widths: int,
) -> Tuple[Tensor, Tensor]:
    """Stack per-run batches and broadcast across widths.
    
    Args:
        x_list: list[num_runs] of [m, d] tensors
        y_list: list[num_runs] of [m] tensors
        num_widths: number of width configurations
        
    Returns:
        x: [num_widths * num_runs, m, d]
        y: [num_widths * num_runs, m]
    """
    x = torch.stack(x_list, dim=0)  # [num_runs, m, d]
    y = torch.stack(y_list, dim=0)  # [num_runs, m]
    m_local = x.shape[1]
    d_local = x.shape[2]
    num_runs = len(x_list)
    x = x.unsqueeze(0).expand(num_widths, -1, -1, -1).reshape(num_widths * num_runs, m_local, d_local)
    y = y.unsqueeze(0).expand(num_widths, -1, -1).reshape(num_widths * num_runs, m_local)
    return x, y

def _train_parallel(
    device: torch.device,
    problem,
    c: Config,
    num_runs: int,
    clean_mode: bool,
    width_range: list[int],
) -> Tuple[_VectorizedModel, List[Tuple[Tensor, Tensor]]]:
    """Internal: Train num_widths * num_runs models in parallel."""
    if c.optimizer == "reg_adam_w":
        raise ValueError("reg_adam_w optimizer is not supported in multi_runner. Please use single_runner instead.")

    training_sets = _generate_training_sets(problem, c, num_runs, device, clean_mode)
    model_lists = _build_models(c, width_range, num_runs, device)
    
    num_widths = len(width_range)
    width_max = max(width_range)
    n = c.n
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
        if c.num_class == 2:
            loss = F.binary_cross_entropy_with_logits(z, y)
        else:
            loss = F.cross_entropy(z, y.long())
        
        # Add logit regularization if c is specified
        if c.c is not None:
            logit_reg = c.c * torch.mean(z ** 2)
            loss = loss + logit_reg
        
        return loss
    loss_and_grad = grad_and_value(all_loss)
    # randomness='different' ensures independent dropout masks per model if dropout is used
    vectorized_models = vmap(loss_and_grad, in_dims=(0, 0, 0, 0, 0), randomness='different')

    # Training
    x_full_list, y_full_list = zip(*training_sets)

    # Calculate training steps for WSD scheduler
    steps_per_epoch = math.ceil(n / batch_size)
    training_steps = steps_per_epoch * c.epochs

    # Adjust initial lr for scheduler if needed
    initial_lr = c.lr
    if c.lr_scheduler == 'wsd':
        warmup_steps = round(0.05 * training_steps)
        if warmup_steps > 0:
            initial_lr = c.lr / warmup_steps

    # Single optimizer for all parameters
    if c.optimizer == "adam_w":
        opt = torch.optim.AdamW(params.values(), lr=initial_lr, weight_decay=c.weight_decay, eps=c.adam_eps)
    elif c.optimizer == "sgd":
        opt = torch.optim.SGD(params.values(), lr=initial_lr, weight_decay=c.weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {c.optimizer}")

    # Create learning rate scheduler if enabled
    scheduler = create_lr_scheduler(opt, training_steps, c.lr, c.lr_scheduler)
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
            # Step WSD scheduler if enabled
            if scheduler is not None:
                scheduler.step()
            opt.zero_grad(set_to_none=True)
        
        # Compute and report average training loss per width for this epoch
        epoch_loss_mean_per_model = epoch_loss_sum_per_model / epoch_sample_count
        epoch_loss_mean_per_w = epoch_loss_mean_per_model.view(num_widths, num_runs).mean(dim=1)
        
        print(f"Epoch {epoch + 1:3d}/{c.epochs}: ", end="")
        if c.width_varyer is None:
            # No width variation - just print overall loss
            print(f"loss={epoch_loss_mean_per_w[0]:.4f}", end="")
        else:
            for w_idx, w in enumerate(width_range):
                print(f"width={w:2d}: {epoch_loss_mean_per_w[w_idx]:.4f}", end="  ")
        print()

    # Create _VectorizedModel for inference
    vectorized_model = _VectorizedModel(
        params=params,
        buffers=buffers,
        template=template,
        width_mask=width_mask,
        num_widths=num_widths,
        num_runs=num_runs,
        num_class=c.num_class,
    )

    return vectorized_model, training_sets


def train_and_compute_metrics(
    device: torch.device,
    problem,
    c: Config,
    num_runs: int,
    clean_mode: bool,
    x_val: Tensor,
    y_val: Tensor,
    width_range: Optional[list[int]] = None,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Train models in parallel and compute metrics.

    Args:
        device: torch device
        problem: Dataset generator with generate_dataset method
        c: Config object
        num_runs: Number of runs per width
        clean_mode: Whether to generate clean data
        x_val: Validation inputs
        y_val: Validation labels
        width_range: Optional list of widths. Required if c.width_varyer is set.
                    If None and c.width_varyer is None, uses [1] internally.

    Returns:
        Tuple of (train_loss_per_w, val_acc_per_w, val_loss_per_w, val_logits_all)
    """
    # Validate width_varyer and width_range consistency
    if c.width_varyer is None and width_range is not None:
        raise ValueError("width_range must be None when c.width_varyer is None")
    if c.width_varyer is not None and width_range is None:
        raise ValueError("width_range is required when c.width_varyer is set")

    # Use [1] internally when no width variation
    if width_range is None:
        width_range = [1]

    model, training_sets = _train_parallel(device, problem, c, num_runs, clean_mode, width_range)
    return _compute_metrics(model, training_sets, x_val, y_val, c)


def _compute_metrics(
    model: _VectorizedModel,
    training_sets: List[Tuple[Tensor, Tensor]],
    x_val: Tensor,
    y_val: Tensor,
    c: Config,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Internal: Compute train/val metrics and return validation logits."""
    num_models = model.num_models
    num_widths = model.num_widths
    num_runs = model.num_runs
    batch_size = c.batch_size
    n = c.n
    n_val = x_val.shape[0]
    device = x_val.device
    
    x_full_list, y_full_list = zip(*training_sets)

    # Track per-model sums
    train_loss_sum_per_model = torch.zeros(num_models, device=device)
    train_count_per_model = torch.zeros(num_models, device=device)
    val_loss_sum_per_model = torch.zeros(num_models, device=device)
    val_correct_sum_per_model = torch.zeros(num_models, device=device)
    val_count = 0
    
    # Store logits for variance calculation
    if c.num_class == 2:
        val_logits_all = torch.empty(num_widths, num_runs, n_val, device=device)
    else:
        val_logits_all = torch.empty(num_widths, num_runs, n_val, c.num_class, device=device)

    model.template.eval()
    with torch.no_grad():
        # Evaluate training loss
        for b in range(0, n, batch_size):
            m = min(batch_size, n - b)
            idx_list = [torch.arange(b, b + m, device=device) for _ in range(num_runs)]
            x_list, y_list = materialize_run_batches(x_full_list, y_full_list, idx_list)
            x_batch, y_batch = stack_and_broadcast_runs(x_list, y_list, num_widths)

            z = model.forward(x_batch)
            if c.num_class == 2:
                batch_losses = F.binary_cross_entropy_with_logits(z, y_batch.float(), reduction='none').sum(dim=1)
            else:
                z_flat = z.reshape(-1, c.num_class)
                y_flat = y_batch.reshape(-1).long()
                losses_flat = F.cross_entropy(z_flat, y_flat, reduction='none')
                batch_losses = losses_flat.reshape(num_models, m).sum(dim=1)

            train_loss_sum_per_model += batch_losses
            train_count_per_model += m

        # Evaluate validation metrics
        write_start = 0
        for b in range(0, n_val, batch_size):
            m = min(batch_size, n_val - b)
            xb = x_val[b:b + m]
            yb = y_val[b:b + m]

            xb_models = xb.unsqueeze(0).expand(num_models, -1, -1)
            yb_models = yb.unsqueeze(0).expand(num_models, -1)

            z = model.forward(xb_models)

            if c.num_class == 2:
                val_loss_sum_per_model += F.binary_cross_entropy_with_logits(z, yb_models.float(), reduction='none').sum(dim=1)
                val_correct_sum_per_model += ((z > 0).float() == yb_models.float()).sum(dim=1)
                z_hr = z.reshape(num_widths, num_runs, m)
                val_logits_all[:, :, write_start:write_start + m] = z_hr
            else:
                z_flat = z.reshape(-1, c.num_class)
                y_flat = yb_models.reshape(-1).long()
                losses_flat = F.cross_entropy(z_flat, y_flat, reduction='none')
                val_loss_sum_per_model += losses_flat.reshape(num_models, m).sum(dim=1)
                preds = z.argmax(dim=-1)
                val_correct_sum_per_model += (preds == yb_models.long()).sum(dim=1)
                z_hr = z.reshape(num_widths, num_runs, m, c.num_class)
                val_logits_all[:, :, write_start:write_start + m, :] = z_hr
            
            val_count += m
            write_start += m

    # Aggregate metrics per width
    train_loss_mean_per_model = train_loss_sum_per_model / torch.clamp_min(train_count_per_model, 1)
    val_loss_mean_per_model = val_loss_sum_per_model / max(val_count, 1)
    val_acc_mean_per_model = val_correct_sum_per_model / max(val_count, 1)

    train_loss_per_w = train_loss_mean_per_model.view(num_widths, num_runs).mean(dim=1)
    val_loss_per_w = val_loss_mean_per_model.view(num_widths, num_runs).mean(dim=1)
    val_acc_per_w = val_acc_mean_per_model.view(num_widths, num_runs).mean(dim=1)

    return train_loss_per_w, val_acc_per_w, val_loss_per_w, val_logits_all


def _print_basic_metrics(
    model: _VectorizedModel,
    training_sets: List[Tuple[Tensor, Tensor]],
    x_val: Tensor,
    y_val: Tensor,
    c: Config,
    width_range: list[int],
) -> None:
    """Internal: Compute and print basic train/val metrics (no variance tracking)."""
    train_loss_per_w, val_acc_per_w, val_loss_per_w, _ = _compute_metrics(
        model, training_sets, x_val, y_val, c
    )

    if c.width_varyer is None:
        print(f"mean train loss = {train_loss_per_w[0]:.4f} | mean val acc = {val_acc_per_w[0]:.4f} | mean val loss = {val_loss_per_w[0]:.4f}")
    else:
        for i, w in enumerate(width_range):
            print(f"width = {w:2d} | mean train loss = {train_loss_per_w[i]:.4f} | mean val acc = {val_acc_per_w[i]:.4f} | mean val loss = {val_loss_per_w[i]:.4f}")