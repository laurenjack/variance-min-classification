import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from jl.model_creator import create_model
from jl.config import Config
from jl.scheduler import create_lr_scheduler


def soft_cross_entropy(logits, soft_targets):
    """
    Compute cross-entropy loss with soft targets.
    
    Args:
        logits: Raw model outputs of shape [batch_size, num_classes]
        soft_targets: Soft probability targets of shape [batch_size, num_classes]
    
    Returns:
        Scalar loss value
    """
    log_probs = F.log_softmax(logits, dim=1)
    return -torch.sum(soft_targets * log_probs, dim=1).mean()


def get_segments_for_model(model_idx, num_models, segments_per_model):
    """
    Get the list of segment indices that a model is trained on.
    Model i is trained on segments [i, i+1, ..., i+segments_per_model-1] mod num_models.
    
    Args:
        model_idx: Model index
        num_models: Total number of models/segments
        segments_per_model: Number of segments each model is trained on
    
    Returns:
        List of segment indices
    """
    return [(model_idx + j) % num_models for j in range(segments_per_model)]


def get_models_not_trained_on_segment(segment_idx, num_models, segments_per_model):
    """
    Get the list of model indices that were NOT trained on a given segment.
    
    Args:
        segment_idx: Segment index
        num_models: Total number of models/segments
        segments_per_model: Number of segments each model is trained on
    
    Returns:
        List of model indices that did not see this segment
    """
    # Models that don't include segment j are: (j+1) mod num_models, ..., (j+num_models-segments_per_model) mod num_models
    return [(segment_idx + k) % num_models for k in range(segments_per_model, num_models)]


def split_data_into_segments(x, y, num_segments):
    """
    Split data into equal segments.
    
    Args:
        x: Input tensor of shape [n, d]
        y: Labels tensor of shape [n]
        num_segments: Number of segments to split into
    
    Returns:
        List of (x_segment, y_segment, segment_idx) tuples
    """
    n = x.shape[0]
    segment_size = n // num_segments
    segments = []
    for i in range(num_segments):
        start_idx = i * segment_size
        end_idx = (i + 1) * segment_size if i < num_segments - 1 else n
        segments.append((x[start_idx:end_idx], y[start_idx:end_idx], i))
    return segments


def create_model_dataset(segments, model_idx, num_models, segments_per_model):
    """
    Create a dataset for a specific model by combining its assigned segments.
    
    Args:
        segments: List of (x_segment, y_segment, segment_idx) tuples
        model_idx: Model index
        num_models: Total number of models/segments
        segments_per_model: Number of segments each model is trained on
    
    Returns:
        TensorDataset with (x, y, segment_indices)
    """
    assigned_segments = get_segments_for_model(model_idx, num_models, segments_per_model)
    x_parts = []
    y_parts = []
    seg_idx_parts = []
    
    for seg_idx in assigned_segments:
        x_seg, y_seg, _ = segments[seg_idx]
        x_parts.append(x_seg)
        y_parts.append(y_seg)
        # Store segment index for each data point
        seg_idx_parts.append(torch.full((x_seg.shape[0],), seg_idx, dtype=torch.long, device=x_seg.device))
    
    x_combined = torch.cat(x_parts, dim=0)
    y_combined = torch.cat(y_parts, dim=0)
    seg_indices = torch.cat(seg_idx_parts, dim=0)
    
    return TensorDataset(x_combined, y_combined, seg_indices)


def create_optimizer(model, c, initial_lr):
    """Create optimizer for a model based on config."""
    if c.optimizer == "adam_w":
        return optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=c.weight_decay,
                           eps=c.adam_eps, betas=c.adam_betas)
    elif c.optimizer == "sgd":
        return optim.SGD(model.parameters(), lr=initial_lr, momentum=c.sgd_momentum,
                         weight_decay=c.weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {c.optimizer}")


def compute_soft_target_for_batch(batch_x, batch_y, batch_seg_indices, models, c, num_models, segments_per_model):
    """
    Compute soft targets for a batch by averaging predictions from models
    that were not trained on each data point's segment.
    
    Args:
        batch_x: Batch inputs
        batch_y: Batch labels
        batch_seg_indices: Segment index for each sample in batch
        models: List of all models
        c: Config object
        num_models: Total number of models/segments
        segments_per_model: Number of segments each model is trained on
    
    Returns:
        Soft targets tensor of shape [batch_size, num_classes]
    """
    batch_size = batch_x.shape[0]
    soft_targets = torch.zeros(batch_size, c.num_class, device=batch_x.device)
    
    # One-hot encode the labels
    y_onehot = F.one_hot(batch_y.long(), num_classes=c.num_class).float()
    
    # For each unique segment in the batch, compute average predictions from non-trained models
    unique_segments = batch_seg_indices.unique()
    
    with torch.no_grad():
        for seg_idx in unique_segments:
            seg_mask = (batch_seg_indices == seg_idx)
            seg_x = batch_x[seg_mask]
            
            # Get models that were NOT trained on this segment
            non_trained_models = get_models_not_trained_on_segment(seg_idx.item(), num_models, segments_per_model)
            
            # Average predictions from these models
            avg_probs = torch.zeros(seg_x.shape[0], c.num_class, device=seg_x.device)
            for model_idx in non_trained_models:
                logits = models[model_idx](seg_x)
                probs = F.softmax(logits, dim=1)
                avg_probs += probs
            avg_probs /= len(non_trained_models)
            
            soft_targets[seg_mask] = avg_probs
    
    # Combine with one-hot labels using prob_weight
    soft_targets = (y_onehot + c.prob_weight * soft_targets) / (1.0 + c.prob_weight)
    
    return soft_targets


def compute_metrics(logits, y_true, criterion):
    """Compute loss and accuracy from logits."""
    loss = criterion(logits, y_true.long()).item()
    preds = logits.argmax(dim=1)
    acc = (preds == y_true.long()).float().mean().item()
    return loss, acc


def train_multi(device, problem, validation_set, c: Config):
    """
    Create and train multiple models with cross-model soft target averaging.
    
    Each model is trained on half the segments (with wraparound).
    For each training point, the soft target is the average of predictions
    from the models that were NOT trained on that point's segment.
    
    Parameters:
        device: torch.device to run computation on
        problem: Dataset generator with generate_dataset method
        validation_set: Tuple of (x_val, y_val, center_indices) tensors
        c: Config containing training and architecture parameters
            - c.num_models must be set (even number required)
        
    Returns:
        List of trained models
        
    Raises:
        ValueError: If c.num_class <= 2, c.is_hashed_dropout is True, or c.num_models is None/odd
    """
    # Validate constraints
    if c.num_class <= 2:
        raise ValueError(f"train_multi only supports num_class > 2, got {c.num_class}")
    if c.is_hashed_dropout:
        raise ValueError("train_multi does not support hashed dropout")
    if c.num_models is None:
        raise ValueError("train_multi requires c.num_models to be set")
    if c.num_models % 2 != 0:
        raise ValueError(f"train_multi requires c.num_models to be even, got {c.num_models}")
    
    num_models = c.num_models
    segments_per_model = num_models // 2
    
    # Generate one training set and split into segments
    x_train, y_train, _ = problem.generate_dataset(c.n, shuffle=True, clean_mode=False)
    segments = split_data_into_segments(x_train, y_train, num_models)
    
    # Create datasets and data loaders for each model
    datasets = [create_model_dataset(segments, i, num_models, segments_per_model) for i in range(num_models)]
    loaders = [DataLoader(ds, batch_size=c.batch_size, shuffle=True) for ds in datasets]
    
    # Calculate training steps for scheduler (use first loader as reference)
    steps_per_epoch = len(loaders[0])
    training_steps = steps_per_epoch * c.epochs
    
    # Adjust initial lr for scheduler if needed
    initial_lr = c.lr
    if c.lr_scheduler == 'wsd':
        warmup_steps = round(0.05 * training_steps)
        if warmup_steps > 0:
            initial_lr = c.lr / warmup_steps
    
    # Create models, optimizers, and schedulers
    models = [create_model(c).to(device) for _ in range(num_models)]
    optimizers = [create_optimizer(models[i], c, initial_lr) for i in range(num_models)]
    schedulers = [create_lr_scheduler(opt, training_steps, c.lr, c.lr_scheduler) for opt in optimizers]
    
    # Loss function for computing metrics
    criterion = nn.CrossEntropyLoss()
    
    # Get validation set
    x_val, y_val, _ = validation_set
    
    # Training loop
    for epoch in range(c.epochs):
        # Set all models to train mode
        for model in models:
            model.train()
        
        # Iterate through all loaders simultaneously
        # Use zip - all loaders have similar size (half the segments each)
        for batches in zip(*loaders):
            # Each batch is (batch_x, batch_y, batch_seg_indices)
            
            # Train each model on its batch
            for model_idx, (batch_x, batch_y, batch_seg_indices) in enumerate(batches):
                model = models[model_idx]
                optimizer = optimizers[model_idx]
                scheduler = schedulers[model_idx]
                
                # Compute soft target using models not trained on this data
                model.eval()  # Set current model to eval for consistent behavior
                soft_target = compute_soft_target_for_batch(
                    batch_x, batch_y, batch_seg_indices, models, c, num_models, segments_per_model
                )
                model.train()
                
                # Training step
                optimizer.zero_grad()
                logits = model(batch_x)
                loss = soft_cross_entropy(logits, soft_target)
                
                # Add logit regularization if specified
                if c.c is not None:
                    centered_logits = logits - logits.mean(dim=1, keepdim=True)
                    logit_reg = c.c * torch.mean(centered_logits ** 2)
                    loss = loss + logit_reg
                
                loss.backward()
                optimizer.step()
                
                # Step scheduler if enabled
                if scheduler is not None:
                    scheduler.step()
    
    # Final evaluation
    for model in models:
        model.eval()
    
    with torch.no_grad():
        # Compute metrics for each model
        val_metrics = []
        train_metrics = []
        
        for i, model in enumerate(models):
            # Validation metrics
            val_logits = model(x_val)
            val_loss, val_acc = compute_metrics(val_logits, y_val, criterion)
            val_metrics.append((val_loss, val_acc))
            
            # Training metrics (on this model's training data)
            ds = datasets[i]
            train_x = ds.tensors[0]
            train_y = ds.tensors[1]
            train_logits = model(train_x)
            train_loss, train_acc = compute_metrics(train_logits, train_y, criterion)
            train_metrics.append((train_loss, train_acc))
        
        # Ensemble metrics: average logits from all models
        all_val_logits = torch.stack([model(x_val) for model in models], dim=0)
        ensemble_val_logits = all_val_logits.mean(dim=0)
        ensemble_val_loss, ensemble_val_acc = compute_metrics(ensemble_val_logits, y_val, criterion)
    
    # Print metrics for all models
    for i in range(num_models):
        val_loss, val_acc = val_metrics[i]
        train_loss, train_acc = train_metrics[i]
        print(f"Model {i} Final: Validation Loss = {val_loss:.6f}, Validation Accuracy = {val_acc:.4f}, "
              f"Training Loss = {train_loss:.6f}, Training Accuracy = {train_acc:.4f}")
    
    print(f"Ensemble Final: Validation Loss = {ensemble_val_loss:.6f}, Validation Accuracy = {ensemble_val_acc:.4f}")
    
    return models
