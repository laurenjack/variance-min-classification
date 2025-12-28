import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


from jl.models import create_model
from jl.config import Config
from jl.feature_experiments.optimizer import RegAdamW, register_reg_adam_w_hooks
from jl.scheduler import create_lr_scheduler


def train_once(device, problem, validation_set, c: Config, clean_mode: bool = False):
    """
    Create and train a model with weight tracking for testing purposes.
    
    Parameters:
        device: torch.device to run computation on
        problem: Dataset generator with generate_dataset method. If c.use_covariance is True,
                 problem must have a 'covariance' property.
        validation_set: Tuple of (x_val, y_val, center_indices) tensors
        c: Config containing training and architecture parameters
        clean_mode: Whether to generate clean data (no noise)
    """
    if c.model_type == 'mlp':
        model_desc = f"{c.model_type.upper()}: d={c.d}, h={c.h}"
    elif c.model_type == 'multi-linear':
        model_desc = f"{c.model_type.upper()}: d={c.d}, h={c.h}"
    else:
        model_desc = f"{c.model_type.upper()}: d={c.d}, d_model={c.d_model}"
    print(f"Starting training with {model_desc}")
    
    # Generate training data
    x_train, y_train, train_center_indices = problem.generate_dataset(c.n, shuffle=True, clean_mode=clean_mode)
    
    # Create data loader for batch training
    train_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=c.batch_size, shuffle=True)

    # Calculate training steps for scheduler
    steps_per_epoch = len(train_loader)
    training_steps = steps_per_epoch * c.epochs

    # Adjust initial lr for scheduler if needed
    initial_lr = c.lr
    if c.lr_scheduler == 'wsd':
        warmup_steps = round(0.05 * training_steps)
        if warmup_steps > 0:
            initial_lr = c.lr / warmup_steps

    # Create model
    model = create_model(c).to(device)
    
    # Create weight tracker
    tracker = model.get_tracker(c)
    
    # Set up loss function based on classification type
    if c.num_class == 2:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()
    
    # Select optimizer based on config
    if c.optimizer == "adam_w":
        optimizer = optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=c.weight_decay, eps=c.adam_eps, betas=c.adam_betas)
    elif c.optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=initial_lr, momentum=c.sgd_momentum, weight_decay=c.weight_decay)
    elif c.optimizer == "reg_adam_w":
        # Register hooks on Linear modules for RegAdamW
        register_reg_adam_w_hooks(model)
        optimizer = RegAdamW(model.parameters(), lr=initial_lr, weight_decay=c.weight_decay, eps=c.adam_eps, betas=c.adam_betas)
    else:
        raise ValueError(f"Unknown optimizer: {c.optimizer}")

    # Create learning rate scheduler if enabled
    scheduler = create_lr_scheduler(optimizer, training_steps, c.lr, c.lr_scheduler)
    
    # Get validation set
    x_val, y_val, center_indices = validation_set
    
    print("Starting training...")
    
    # Training loop
    for epoch in range(c.epochs):
        # Training phase
        model.train()
        
        for batch_x, batch_y in train_loader:
            # Calculate validation accuracy and loss for tracker
            model.eval()
            with torch.no_grad():
                val_logits = model(x_val)
                if c.num_class == 2:
                    val_loss = criterion(val_logits, y_val.float()).item()
                    val_preds = (torch.sigmoid(val_logits) >= 0.5).float()
                    val_acc = (val_preds == y_val.float()).float().mean().item()
                else:
                    # For multi-class, logits should be (batch_size, num_classes)
                    val_loss = criterion(val_logits, y_val.long()).item()
                    val_preds = val_logits.argmax(dim=1)
                    val_acc = (val_preds == y_val.long()).float().mean().item()
            
            model.train()
            
            # Training step
            optimizer.zero_grad()
            logits = model(batch_x)
            if c.num_class == 2:
                loss = criterion(logits, batch_y.float())
            else:
                # For multi-class, logits should be (batch_size, num_classes)
                loss = criterion(logits, batch_y.long())
            
            # Add logit regularization if c is specified
            if c.c is not None:
                logit_reg = c.c * torch.mean(logits ** 2)
                loss = loss + logit_reg

            train_loss = loss.item()
            
            # Calculate training accuracy
            with torch.no_grad():
                if c.num_class == 2:
                    train_preds = (torch.sigmoid(logits) >= 0.5).float()
                    train_acc = (train_preds == batch_y.float()).float().mean().item()
                else:
                    train_preds = logits.argmax(dim=1)
                    train_acc = (train_preds == batch_y.long()).float().mean().item()
            
            loss.backward()
            
            # For 'full_step' mode, compute the optimizer step direction (excluding lr and weight_decay)
            if c.weight_tracker == 'full_step':
                # Store old weights before optimizer step
                old_weights = {id(p): p.data.clone() for p in model.parameters()}
                
                optimizer.step()
                
                # Compute and store the full step on each parameter
                # Δw = -lr * (adam_update + wd * w_old)
                # full_step = Δw/lr + wd*w_old = -adam_update (just the gradient-based update)
                for p in model.parameters():
                    delta_w = p.data - old_weights[id(p)]
                    p._step = delta_w / c.lr # + c.weight_decay * old_weights[id(p)]
                
                # Update tracker after step computation
                tracker.update(model, val_acc, train_acc, val_loss, train_loss)
            else:
                # For 'weight' mode or None, update tracker before optimizer.step()
                tracker.update(model, val_acc, train_acc, val_loss, train_loss)
                optimizer.step()

            # Step WSD scheduler if enabled
            if scheduler is not None:
                scheduler.step()
    
    # Final evaluation
    model.eval()
    with torch.no_grad():
        # Calculate final validation accuracy
        val_logits = model(x_val)
        if c.num_class == 2:
            val_loss = criterion(val_logits, y_val.float()).item()
            val_preds = (torch.sigmoid(val_logits) >= 0.5).float()
            val_acc = (val_preds == y_val.float()).float().mean().item()
        else:
            val_loss = criterion(val_logits, y_val.long()).item()
            val_preds = val_logits.argmax(dim=1)
            val_acc = (val_preds == y_val.long()).float().mean().item()
        
        # Calculate final training accuracy and loss
        train_logits = model(x_train)
        if c.num_class == 2:
            train_loss = criterion(train_logits, y_train.float()).item()
            train_preds = (torch.sigmoid(train_logits) >= 0.5).float()
            train_acc = (train_preds == y_train.float()).float().mean().item()
        else:
            train_loss = criterion(train_logits, y_train.long()).item()
            train_preds = train_logits.argmax(dim=1)
            train_acc = (train_preds == y_train.long()).float().mean().item()
    
    print(f"Final: Validation Loss = {val_loss:.6f}, Validation Accuracy = {val_acc:.4f}, Training Loss = {train_loss:.6f}, Training Accuracy = {train_acc:.4f}")
    
    # Show plots
    print("Generating weight evolution plots...")
    tracker.plot()
    
    return model, tracker, x_train, y_train, train_center_indices
