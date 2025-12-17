import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from jl.models import create_model
from jl.config import Config


def _setup_training(device, problem, validation_set, c: Config, clean_mode: bool):
    """
    Common setup for both standard and logit prior training.
    
    Returns:
        Tuple of (model, tracker, criterion, optimizer, train_loader, x_train, y_train, 
                  train_center_indices, x_val, y_val, center_indices)
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
    
    # Create model
    model = create_model(c).to(device)
    
    # Create weight tracker with tracking enabled
    tracker = model.get_tracker(track_weights=c.is_weight_tracker)
    
    # Set up loss function based on classification type
    if c.num_class == 2:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()
    
    # Select optimizer based on config
    if c.is_adam_w:
        optimizer = optim.AdamW(model.parameters(), lr=c.lr, weight_decay=c.weight_decay, eps=c.adam_eps)
    else:
        optimizer = optim.SGD(model.parameters(), lr=c.lr, momentum=0.9, weight_decay=c.weight_decay)
    
    # Get validation set
    x_val, y_val, center_indices = validation_set
    
    return (model, tracker, criterion, optimizer, train_loader, 
            x_train, y_train, train_center_indices, x_val, y_val, center_indices)


def _evaluate(model, x, y, criterion, num_class):
    """
    Evaluate model on given data.
    
    Returns:
        Tuple of (loss, accuracy)
    """
    with torch.no_grad():
        logits = model(x)
        if num_class == 2:
            loss = criterion(logits, y.float()).item()
            preds = (torch.sigmoid(logits) >= 0.5).float()
            acc = (preds == y.float()).float().mean().item()
        else:
            loss = criterion(logits, y.long()).item()
            preds = logits.argmax(dim=1)
            acc = (preds == y.long()).float().mean().item()
    return loss, acc


def _train_once_standard(model, tracker, criterion, optimizer, train_loader, 
                         x_val, y_val, c: Config):
    """
    Standard training loop.
    """
    print("Starting training...")
    
    for epoch in range(c.epochs):
        model.train()
        
        for batch_x, batch_y in train_loader:
            # Calculate validation accuracy and loss for tracker
            model.eval()
            val_loss, val_acc = _evaluate(model, x_val, y_val, criterion, c.num_class)
            model.train()
            
            # Training step
            optimizer.zero_grad()
            logits = model(batch_x)
            if c.num_class == 2:
                loss = criterion(logits, batch_y.float())
            else:
                loss = criterion(logits, batch_y.long())
            
            # Add logit regularization if c.c is specified
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
            
            # Update tracker with both accuracies and losses
            tracker.update(model, val_acc, train_acc, val_loss, train_loss)
            
            loss.backward()
            optimizer.step()


def _train_once_logit_prior(model, tracker, criterion, optimizer, train_loader, 
                            x_val, y_val, c: Config):
    """
    Logit prior training loop.
    
    Uses double batch approach: concatenates each batch with itself, feeds forward,
    then applies standard loss L to first half of logits and sum-of-squared-logits 
    loss Z to second half. The two losses are summed for backpropagation.
    """
    print("Starting logit prior training...")
    
    for epoch in range(c.epochs):
        model.train()
        
        for batch_x, batch_y in train_loader:
            # Calculate validation accuracy and loss for tracker (normal evaluation)
            model.eval()
            val_loss, val_acc = _evaluate(model, x_val, y_val, criterion, c.num_class)
            model.train()
            
            # Training step with double batch
            optimizer.zero_grad()
            
            # Double the input batch by concatenating with itself
            double_batch_x = torch.cat([batch_x, batch_x], dim=0)
            
            # Forward pass with double batch
            double_logits = model(double_batch_x)
            
            # Split logits into two halves
            batch_size = batch_x.shape[0]
            logits_L = double_logits[:batch_size]  # For standard loss
            logits_Z = double_logits[batch_size:]  # For squared logits loss
            
            # Compute standard loss L on first half
            if c.num_class == 2:
                loss_L = criterion(logits_L, batch_y.float())
            else:
                loss_L = criterion(logits_L, batch_y.long())
            
            # Compute loss Z: sum across logit dimension, mean across batch dimension
            if c.num_class == 2:
                # For binary, logits are shape (batch_size,)
                loss_Z = torch.abs(logits_Z).sum()
            else:
                # For multi-class, logits are shape (batch_size, num_classes)
                # Sum across logit dimension, mean across batch
                loss_Z = torch.abs(logits_Z).sum(dim=1).sum()
            
            # Combine losses
            loss = loss_L + loss_Z

            # Track only the standard loss L (not Z) for training loss reporting
            train_loss = loss_L.item()
            
            # Calculate training accuracy using first half logits
            with torch.no_grad():
                if c.num_class == 2:
                    train_preds = (torch.sigmoid(logits_L) >= 0.5).float()
                    train_acc = (train_preds == batch_y.float()).float().mean().item()
                else:
                    train_preds = logits_L.argmax(dim=1)
                    train_acc = (train_preds == batch_y.long()).float().mean().item()
            
            # Update tracker with both accuracies and losses
            tracker.update(model, val_acc, train_acc, val_loss, train_loss)
            
            loss.backward()
            optimizer.step()


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
    # Setup
    (model, tracker, criterion, optimizer, train_loader, 
     x_train, y_train, train_center_indices, x_val, y_val, center_indices) = \
        _setup_training(device, problem, validation_set, c, clean_mode)
    
    # Run appropriate training loop
    if c.is_logit_prior:
        _train_once_logit_prior(model, tracker, criterion, optimizer, train_loader, 
                                x_val, y_val, c)
    else:
        _train_once_standard(model, tracker, criterion, optimizer, train_loader, 
                             x_val, y_val, c)
    
    # Final evaluation
    model.eval()
    val_loss, val_acc = _evaluate(model, x_val, y_val, criterion, c.num_class)
    train_loss, train_acc = _evaluate(model, x_train, y_train, criterion, c.num_class)
    
    print(f"Final: Validation Loss = {val_loss:.6f}, Validation Accuracy = {val_acc:.4f}, "
          f"Training Loss = {train_loss:.6f}, Training Accuracy = {train_acc:.4f}")
    
    # Show plots
    print("Generating weight evolution plots...")
    tracker.plot()
    
    return model, tracker, x_train, y_train, train_center_indices
