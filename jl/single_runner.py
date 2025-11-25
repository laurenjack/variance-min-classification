import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from jl.models import create_model
from jl.config import Config


def train_once(device, problem, validation_set, c: Config, clean_mode: bool = False):
    """
    Create and train a model (Resnet or MLP) with weight tracking for testing purposes.
    
    Parameters:
        device: torch.device to run computation on
        problem: Dataset generator with generate_dataset method. If c.use_covariance is True,
                 problem must have a 'covariance' property.
        validation_set: Tuple of (x_val, y_val) tensors
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
    x_train, y_train = x_train.to(device), y_train.to(device)
    if train_center_indices is not None:
        train_center_indices = train_center_indices.to(device)
    
    # Create data loader for batch training
    train_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=c.batch_size, shuffle=True)
    
    # Create model
    model = create_model(c).to(device)
    
    # Create weight tracker with tracking enabled
    tracker = model.get_tracker(track_weights=c.is_weight_tracker)
    
    # Set up loss function and optimizer based on classification type
    if c.num_class == 2:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()
    
    # Select optimizer based on config
    if c.is_adam_w:
        optimizer = optim.AdamW(model.parameters(), lr=c.lr, weight_decay=c.weight_decay, eps=c.adam_eps)
    else:
        optimizer = optim.SGD(model.parameters(), lr=c.lr, weight_decay=c.weight_decay)
    
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
            
            # Update tracker with both accuracies and losses
            tracker.update(model, val_acc, train_acc, val_loss, train_loss)
            
            loss.backward()
            optimizer.step()
    
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

