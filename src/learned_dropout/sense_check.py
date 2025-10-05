import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from src.learned_dropout.models import Resnet
from src.learned_dropout.config import Config


def train_once(device, problem, validation_set, c: Config, use_percent_correct: bool = True):
    """
    Create and train a Resnet model with weight tracking for testing purposes.
    
    Parameters:
        device: torch.device to run computation on
        problem: Dataset generator with generate_dataset method
        validation_set: Tuple of (x_val, y_val) tensors
        c: Config containing training and architecture parameters
    """
    print(f"Starting training with Resnet: d={c.d}, d_model={c.d_model}, h={c.h}, num_layers={c.num_layers}")
    
    # Generate training data
    x_train, y_train, _ = problem.generate_dataset(c.n, shuffle=True, use_percent_correct=use_percent_correct)
    x_train, y_train = x_train.to(device), y_train.to(device)
    
    # Create data loader for batch training
    train_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=c.batch_size, shuffle=True)
    
    # Create Resnet model
    model = Resnet(c).to(device)
    
    # Create weight tracker with tracking enabled
    tracker = model.get_tracker(track_weights=c.is_weight_tracker)
    
    # Set up loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=c.lr, weight_decay=c.weight_decay)
    
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
                val_logits = model(x_val).squeeze()
                val_loss = criterion(val_logits, y_val.float()).item()
                val_preds = (torch.sigmoid(val_logits) >= 0.5).float()
                val_acc = (val_preds == y_val).float().mean().item()
            
            model.train()
            
            # Training step
            optimizer.zero_grad()
            logits = model(batch_x).squeeze()
            loss = criterion(logits, batch_y.float())
            
            # Add L1 regularization if specified
            l1_reg_loss = model.get_l1_regularization_loss()
            total_loss = loss + l1_reg_loss
            train_loss = total_loss.item()
            
            # Calculate training accuracy
            with torch.no_grad():
                train_preds = (torch.sigmoid(logits) >= 0.5).float()
                train_acc = (train_preds == batch_y).float().mean().item()
            
            # Update tracker with both accuracies and losses
            tracker.update(model, val_acc, train_acc, val_loss, train_loss)
            
            total_loss.backward()
            optimizer.step()
    
    # Final evaluation
    model.eval()
    with torch.no_grad():
        # Calculate final validation accuracy
        val_logits = model(x_val).squeeze()
        val_loss = criterion(val_logits, y_val.float()).item()
        val_preds = (torch.sigmoid(val_logits) >= 0.5).float()
        val_acc = (val_preds == y_val).float().mean().item()
        
        # Calculate final training accuracy
        train_logits = model(x_train).squeeze()
        train_preds = (torch.sigmoid(train_logits) >= 0.5).float()
        train_acc = (train_preds == y_train).float().mean().item()
    
    print(f"Final: Validation Loss = {val_loss:.6f}, Validation Accuracy = {val_acc:.4f}, Training Accuracy = {train_acc:.4f}")
    
    # Show plots
    print("Generating weight evolution plots...")
    tracker.plot()
    
    return model, tracker
