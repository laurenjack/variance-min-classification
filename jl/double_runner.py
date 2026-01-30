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


def train_double(device, problem, validation_set, c: Config):
    """
    Create and train two models side by side with cross-model target averaging.
    
    Each model is trained on its own dataset, but the training target is modified
    to be the average of the true one-hot label and the detached probability
    prediction from the other model on the same batch.
    
    Parameters:
        device: torch.device to run computation on
        problem: Dataset generator with generate_dataset method
        validation_set: Tuple of (x_val, y_val, center_indices) tensors
        c: Config containing training and architecture parameters
        
    Returns:
        Tuple of (model1, model2, x_train1, y_train1, x_train2, y_train2)
        
    Raises:
        ValueError: If c.num_class <= 2, c.is_hashed_dropout is True, or 
                    c.weight_tracker != 'accuracy'
    """
    # Validate constraints
    if c.num_class <= 2:
        raise ValueError(f"double_runner only supports num_class > 2, got {c.num_class}")
    if c.is_hashed_dropout:
        raise ValueError("double_runner does not support hashed dropout")
    if c.weight_tracker != 'accuracy':
        raise ValueError(f"double_runner only supports weight_tracker='accuracy', got {c.weight_tracker}")
    
    # Generate two training sets from the same problem (clean_mode=False)
    x_train1, y_train1, _ = problem.generate_dataset(c.n, shuffle=True, clean_mode=False)
    x_train2, y_train2, _ = problem.generate_dataset(c.n, shuffle=True, clean_mode=False)
    
    # Create datasets and data loaders
    train_dataset1 = TensorDataset(x_train1, y_train1)
    train_dataset2 = TensorDataset(x_train2, y_train2)
    train_loader1 = DataLoader(train_dataset1, batch_size=c.batch_size, shuffle=True)
    train_loader2 = DataLoader(train_dataset2, batch_size=c.batch_size, shuffle=True)
    
    # Calculate training steps for scheduler
    steps_per_epoch = len(train_loader1)
    training_steps = steps_per_epoch * c.epochs
    
    # Adjust initial lr for scheduler if needed
    initial_lr = c.lr
    if c.lr_scheduler == 'wsd':
        warmup_steps = round(0.05 * training_steps)
        if warmup_steps > 0:
            initial_lr = c.lr / warmup_steps
    
    # Create two models
    model1 = create_model(c).to(device)
    model2 = create_model(c).to(device)
    
    # Create weight trackers
    tracker1 = model1.get_tracker(c)
    tracker2 = model2.get_tracker(c)
    
    # Loss function for computing metrics (not for training)
    criterion = nn.CrossEntropyLoss()
    
    # Create optimizers for both models
    def create_optimizer(model):
        if c.optimizer == "adam_w":
            return optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=c.weight_decay,
                               eps=c.adam_eps, betas=c.adam_betas)
        elif c.optimizer == "sgd":
            return optim.SGD(model.parameters(), lr=initial_lr, momentum=c.sgd_momentum,
                             weight_decay=c.weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {c.optimizer}")
    
    optimizer1 = create_optimizer(model1)
    optimizer2 = create_optimizer(model2)
    
    # Create learning rate schedulers
    scheduler1 = create_lr_scheduler(optimizer1, training_steps, c.lr, c.lr_scheduler)
    scheduler2 = create_lr_scheduler(optimizer2, training_steps, c.lr, c.lr_scheduler)
    
    # Get validation set
    x_val, y_val, center_indices = validation_set
    
    # Training loop
    for epoch in range(c.epochs):
        model1.train()
        model2.train()
        
        # Zip the two loaders together - they have the same size
        for batch_data1, batch_data2 in zip(train_loader1, train_loader2):
            batch_x1, batch_y1 = batch_data1
            batch_x2, batch_y2 = batch_data2
            
            # Calculate validation accuracy and loss for both trackers
            model1.eval()
            model2.eval()
            with torch.no_grad():
                # Model 1 validation metrics
                val_logits1 = model1(x_val)
                val_loss1 = criterion(val_logits1, y_val.long()).item()
                val_preds1 = val_logits1.argmax(dim=1)
                val_acc1 = (val_preds1 == y_val.long()).float().mean().item()
                
                # Model 2 validation metrics
                val_logits2 = model2(x_val)
                val_loss2 = criterion(val_logits2, y_val.long()).item()
                val_preds2 = val_logits2.argmax(dim=1)
                val_acc2 = (val_preds2 == y_val.long()).float().mean().item()
            
            model1.train()
            model2.train()
            
            # Forward pass: get cross-model predictions
            with torch.no_grad():
                # Feed batch 1 through model 2 to get probabilities
                logits2_from_1 = model2(batch_x1)
                p2_from_1 = F.softmax(logits2_from_1, dim=1).detach()
                
                # Feed batch 2 through model 1 to get probabilities
                logits1_from_2 = model1(batch_x2)
                p1_from_2 = F.softmax(logits1_from_2, dim=1).detach()
            
            # One-hot encode the labels
            y_onehot1 = F.one_hot(batch_y1.long(), num_classes=c.num_class).float()
            y_onehot2 = F.one_hot(batch_y2.long(), num_classes=c.num_class).float()
            
            # Create soft targets by weighted averaging true label and cross-model prediction
            soft_target1 = (y_onehot1 + c.prob_weight * p2_from_1) / (1.0 + c.prob_weight)
            soft_target2 = (y_onehot2 + c.prob_weight * p1_from_2) / (1.0 + c.prob_weight)
            
            # Training step for model 1
            optimizer1.zero_grad()
            logits1 = model1(batch_x1)
            loss1 = soft_cross_entropy(logits1, soft_target1)
            
            # Add logit regularization if specified
            if c.c is not None:
                centered_logits1 = logits1 - logits1.mean(dim=1, keepdim=True)
                logit_reg1 = c.c * torch.mean(centered_logits1 ** 2)
                loss1 = loss1 + logit_reg1
            
            loss1.backward()
            
            # Calculate training metrics for model 1 (using standard loss, not soft targets)
            with torch.no_grad():
                train_loss1 = criterion(logits1, batch_y1.long()).item()
                train_preds1 = logits1.argmax(dim=1)
                train_acc1 = (train_preds1 == batch_y1.long()).float().mean().item()
            
            # Update tracker for model 1 (before optimizer step for 'accuracy' mode)
            tracker1.update(model1, val_acc1, train_acc1, val_loss1, train_loss1)
            optimizer1.step()
            
            # Training step for model 2
            optimizer2.zero_grad()
            logits2 = model2(batch_x2)
            loss2 = soft_cross_entropy(logits2, soft_target2)
            
            # Add logit regularization if specified
            if c.c is not None:
                centered_logits2 = logits2 - logits2.mean(dim=1, keepdim=True)
                logit_reg2 = c.c * torch.mean(centered_logits2 ** 2)
                loss2 = loss2 + logit_reg2
            
            loss2.backward()
            
            # Calculate training metrics for model 2 (using standard loss, not soft targets)
            with torch.no_grad():
                train_loss2 = criterion(logits2, batch_y2.long()).item()
                train_preds2 = logits2.argmax(dim=1)
                train_acc2 = (train_preds2 == batch_y2.long()).float().mean().item()
            
            # Update tracker for model 2 (before optimizer step for 'accuracy' mode)
            tracker2.update(model2, val_acc2, train_acc2, val_loss2, train_loss2)
            optimizer2.step()
            
            # Step schedulers if enabled
            if scheduler1 is not None:
                scheduler1.step()
            if scheduler2 is not None:
                scheduler2.step()
    
    # Final evaluation
    model1.eval()
    model2.eval()
    with torch.no_grad():
        # Model 1 final validation metrics
        val_logits1 = model1(x_val)
        val_loss1 = criterion(val_logits1, y_val.long()).item()
        val_preds1 = val_logits1.argmax(dim=1)
        val_acc1 = (val_preds1 == y_val.long()).float().mean().item()
        
        # Model 1 final training metrics
        train_logits1 = model1(x_train1)
        train_loss1 = criterion(train_logits1, y_train1.long()).item()
        train_preds1 = train_logits1.argmax(dim=1)
        train_acc1 = (train_preds1 == y_train1.long()).float().mean().item()
        
        # Model 2 final validation metrics
        val_logits2 = model2(x_val)
        val_loss2 = criterion(val_logits2, y_val.long()).item()
        val_preds2 = val_logits2.argmax(dim=1)
        val_acc2 = (val_preds2 == y_val.long()).float().mean().item()
        
        # Model 2 final training metrics
        train_logits2 = model2(x_train2)
        train_loss2 = criterion(train_logits2, y_train2.long()).item()
        train_preds2 = train_logits2.argmax(dim=1)
        train_acc2 = (train_preds2 == y_train2.long()).float().mean().item()
        
        # Ensemble metrics: average logits
        ensemble_val_logits = (val_logits1 + val_logits2) / 2
        ensemble_val_loss = criterion(ensemble_val_logits, y_val.long()).item()
        ensemble_val_preds = ensemble_val_logits.argmax(dim=1)
        ensemble_val_acc = (ensemble_val_preds == y_val.long()).float().mean().item()
    
    # Show plots first
    print("Generating weight evolution plots for Model 1...")
    tracker1.plot()
    print("Generating weight evolution plots for Model 2...")
    tracker2.plot()
    
    # Then print metrics
    print(f"Model 1 Final: Validation Loss = {val_loss1:.6f}, Validation Accuracy = {val_acc1:.4f}, "
          f"Training Loss = {train_loss1:.6f}, Training Accuracy = {train_acc1:.4f}")
    print(f"Model 2 Final: Validation Loss = {val_loss2:.6f}, Validation Accuracy = {val_acc2:.4f}, "
          f"Training Loss = {train_loss2:.6f}, Training Accuracy = {train_acc2:.4f}")
    print(f"Ensemble Final: Validation Loss = {ensemble_val_loss:.6f}, Validation Accuracy = {ensemble_val_acc:.4f}")
    
    return model1, model2, x_train1, y_train1, x_train2, y_train2

