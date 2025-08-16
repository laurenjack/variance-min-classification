from typing import Tuple, List, Union

import torch
from torch import Tensor
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

from src import dataset_creator
from src.learned_dropout.config import EmpiricalConfig


def run_experiment(
    build_model_fn,
    device: torch.device,
    validation_set: Tuple[Tensor, Tensor],
    problem,
    c: EmpiricalConfig
) -> tuple[list, list, list]:
    """Returns mean variances of validation logits, mean training losses, and mean validation accuracies for each hidden size h."""
    x_val, y_val = validation_set
    
    # Generate all training sets once, outside the loops
    training_sets = []
    for _ in range(c.num_runs):
        x_train, y_train = problem.generate_dataset(c.n, shuffle=True)
        x_train, y_train = x_train.to(device), y_train.to(device)
        training_sets.append((x_train, y_train))
    
    mean_vars = []
    mean_train_losses = []
    mean_val_accuracies = []
    for h in c.h_range:
        run_preds = []
        run_train_losses = []
        run_val_accuracies = []
        for run_idx in range(c.num_runs):
            # Use the pre-generated training data for this run
            x_train, y_train = training_sets[run_idx]

            # create data loader for batch training
            train_dataset = TensorDataset(x_train, y_train)
            train_loader = DataLoader(train_dataset, batch_size=c.batch_size, shuffle=False)

            # build model with current hidden parameter
            model = build_model_fn(c, h, device)

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
                    
                    # Add L1 regularization if specified
                    l1_reg_loss = model.get_l1_regularization_loss()
                    total_loss = loss + l1_reg_loss
                    
                    total_loss.backward()
                    optimizer.step()

            # compute total training loss on whole dataset after training
            model.eval()
            with torch.no_grad():
                total_logits = model(x_train).squeeze()
                total_train_loss = criterion(total_logits, y_train.float()).item()
            
            # store total training loss for this run
            run_train_losses.append(total_train_loss)

            # validation predictions
            model.eval()
            with torch.no_grad():
                z_val = model(x_val).squeeze()  # [n_val]
                val_probs = torch.sigmoid(z_val)
                val_predictions = (val_probs > 0.5).float()
                val_accuracy = (val_predictions == y_val).float().mean().item()
            run_preds.append(z_val)
            run_val_accuracies.append(val_accuracy)

        # compute variance across runs for each sample, then mean
        preds_stack = torch.stack(run_preds, dim=0)  # [num_runs, n_val]
        var_z = preds_stack.var(dim=0, unbiased=False)
        mean_var = var_z.mean().item()
        mean_vars.append(mean_var)
        
        # compute mean training loss across runs
        mean_train_loss = sum(run_train_losses) / len(run_train_losses)
        mean_train_losses.append(mean_train_loss)
        
        # compute mean validation accuracy across runs
        mean_val_accuracy = sum(run_val_accuracies) / len(run_val_accuracies)
        mean_val_accuracies.append(mean_val_accuracy)
        
        print(f"h = {h:2d} | mean variance = {mean_var:.4e} | mean train loss = {mean_train_loss:.4f} | mean val acc = {mean_val_accuracy:.4f}")

    return mean_vars, mean_train_losses, mean_val_accuracies



