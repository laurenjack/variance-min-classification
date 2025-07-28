from typing import Tuple, List, Union

import torch
from torch import Tensor
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

from src import dataset_creator


def run_experiment(
    build_model_fn,
    n: int,
    d: int,
    h_values: List[int],
    num_runs: int,
    learning_rate: float,
    num_epochs: int,
    batch_size: int,
    device: torch.device,
    layer_norm: Union[bool, str],  # bool for MLPs, str for ResNets
    validation_set: Tuple[Tensor, Tensor],
    problem
) -> tuple[list, list]:
    """Returns mean variances of validation logits and mean training losses for each hidden size h."""
    x_val, _ = validation_set
    
    # Generate all training sets once, outside the loops
    training_sets = []
    for _ in range(num_runs):
        x_train, y_train = problem.generate_dataset(n, shuffle=True)
        x_train, y_train = x_train.to(device), y_train.to(device)
        training_sets.append((x_train, y_train))
    
    mean_vars = []
    mean_train_losses = []
    for h in h_values:
        run_preds = []
        run_train_losses = []
        for run_idx in range(num_runs):
            # Use the pre-generated training data for this run
            x_train, y_train = training_sets[run_idx]

            # create data loader for batch training
            train_dataset = TensorDataset(x_train, y_train)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

            # build model with current hidden parameter
            model = build_model_fn(d, h, device, layer_norm)

            # loss and optimizer
            criterion = nn.BCEWithLogitsLoss()
            optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.001)

            # training loop
            model.train()
            for epoch in range(num_epochs):
                for batch_x, batch_y in train_loader:
                    optimizer.zero_grad()
                    logits = model(batch_x).squeeze()
                    loss = criterion(logits, batch_y.float())
                    loss.backward()
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
            run_preds.append(z_val)

        # compute variance across runs for each sample, then mean
        preds_stack = torch.stack(run_preds, dim=0)  # [num_runs, n_val]
        var_z = preds_stack.var(dim=0, unbiased=False)
        mean_var = var_z.mean().item()
        mean_vars.append(mean_var)
        
        # compute mean training loss across runs
        mean_train_loss = sum(run_train_losses) / len(run_train_losses)
        mean_train_losses.append(mean_train_loss)
        
        print(f"h = {h:2d} | mean variance = {mean_var:.4e} | mean train loss = {mean_train_loss:.4f} | train_losses = {run_train_losses}")

    return mean_vars, mean_train_losses



