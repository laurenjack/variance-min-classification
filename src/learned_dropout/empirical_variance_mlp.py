from typing import Tuple

import torch
from torch import Tensor
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

from src import dataset_creator
from src.learned_dropout.models_standard import MLPStandard

def build_model_m1(d: int, h: int, device: torch.device, layer_norm: bool) -> nn.Module:
    """Model 1: single-hidden-layer MLPStandard: [d, h, 1] with LayerNorm and ReLU"""
    # hidden dims list for MLPStandard
    h_list = [h, d]
    return MLPStandard(d=d, h_list=h_list, relus=True, layer_norm=layer_norm).to(device)


def build_model_m2(d: int, h: int, device: torch.device, layer_norm: bool) -> nn.Module:
    """Model 2: two-hidden-layer MLPStandard: [d, h//2, d, 1] with LayerNorm and ReLU"""
    # hidden dims list for MLPStandard
    h_list = [h // 2, d, h // 2, d]
    return MLPStandard(d=d, h_list=h_list, relus=True, layer_norm=layer_norm).to(device)

def build_model_m3(d: int, h: int, device: torch.device, layer_norm: bool) -> nn.Module:
    """Model 3: four-hidden-layer MLPStandard: [d, h//4, d, h//4, d, h//4, d, h//4, d, 1] with LayerNorm and ReLU"""
    # hidden dims list for MLPStandard
    h_list = [h // 4, d, h // 4, d, h // 4, d, h // 4, d]
    return MLPStandard(d=d, h_list=h_list, relus=True, layer_norm=layer_norm).to(device)

def build_single_layer_mlp(d, h, device: torch.device, layer_norm: bool) -> nn.Module:
    return MLPStandard(d=d, h_list=[h], relus=True, layer_norm=layer_norm).to(device)


def run_experiment(
    build_model_fn,
    n: int,
    d: int,
    h_values: range,
    num_runs: int,
    learning_rate: float,
    num_epochs: int,
    batch_size: int,
    device: torch.device,
    layer_norm: bool,
    validation_set: Tuple[Tensor, Tensor],
    problem
) -> tuple[list, list]:
    """Returns mean variances of validation logits and mean training losses for each hidden size h."""
    x_val, _ = validation_set
    mean_vars = []
    mean_train_losses = []
    for h in h_values:
        run_preds = []
        run_train_losses = []
        for _ in range(num_runs):
            # generate training data
            x_train, y_train = problem.generate_dataset(n, shuffle=True)
            x_train, y_train = x_train.to(device), y_train.to(device)

            # create data loader for batch training
            train_dataset = TensorDataset(x_train, y_train)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

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
        
        print(f"h = {h:2d} | mean variance = {mean_var:.4e} | mean train loss = {mean_train_loss:.4f}")

    return mean_vars, mean_train_losses



