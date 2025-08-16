import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from src import dataset_creator as dc

class Model(nn.Module):

    def __init__(self, d, h_list):
        super(Model, self).__init__()
        self.layers = nn.ModuleList()
        h = d
        for h in h_list:
            self.layers.append(nn.Linear(d, h, bias=True))
            self.layers.append(nn.ReLU())
            d = h
        self.layers.append(nn.Linear(h, 1, bias=True))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


torch.manual_seed(4565)
runs = 100
n = 2000
batch_size = 1000
d = 30
h_list = [20, 10]
epochs = 500
# desired_success_rate = 0.95
# alpha = (1 - desired_success_rate ** (1 / d))
# alpha = 0.2
k = 1.2

zero_better_count = 0
for r in range(runs):
    problem = dc.Gaussian(d=d, perfect_class_balance=False)
    model = Model(d, h_list)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.0)
    x, y = problem.generate_dataset(n, shuffle=True)
    y = y.float()
    for e in range(epochs):
        permutation = torch.randperm(n)
        for i in range(0, n, batch_size):
            batch_indices = permutation[i:i + batch_size]
            x_batch = x[batch_indices]
            y_batch = y[batch_indices]
            z = model(x_batch).squeeze(1)
            loss = criterion(z, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # if e % 10 == 0:
        #     z_full = model(x).squeeze(1)
        #     loss_full = criterion(z_full, y)
        #     print(f"Epoch: {e}, Training Loss: {loss_full.item()}")
        if e + 1 == epochs:
            z_full = model(x).squeeze(1)
            loss_full = criterion(z_full, y)
            final_loss = loss_full.item()
    z0 = torch.zeros(n)
    sizes = [d] + h_list + [1]
    param_count = 0
    for s0, s1 in zip(sizes[:-1], sizes[1:]):
        param_count += (s0 + 1) * s1
    prior_constant = k * param_count / n
    loss0 = criterion(z0, y) - prior_constant
    if loss0 < final_loss:
        zero_better_count += 1
    print(f"Parameter Count: {param_count}")
    print(f"Run: {r}")
    print(f"    Final training loss: {final_loss}")
    # print(f"Prior Constant: {prior_constant}")
    print(f"    Zero training Loss: {loss0}")
print("")
print(f"Zero better count: {zero_better_count}")




