import math
import torch
import matplotlib.pyplot as plt

from src import dataset_creator


def train_single_hidden_node(x, y, epochs, lr):
    """
    Train a single-hidden-node network on binary classification data.

    Arguments:
        x (Tensor): shape (n, d), input features
        y (Tensor): shape (n,), binary targets in {0, 1}
        epochs (int): number of epochs to train
        lr (float): learning rate for the Adam optimizer
    """
    # Ensure x and y are on the same device (e.g., CPU vs GPU).
    device = x.device
    x = x.float().to(device)
    y = y.float().to(device)

    n, d = x.shape

    # ----- Model Parameters -----
    # First-layer weight: w in R^{d x 1}, no bias
    # Initialize w ~ N(0, sqrt(1/n)) so that Var = 1/n
    w = torch.randn(d, 1, device=device) * (1.0 / n) ** 0.5
    w.requires_grad = True

    # Second-layer bias: b in R^{1}, no weight
    # Initialize b = 0
    b = torch.zeros(1, device=device, requires_grad=True)

    # ----- Optimizer -----
    optimizer = torch.optim.Adam([w, b], lr=lr)  # standard Adam, no weight decay

    # ----- Loss Function -----
    loss_fn = torch.nn.BCEWithLogitsLoss()

    # w_prev is w_(e-1)
    w_prev = w.clone().detach()  # w_0 at initialization

    # Lists to store f(e), g(e), c(e)
    f_values = []
    g_values = []
    c_values = []

    # ----- Training Loop -----
    for e in range(1, epochs + 1):
        print(f'Epoch {e}')
        # ---- Forward pass ----
        z = x @ w  # shape: (n, 1)
        hidden = torch.relu(z)  # shape: (n, 1)
        logits = hidden + b  # shape: (n, 1)

        # ---- Compute loss ----
        loss = loss_fn(logits.view(-1), y)

        # ---- Backprop and Update ----
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ---- Compute f(e) ----
        w_e = w.clone().detach()  # w_e is the updated weight after this epoch
        dot_product = torch.dot(w_prev.view(-1), w_e.view(-1))
        norm_prev = w_prev.norm(p=2)
        norm_current = w_e.norm(p=2)

        # Avoid out-of-range for arccos
        if norm_prev.item() > 0 and norm_current.item() > 0:
            cos_sim = dot_product / (norm_prev * norm_current)
            cos_sim = torch.clip(cos_sim, -1.0, 1.0)  # numeric safety
        else:
            cos_sim = torch.tensor(1.0, device=device)  # angle = 0 if either norm=0

        f_e = torch.arccos(cos_sim) / math.pi
        f_values.append(f_e.item())

        # ---- Compute g(e) ----
        with torch.no_grad():
            activation_prev = (x @ w_prev > 0).view(-1)
            activation_curr = (x @ w_e > 0).view(-1)
            changed = (activation_prev ^ activation_curr).sum().item()
            g_e = changed / n
            g_values.append(g_e)

        # ---- Compute c(e) ----
        # c(e) = mean of elementwise absolute differences between w_e and w_(e-1)
        with torch.no_grad():
            diff = torch.abs(w_e - w_prev)  # shape (d, 1)
            c_e = diff.mean().item()
            c_values.append(c_e / lr / 10)

        # Update w_prev
        w_prev = w_e

    # ----- Plot f(e), g(e), c(e) -----
    plt.figure(figsize=(9, 5))
    plt.plot(range(1, epochs + 1), f_values, label='f(e)', marker='o')
    plt.plot(range(1, epochs + 1), g_values, label='g(e)', marker='s')
    plt.plot(range(1, epochs + 1), c_values, label='c(e)', marker='^')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('f(e), g(e), and c(e) across epochs')
    plt.legend()
    plt.grid(True)
    plt.show()


# Example usage:
if __name__ == "__main__":
    n = 100_000
    d = 100
    epochs = 100
    lr = 0.001
    problem = dataset_creator.Gaussian(d=d)
    x, y = problem.generate_dataset(n, shuffle=True)
    train_single_hidden_node(x, y, epochs, lr)