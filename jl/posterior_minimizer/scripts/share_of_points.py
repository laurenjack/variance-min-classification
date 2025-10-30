import numpy as np
import torch
import matplotlib.pyplot as plt

from jl.posterior_minimizer import dataset_creator

def train_ten_hidden_nodes_plot_activation_histogram(x, y, epochs, lr, h):
    """
    Trains a model with:
      - 10 hidden nodes (no bias) in the first layer
      - A second weight matrix w1 (shape: (h,1)) for the output layer
      - ReLU activation
      - BCEWithLogitsLoss

    Instead of plotting h(e), we collect how many epochs each sample
    was activated by the *first* hidden node, then plot a histogram
    of these counts.
    """
    device = x.device
    x = x.float().to(device)
    y = y.float().to(device)

    n, d = x.shape

    # Initialize w ~ N(0, sqrt(1/d))
    w = torch.randn(d, h, device=device) * (1.0 / d)**0.5
    w.requires_grad = True

    # Initialize w1 ~ N(0, sqrt(1/h))
    w1 = torch.randn(h, 1, device=device) * (1.0 / h)**0.5
    w1.requires_grad = True

    # Optimizer
    optimizer = torch.optim.Adam([w, w1], lr=lr)

    # Loss
    loss_fn = torch.nn.BCEWithLogitsLoss()

    # Track how many epochs each sample is activated by the *first* hidden node
    active_counts = torch.zeros(n, dtype=torch.int32, device=device)

    # ----- Training Loop -----
    for e in range(1, epochs + 1):
        print(f"Epoch {e}")

        # Forward pass
        hidden = torch.relu(x @ w)   # shape: (n, h)
        logits = hidden @ w1        # shape: (n, 1)

        # Compute loss
        loss = loss_fn(logits.view(-1), y)

        # Backprop + update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update active counts for the first hidden node
        with torch.no_grad():
            activation_curr = (x @ w[:, 0] > 0)  # boolean vector (n,)
            # Increment count for each sample activated in this epoch
            active_counts += activation_curr.int()

    # ----- After training, plot histogram of active_counts -----
    # Move data to CPU and convert to NumPy for plotting
    active_counts_cpu = active_counts.cpu().numpy()
    print(f"Mean active count: {np.mean(active_counts_cpu)}")
    always = np.sum(active_counts_cpu == 100) / n
    print(f"Always: {always}")
    never = np.sum(active_counts_cpu == 0) / n
    print(f"Never: {never}")
    between = 1 - always - never
    print(f"Between {between}")


    plt.figure(figsize=(8, 5))
    plt.hist(active_counts_cpu, bins=30, edgecolor='k')
    plt.xlabel('Number of epochs a point was active (first hidden node)')
    plt.ylabel('Count of points')
    plt.title('Histogram of active counts across samples')
    plt.grid(True)
    plt.show()


# Example usage
if __name__ == "__main__":
    n = 100_000
    d = 100
    h = 100
    epochs = 1000
    lr = 0.01
    # Note: Perfect class balance = False, as in your example
    problem = dataset_creator.Gaussian(d=d, perfect_class_balance=False)
    x, y = problem.generate_dataset(n, shuffle=True)
    train_ten_hidden_nodes_plot_activation_histogram(x, y, epochs, lr, h)
