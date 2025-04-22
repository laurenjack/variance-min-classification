import torch
import torch.nn as nn
import torch.optim as optim


def train_and_get_sum(n_samples: int, d: int, epochs: int = 100, lr: float = 1e-2) -> torch.Tensor:
    """
    Generates a dataset with n_samples from N(0,1), a gating matrix R of shape (n_samples, d)
    with entries r_{ij} ~ Bernoulli(0.5), fits weights w_j (j=1..d) to minimize MSE over several epochs,
    and returns sum_j w_j.
    """
    # Generate data
    x = torch.randn(n_samples)                   # shape: (n_samples,)
    # Gating: sample independent Bernoulli for each sample and each weight
    R = torch.bernoulli(0.5 * torch.ones(n_samples, d))  # shape: (n_samples, d)

    # Initialize weights
    w = torch.zeros(d, requires_grad=True)
    optimizer = optim.Adam([w], lr=lr)
    criterion = nn.MSELoss(reduction='mean')

    # Training loop
    for epoch in range(epochs):
        optimizer.zero_grad()
        # Compute predictions: preds_i = sum_j R[i,j] * w[j]
        preds = R.matmul(w)  # shape: (n_samples,)
        loss = criterion(preds, x)
        loss.backward()
        optimizer.step()

    # Return sum of weights
    return w.detach().sum()


if __name__ == '__main__':
    # Experiment: run 30 independent trainings and collect sum(w_j)
    n_runs = 30
    results = []
    for _ in range(n_runs):
        total_w = train_and_get_sum(n_samples=1000, d=10, epochs=200, lr=1e-2)
        results.append(total_w)

    # Compute empirical variance of sum(w_j)
    results_tensor = torch.stack(results)  # shape: (n_runs,)
    empirical_variance = results_tensor.var(unbiased=False)
    print(f"Empirical variance of sum(w_j) over {n_runs} runs: {empirical_variance.item():.6f}")
