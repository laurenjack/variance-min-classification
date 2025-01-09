import torch
import torch.nn as nn
import torch.optim as optim

from src import dataset_creator


class SingleLayerClassifier(nn.Module):
    def __init__(self, input_dim):
        super(SingleLayerClassifier, self).__init__()
        std = (1.0 / input_dim) ** 0.5
        self.weight = nn.Parameter(torch.normal(0, std, size=(input_dim, 1)))

    def forward(self, x, eye_d):
        z = torch.relu(x @ self.weight)
        mask = (z > 0).float().detach()
        weight_product = self.weight.t() @ eye_d
        weight_product = weight_product * mask
        return weight_product # (eye_d @ self.weight)


def direct_mean_loss(model, x, y):
    n, d = x.shape
    y_shift = y.view(n, 1) * 2.0 - 1
    mean = model(x, torch.eye(d))
    return torch.sum(torch.mean((2 * x * y_shift - mean) ** 2, axis=0)) / 2

def train_model(x, y, model, num_epochs, learning_rate, step_size, gamma, verbose):
    # criterion = nn.BCEWithLogitsLoss()

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    for epoch in range(num_epochs):
        # outputs = model(x)
        loss = direct_mean_loss(model, x, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if verbose and (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    return model


def compute_weight_statistics(num_runs, n, d, num_epochs, learning_rate, step_size, gamma, verbose):
    final_weights = []

    for run in range(num_runs):
        if verbose:
            print(f"\nStarting run {run + 1}/{num_runs}")

        problem = dataset_creator.Gaussian(d=d)
        x, y = problem.generate_dataset(n, d)
        model = SingleLayerClassifier(d)
        trained_model = train_model(x, y, model, num_epochs, learning_rate, step_size, gamma, verbose)
        final_weights.append(trained_model.weight.data.clone())

    final_weights = torch.stack(final_weights)
    weight_variance = torch.var(final_weights, dim=0)

    return weight_variance


# Hyperparameters
d = 100
n = 200
num_runs = 100
num_epochs = 200
learning_rate = 1.0 / d ** 0.5
verbose = True
step_size = 7
gamma = 0.85

# Compute weight statistics
weight_variance = compute_weight_statistics(
    num_runs=num_runs,
    n=n,
    d=d,
    num_epochs=num_epochs,
    learning_rate=learning_rate,
    step_size=step_size,
    gamma=gamma,
    verbose=verbose
)

sd = weight_variance ** 0.5

print("\nWeight variance tensor shape:", weight_variance.shape)
print("\nWeight variances:\n", sd)
print("\nMaximum variance:", sd.max().item())
print("Minimum variance:", sd.min().item())
print("Mean variance:", sd.mean().item())