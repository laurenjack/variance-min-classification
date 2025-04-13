import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from src import dataset_creator


class ResNetBlock(nn.Module):
    def __init__(self, d, h):
        """
        Constructs the ResNet block.
        - d: input dimension
        - h: hidden dimension for the inner block
        """
        super(ResNetBlock, self).__init__()
        self.layer_in = nn.Linear(d, h, bias=False)
        self.layer_out = nn.Linear(h, d, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Compute the residual: x -> Linear (d->h) -> ReLU -> Linear (h->d)
        residual = self.layer_out(self.relu(self.layer_in(x)))
        # Add the skip connection: output has shape [n, d]
        return x + residual


class ResNetModel(nn.Module):
    def __init__(self, d, h):
        """
        Constructs the full model:
        - d: input and output dimension for the residual block and final layer
        - h: hidden dimension for the residual block
        """
        super(ResNetModel, self).__init__()
        self.res_block = ResNetBlock(d, h)
        self.final_layer = nn.Linear(d, 1, bias=False)

    def forward(self, x):
        out = self.res_block(x)
        out = self.final_layer(out)
        return out


def run_experiment(n, d, hidden_sizes, num_runs, learning_rate, num_epochs):
    """
    Runs experiments for a given input dimension d.
    For each hidden layer size in hidden_sizes, trains num_runs ResNet models,
    computes the model's output on a fixed validation set, and calculates
    mean_var: the mean over the validation set of the variance (across runs)
    of the model outputs.
    Returns the list of mean_var values (one per hidden layer size).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Generate a validation set using the dataset creator
    n_val = 1000
    problem_val = dataset_creator.Gaussian(d=d, perfect_class_balance=False)
    x_val, y_val = problem_val.generate_dataset(n_val, shuffle=True)
    x_val, y_val = x_val.to(device), y_val.to(device)

    mean_var_values = []  # To store mean_var for each hidden layer size

    for h in hidden_sizes:
        run_predictions = []  # To store validation predictions for each run

        for run in range(num_runs):
            # Instantiate the ResNet model
            model = ResNetModel(d, h).to(device)

            # Generate training data for current d
            problem = dataset_creator.Gaussian(d=d, perfect_class_balance=False)
            x, y = problem.generate_dataset(n, shuffle=True)
            x, y = x.to(device), y.to(device)

            # Define loss and optimizer
            criterion = nn.BCEWithLogitsLoss()
            optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0003)

            # Training loop
            model.train()
            for epoch in range(num_epochs):
                optimizer.zero_grad()
                outputs = model(x).squeeze()  # Should be shape [n]
                loss = criterion(outputs, y.float())
                loss.backward()
                optimizer.step()

            # Evaluate on the validation set
            model.eval()
            with torch.no_grad():
                z_val = model(x_val)  # Shape: [n_val, 1]
                # Ensure shape consistency if needed
                if len(z_val.shape) == 1:
                    z_val = z_val.unsqueeze(1)
            run_predictions.append(z_val.detach())
            print(f"d: {d}, Hidden units: {h}, Run: {run + 1}, Validation output sample: {z_val[:3].view(-1)}")

        # Stack predictions to a tensor of shape [num_runs, n_val, 1]
        predictions_tensor = torch.stack(run_predictions, dim=0)
        # Compute variance of the predictions for each validation observation (across runs)
        var_z = predictions_tensor.var(dim=0, unbiased=False)  # Shape: [n_val, 1]
        # mean_var is the average variance across the validation set
        mean_var = var_z.mean().item()
        mean_var_values.append(mean_var)
        print(f"d: {d}, Hidden units: {h}, Mean variance over validation set: {mean_var}")

    return mean_var_values


def main(n, d_list):
    # Configuration
    hidden_sizes = list(range(1, 11))  # h from 1 to 10
    num_runs = 30
    learning_rate = 0.001
    num_epochs = 300

    plt.figure(figsize=(10, 7))

    # Run experiments for each d in d_list and plot on the same graph
    for d in d_list:
        mean_var_values = run_experiment(n, d, hidden_sizes, num_runs, learning_rate, num_epochs)
        plt.plot(hidden_sizes, mean_var_values, marker='o', label=f'd={d}')

    plt.xlabel("Hidden Layer Size (h)")
    plt.ylabel("Mean Variance of Validation Outputs")
    plt.title("Mean Variance vs Hidden Layer Size for different d values (ResNet Model)")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main(100, [5, 10, 20])
