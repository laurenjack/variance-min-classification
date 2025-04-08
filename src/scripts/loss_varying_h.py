import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from src import dataset_creator


def run_experiment(n, d, hidden_sizes, num_runs, learning_rate, num_epochs):
    """
    Runs experiments for a given input dimension d.
    Returns the list of average v values (one per hidden layer size).
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    avg_v_values = []  # List to store average v for each hidden size

    for h in hidden_sizes:
        run_v_values = []
        for run in range(num_runs):
            # Define the MLP model with one hidden layer of size h, no biases.
            model = nn.Sequential(
                nn.Linear(d, h, bias=False),
                nn.ReLU(),
                nn.Linear(h, 1, bias=False)  # Single output for binary classification
            ).to(device)

            # Generate the problem data for current d
            problem = dataset_creator.Gaussian(d=d, perfect_class_balance=False)
            x, y = problem.generate_dataset(n, shuffle=True)
            x, y = x.to(device), y.to(device)

            # Define loss and optimizer
            criterion = nn.BCEWithLogitsLoss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)

            # Training loop
            for epoch in range(num_epochs):
                model.train()
                optimizer.zero_grad()
                outputs = model(x).squeeze()  # Shape: [n]
                loss = criterion(outputs, y.float())
                loss.backward()
                optimizer.step()

            # --- Compute v for the current run ---
            # 1. Create v_init: n copies of the d-dimensional identity matrix: shape [n, d, d]
            v_init = torch.eye(d, device=device).unsqueeze(0).repeat(n, 1, 1)
            # 2. Feed v_init through the first layer to get v_hidden: shape [n, d, h]
            v_hidden = model[0](v_init)
            # 3. Compute z = x @ W.t() using the weight matrix of the first layer; shape [n, h]
            z = x @ model[0].weight.t()
            # 4. Create a mask where z > 0 becomes 1, else 0
            mask = (z > 0).float()
            # 5. Multiply v_hidden by the mask (unsqueezed for broadcasting): shape [n, d, h]
            v_active = v_hidden * mask.unsqueeze(1)
            # 6. Feed v_active through the final layer to get v_final: shape [n, d, 1]
            v_final = model[2](v_active)
            # 7. Compute v as the squared sum of all elements in v_final (a scalar)
            v_scalar = (v_final ** 2).sum().item()
            run_v_values.append(v_scalar)
            print(f"d: {d}, Hidden units: {h}, Run: {run + 1}, v: {v_scalar}")

        # Average v over the runs for the current hidden layer size
        avg_v = sum(run_v_values) / num_runs
        avg_v_values.append(avg_v)
        print(f"d: {d}, Hidden units: {h}, Average v over {num_runs} runs: {avg_v}")

    return avg_v_values


def main(n, d_list):
    # Configuration
    hidden_sizes = list(range(1, 11))  # h from 1 to 20
    num_runs = 30
    learning_rate = 0.001
    num_epochs = 300

    plt.figure(figsize=(10, 7))

    # Run experiments for each d in d_list and plot on the same graph
    for d in d_list:
        avg_v_values = run_experiment(n, d, hidden_sizes, num_runs, learning_rate, num_epochs)
        plt.plot(hidden_sizes, avg_v_values, marker='o', label=f'd={d}')

    plt.xlabel("Hidden Layer Size (h)")
    plt.ylabel("Average v (Squared Sum of v_final)")
    plt.title("Average v vs Hidden Layer Size for different d values")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main(10, [5, 10])