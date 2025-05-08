import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from src import dataset_creator


def build_model(d: int, h1: int, h2: int, device: torch.device) -> nn.Module:
    """Returns a two‑hidden‑layer MLP (no biases) with sizes [h1, h2]."""
    return nn.Sequential(
        nn.Linear(d, h1, bias=False),
        nn.ReLU(),
        nn.Linear(h1, h2, bias=False),
        nn.ReLU(),
        nn.Linear(h2, 1, bias=False)  # single logit output
    ).to(device)


def run_experiment(
    n: int,
    d: int,
    h_values: range,
    num_runs: int,
    learning_rate: float,
    num_epochs: int,
    vary_layer: str,
):
    """Returns a list of mean variances (one per h in *h_values*).

    Parameters
    ----------
    vary_layer : str
        Either "first" (vary first hidden size) or "second" (vary second hidden size).
    """
    assert vary_layer in {"first", "second"}, "vary_layer must be 'first' or 'second'"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # fixed sizes for non‑varied layer
    fixed_size = 10

    # validation set (same for all h and runs)
    n_val = 1000
    problem_val = dataset_creator.Gaussian(d=d, perfect_class_balance=False)
    x_val, y_val = problem_val.generate_dataset(n_val, shuffle=True)
    x_val, y_val = x_val.to(device), y_val.to(device)

    mean_vars = []
    for h in h_values:
        run_preds = []
        for _ in range(num_runs):
            # choose hidden sizes depending on which layer varies
            h1, h2 = (h, fixed_size) if vary_layer == "first" else (fixed_size, h)
            model = build_model(d, h1, h2, device)

            # dataset for this run
            problem = dataset_creator.Gaussian(d=d, perfect_class_balance=False)
            x_train, y_train = problem.generate_dataset(n, shuffle=True)
            x_train, y_train = x_train.to(device), y_train.to(device)

            # loss & optimiser
            criterion = nn.BCEWithLogitsLoss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)

            # training loop
            for _ in range(num_epochs):
                model.train()
                optimizer.zero_grad()
                logits = model(x_train).squeeze()
                loss = criterion(logits, y_train.float())
                loss.backward()
                optimizer.step()

            # validation predictions
            model.eval()
            with torch.no_grad():
                z_val = model(x_val)  # [n_val, 1]
            run_preds.append(z_val.detach())

        # stack -> [num_runs, n_val, 1]
        preds_stack = torch.stack(run_preds, dim=0)
        var_z = preds_stack.var(dim=0, unbiased=False)  # variance across runs for each sample
        mean_var = var_z.mean().item()
        mean_vars.append(mean_var)
        print(
            f"Layer: {vary_layer:>6} | h = {h:2d} | mean variance on validation = {mean_var:.4e}"
        )

    return mean_vars


def main(n: int = 100, d: int = 10):
    """Runs both experiments and plots the results."""
    # configuration
    h_range = range(1, 11)  # 1‑10 inclusive
    num_runs = 30
    lr = 1e-3
    epochs = 300

    first_layer_results = run_experiment(
        n, d, h_range, num_runs, lr, epochs, vary_layer="first"
    )
    second_layer_results = run_experiment(
        n, d, h_range, num_runs, lr, epochs, vary_layer="second"
    )

    # plot
    plt.figure(figsize=(10, 7))
    plt.plot(list(h_range), first_layer_results, marker="o", label="Vary first hidden size")
    plt.plot(list(h_range), second_layer_results, marker="s", label="Vary second hidden size")
    plt.xlabel("Hidden size being varied (h)")
    plt.ylabel("Mean variance of validation logits")
    plt.title("Effect of hidden‑layer width on ensemble variance (d = 10)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
