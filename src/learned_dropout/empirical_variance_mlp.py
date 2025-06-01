import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from src import dataset_creator
from src.learned_dropout.models_standard import MLPStandard

def build_model_m1(d: int, h: int, device: torch.device) -> nn.Module:
    """Model 1: single-hidden-layer MLPStandard: [d, h, 1] with LayerNorm and ReLU"""
    # hidden dims list for MLPStandard
    h_list = [h // 4, d, h // 4, d, h // 2, d]
    return MLPStandard(d=d, h_list=h_list, relus=True, layer_norm=True).to(device)


def build_model_m2(d: int, h: int, device: torch.device) -> nn.Module:
    """Model 2: two-hidden-layer MLPStandard: [d, h//2, d, 1] with LayerNorm and ReLU"""
    # hidden dims list for MLPStandard
    h_list = [h // 2, d, h // 4, d, h // 4, d]
    return MLPStandard(d=d, h_list=h_list, relus=True, layer_norm=True).to(device)

def build_model_m3(d: int, h: int, device: torch.device) -> nn.Module:
    """Model 2: two-hidden-layer MLPStandard: [d, h//2, d, 1] with LayerNorm and ReLU"""
    # hidden dims list for MLPStandard
    h_list = [h // 4, d, h // 4, d, h // 4, d, h // 4, d]
    return MLPStandard(d=d, h_list=h_list, relus=True, layer_norm=True).to(device)


def run_experiment(
    build_model_fn,
    n: int,
    d: int,
    h_values: range,
    num_runs: int,
    learning_rate: float,
    num_epochs: int,
    device: torch.device
) -> list:
    """Returns mean variances of validation logits for each hidden size h."""
    # prepare validation set (same across runs)
    n_val = 1000
    problem_val = dataset_creator.Gaussian(d=d, perfect_class_balance=False)
    x_val, y_val = problem_val.generate_dataset(n_val, shuffle=True)
    x_val, y_val = x_val.to(device), y_val.to(device)

    mean_vars = []
    for h in h_values:
        run_preds = []
        for _ in range(num_runs):
            # build model with current hidden parameter
            model = build_model_fn(d, h, device)

            # generate training data
            problem = dataset_creator.Gaussian(d=d, perfect_class_balance=False)
            x_train, y_train = problem.generate_dataset(n, shuffle=True)
            x_train, y_train = x_train.to(device), y_train.to(device)

            # loss and optimizer
            criterion = nn.BCEWithLogitsLoss()
            optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.001)

            # training loop
            model.train()
            for _ in range(num_epochs):
                optimizer.zero_grad()
                logits = model(x_train).squeeze()
                loss = criterion(logits, y_train.float())
                loss.backward()
                optimizer.step()

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
        print(f"h = {h:2d} | mean variance = {mean_var:.4e}")

    return mean_vars


def main(n: int = 1000, d: int = 40):
    # configuration
    h_range = range(4, 101, 4)  # 4, 6, 8, ..., 20
    num_runs = 20
    lr = 3e-3
    epochs = 300
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # # run experiments for both models
    print("Running Model 1")
    m1_results = run_experiment(
        build_model_m1, n, d, h_range, num_runs, lr, epochs, device
    )

    print("\nRunning Model 2")
    m2_results = run_experiment(
        build_model_m2, n, d, h_range, num_runs, lr, epochs, device
    )

    print("\nRunning Model 3")
    m3_results = run_experiment(
        build_model_m3, n, d, h_range, num_runs, lr, epochs, device
    )

    # plot results
    plt.figure(figsize=(10, 7))
    plt.plot(list(h_range), m1_results, marker="o", label="Model 1")
    plt.plot(list(h_range), m2_results, marker="s", label="Model 2")
    plt.plot(list(h_range), m3_results, marker="s", label="Model 3")
    plt.xlabel("Hidden size parameter h")
    plt.ylabel("Mean variance of validation logits")
    plt.title(f"Ensemble variance vs. hidden size (d = {d})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
