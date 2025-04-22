import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from src import dataset_creator
from src.learned_dropout.models_standard import ResNetStandard
from src.learned_dropout.trainer import train_standard


def run_experiment(n, d, num_layers_list, hidden_size, num_runs, learning_rate, num_epochs, batch_size, weight_decay):
    """
    Runs experiments for input dim d.
    For each number of blocks L in num_layers_list, trains num_runs pairs of ResNets
    (no-LN vs with-LN) with L hidden layers of size hidden_size,
    then computes mean zero-centered variance (i.e. E[z^2]) of their validation outputs.
    Returns two lists: mean_var_no_ln, mean_var_with_ln.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Fixed validation set
    n_val = 1000
    problem = dataset_creator.Gaussian(d=d, perfect_class_balance=False)
    x_val, y_val = problem.generate_dataset(n_val, shuffle=True)
    x_val, y_val = x_val.to(device), y_val.float().to(device)

    mean_var_no_ln = []
    mean_var_with_ln = []

    for L in num_layers_list:
        preds_no_ln = []
        preds_with_ln = []

        for run in range(num_runs):
            # Shared training data for both models
            x_train, y_train = problem.generate_dataset(n, shuffle=True)
            x_train, y_train = x_train.to(device), y_train.float().to(device)
            dataset = (x_train, y_train, x_val, y_val)

            # define hidden sizes list: L layers of fixed size
            hidden_sizes = [hidden_size] * L

            # Model without LayerNorm
            model_no_ln = ResNetStandard(d, hidden_sizes, True, False).to(device)
            model_no_ln = train_standard(dataset, model_no_ln,
                                         batch_size, num_epochs,
                                         learning_rate, weight_decay,
                                         do_track=False, track_weights=False)
            model_no_ln.eval()
            with torch.no_grad():
                z_no = model_no_ln(x_val)
                if z_no.dim() == 1:
                    z_no = z_no.unsqueeze(1)
            preds_no_ln.append(z_no.cpu())

            # Model with LayerNorm
            model_ln = ResNetStandard(d, hidden_sizes, True, True).to(device)
            model_ln = train_standard(dataset, model_ln,
                                       batch_size, num_epochs,
                                       learning_rate, weight_decay,
                                       do_track=False, track_weights=False)
            model_ln.eval()
            with torch.no_grad():
                z_ln = model_ln(x_val)
                if z_ln.dim() == 1:
                    z_ln = z_ln.unsqueeze(1)
            preds_with_ln.append(z_ln.cpu())

            print(f"d={d}, L={L}, run={run+1}/{num_runs} — "
                  f"no-LN sample: {z_no[:3].view(-1).tolist()}, "
                  f"with-LN sample: {z_ln[:3].view(-1).tolist()}")

        # Stack into [num_runs, n_val, 1]
        t_no = torch.stack(preds_no_ln, dim=0)
        t_ln = torch.stack(preds_with_ln, dim=0)

        # Compute mean E[z^2] across runs (zero-centered variance)
        sec_mom_no = (t_no ** 2).mean(dim=0)     # [n_val,1]
        sec_mom_ln = (t_ln ** 2).mean(dim=0)

        mean_var_no = sec_mom_no.mean().item()
        mean_var_ln = sec_mom_ln.mean().item()

        mean_var_no_ln.append(mean_var_no)
        mean_var_with_ln.append(mean_var_ln)

        print(f"L={L}: mean zero‑var → no‑LN: {mean_var_no:.4f}, with‑LN: {mean_var_ln:.4f}")

    return mean_var_no_ln, mean_var_with_ln


if __name__ == "__main__":
    # Experiment settings
    n = 2000
    d_list = [10]  # input dimensions to test
    num_layers_list = list(range(1, 9))  # vary number of hidden layers from 1 to 4
    hidden_size = 10                      # fixed hidden layer size
    num_runs = 10
    lr = 0.003
    epochs = 750
    bs = 100
    wd = 0.0003

    plt.figure(figsize=(10, 7))

    for d in d_list:
        mv_no_ln, mv_ln = run_experiment(
            n, d, num_layers_list, hidden_size,
            num_runs, lr, epochs, bs, wd
        )

        plt.plot(num_layers_list, mv_no_ln, marker='o', linestyle='-',
                 label=f'd={d}, no LN (E[z²])')
        plt.plot(num_layers_list, mv_ln, marker='s', linestyle='--',
                 label=f'd={d}, with LN (E[z²])')

    plt.xlabel("Number of Hidden Layers (L)")
    plt.ylabel("Mean Zero‑Centered Variance (E[z²])")
    plt.title("Zero‑Centered Output Variance vs. Number of Layers: LayerNorm vs No LayerNorm")
    plt.xticks(num_layers_list)
    plt.legend()
    plt.grid(True)
    plt.show()
