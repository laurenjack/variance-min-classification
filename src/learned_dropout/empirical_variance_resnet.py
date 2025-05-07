import torch
import matplotlib.pyplot as plt

from src import dataset_creator
from src.learned_dropout.models_standard import ResNetStandard
from src.learned_dropout.trainer import train_standard


def run_experiment(n, d, num_layers_list,
                   hidden_size, num_runs,
                   learning_rate, num_epochs,
                   batch_size, weight_decay):
    """
    For a fixed d and training‐set size n, sweep over num_layers_list,
    train num_runs ResNets (layer_norm="param") per L on n samples,
    share a common validation set, and return [mean_var_L for each L].
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Shared validation set
    n_val = 1000
    problem = dataset_creator.Gaussian(d=d, perfect_class_balance=False)
    x_val, y_val = problem.generate_dataset(n_val, shuffle=True)
    x_val, y_val = x_val.to(device), y_val.float().to(device)

    mean_vars = []

    for L in num_layers_list:
        preds = []
        for run in range(num_runs):
            # generate fresh training set of size n
            x_train, y_train = problem.generate_dataset(n, shuffle=True)
            x_train, y_train = x_train.to(device), y_train.float().to(device)
            dataset = (x_train, y_train, x_val, y_val)

            # build and train ResNet with LayerNorm(params)
            hidden_sizes = [hidden_size] * L
            model = ResNetStandard(d, hidden_sizes, True, layer_norm="param").to(device)
            model = train_standard(dataset, model,
                                   batch_size, num_epochs,
                                   learning_rate, weight_decay,
                                   do_track=False, track_weights=False)
            model.eval()

            with torch.no_grad():
                z = model(x_val)
                if z.dim() == 1:
                    z = z.unsqueeze(1)
            preds.append(z.cpu())

            print(f"d={d}, L={L}, run={run+1}/{num_runs} — sample z: {z[:3].view(-1).tolist()}")

        # Stack [runs, n_val, 1], compute E[z^2] per example, then average over validation
        t = torch.stack(preds, dim=0)
        sec_mom = (t ** 2).mean(dim=0)  # [n_val,1]
        mean_vars.append(sec_mom.mean().item())

        print(f"L={L}: mean E[z²]={mean_vars[-1]:.4f}")

    return mean_vars


if __name__ == "__main__":
    # three training‐set sizes to compare
    n_list = [1000, 1500, 2000]
    d = 10
    num_layers_list = list(range(1, 9))  # L = 1…8
    hidden_size = 10
    num_runs = 10
    lr = 0.003
    epochs = 750
    bs = 200
    wd = 0.0003

    plt.figure(figsize=(10, 7))

    # run for each n and plot
    for marker, (n, style) in zip(
        ["o", "s", "^"],
        zip(n_list, ["-", "--", ":"])
    ):
        mv = run_experiment(n, d, num_layers_list,
                            hidden_size, num_runs,
                            lr, epochs, bs, wd)
        plt.plot(num_layers_list, mv,
                 marker=marker,
                 linestyle=style,
                 label=f"n={n}")

    plt.xlabel("Number of Hidden Layers (L)")
    plt.ylabel("Mean Zero-Centered Variance (E[z²])")
    plt.title("Effect of Training-Set Size on Output Variance\n(all models use LayerNorm(param), d=10)")
    plt.xticks(num_layers_list)
    plt.legend()
    plt.grid(True)
    plt.show()
