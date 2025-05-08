import torch
import matplotlib.pyplot as plt

from src import dataset_creator
from src.learned_dropout.models_standard import ResNetStandard
from src.learned_dropout.trainer import train_standard


def run_variation_block(block_to_vary: int,
                        hidden_sizes_varied: list[int],
                        *,
                        n: int,
                        d: int,
                        num_runs: int,
                        learning_rate: float,
                        num_epochs: int,
                        batch_size: int,
                        weight_decay: float,
                        device: torch.device) -> list[float]:
    """Compute the mean zero‑centered variance E[z²] on a shared validation set
    for a suite of models in which *one* residual block size is varied.

    Args:
        block_to_vary: Which residual block (0‑based index) should have its
            hidden size swept over ``hidden_sizes_varied``.
        hidden_sizes_varied: List of hidden sizes to plug into the chosen block.
        n: Training‑set size.
        d: Input dimension.
        num_runs: Number of independent training runs per configuration.
        learning_rate, num_epochs, batch_size, weight_decay: Training hyper‑parameters.
        device: CPU or CUDA device.

    Returns:
        List of mean E[z²]—one entry per hidden size in *hidden_sizes_varied*.
    """

    # Shared validation set
    n_val = 1000
    problem = dataset_creator.Gaussian(d=d, perfect_class_balance=False)
    x_val, y_val = problem.generate_dataset(n_val, shuffle=True)
    x_val, y_val = x_val.to(device), y_val.float().to(device)

    mean_vars: list[float] = []

    for h in hidden_sizes_varied:
        preds = []
        for run in range(num_runs):
            # Fresh training set of size n
            x_train, y_train = problem.generate_dataset(n, shuffle=True)
            x_train, y_train = x_train.to(device), y_train.float().to(device)

            hidden_sizes = [10, 10, 10]  # default sizes for the 3 residual blocks
            hidden_sizes[block_to_vary] = h           # vary the chosen block

            model = ResNetStandard(d, hidden_sizes, True, layer_norm="param").to(device)
            dataset = (x_train, y_train, x_val, y_val)
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

            print(f"block={block_to_vary+1}, h={h}, run={run+1}/{num_runs} — sample z: {z[:3].view(-1).tolist()}")

        # Stack shape: [num_runs, n_val, 1]
        t = torch.stack(preds, dim=0)
        sec_mom = (t ** 2).mean(dim=0)          # [n_val, 1]
        mean_vars.append(sec_mom.mean().item())

        print(f"Varying block {block_to_vary+1}, h={h}: mean E[z²]={mean_vars[-1]:.4f}")

    return mean_vars


if __name__ == "__main__":
    # Experimental setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n = 1000              # training‑set size
    d = 10                # input dimension
    num_runs = 10         # independent runs per configuration
    lr = 0.003            # learning rate
    epochs = 750          # training epochs
    bs = 200              # batch size
    wd = 0.0003           # weight decay

    hidden_sizes_varied = list(range(30, 110, 10))  # 5, 10, …, 40

    plt.figure(figsize=(10, 7))

    # ── Model 1: vary *first* residual block ──────────────────────────────────
    mv1 = run_variation_block(block_to_vary=0,
                              hidden_sizes_varied=hidden_sizes_varied,
                              n=n, d=d,
                              num_runs=num_runs,
                              learning_rate=lr,
                              num_epochs=epochs,
                              batch_size=bs,
                              weight_decay=wd,
                              device=device)
    plt.plot(hidden_sizes_varied, mv1,
             marker="o", linestyle="-", label="Vary first block")

    # ── Model 2: vary *third* residual block ─────────────────────────────────
    mv2 = run_variation_block(block_to_vary=2,
                              hidden_sizes_varied=hidden_sizes_varied,
                              n=n, d=d,
                              num_runs=num_runs,
                              learning_rate=lr,
                              num_epochs=epochs,
                              batch_size=bs,
                              weight_decay=wd,
                              device=device)
    plt.plot(hidden_sizes_varied, mv2,
             marker="s", linestyle="--", label="Vary third block")

    # ── Plot styling ─────────────────────────────────────────────────────────
    plt.xlabel("Hidden Units in Varied Block")
    plt.ylabel("Mean Zero‑Centered Variance (E[z²])")
    plt.title("Effect of Hidden Units per Block on Output Variance\n"
              "(d=10, n=1000, 3 residual blocks, LayerNorm(param))")
    plt.xticks(hidden_sizes_varied)
    plt.legend()
    plt.grid(True)
    plt.show()
