import torch

from jl.feature_experiments.feature_problem import Kaleidoscope
from jl.config import Config
from jl.single_runner import train_once


def main():
    torch.manual_seed(32911)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clean_mode = True

    problem = Kaleidoscope(
        d=20,
        centers=[10, 10, 10],
        device=device,
    )

    n = 200
    model_config = Config(
        model_type="mlp",
        d=problem.d,
        n_val=n,
        n=n,
        batch_size=n // 4,
        lr=0.3,
        epochs=1000,
        weight_decay=0.0,
        num_layers=2,
        num_class=problem.num_classes(),
        h=40,
        is_weight_tracker=False,
        down_rank_dim=None,
        width_varyer=None,
        is_norm=False,
        is_adam_w=False,
        frobenius_reg_k=0.01,
    )

    x_val, y_val, center_indices = problem.generate_dataset(
        model_config.n_val,
        shuffle=True,
        clean_mode=clean_mode,
    )
    validation_set = (
        x_val.to(device),
        y_val.to(device),
        center_indices.to(device) if center_indices is not None else None,
    )

    train_once(
        device,
        problem,
        validation_set,
        model_config,
        clean_mode=clean_mode,
    )


if __name__ == "__main__":
    main()

