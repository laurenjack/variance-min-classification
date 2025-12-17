import torch

from jl.feature_experiments.feature_problem import TiltedKaleidoscope, Kaleidoscope
from jl.config import Config
from jl.single_runner import train_once


def main():
    torch.manual_seed(594732)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    problem = TiltedKaleidoscope(
        d=12,
        centers=[10, 10, 10, 10],
        device=device,
    )

    n = 1000
    model_config = Config(
        model_type="resnet",
        d=problem.d,
        n_val=n,
        n=n,
        batch_size=n // 10,
        lr=0.001,
        epochs=500,
        weight_decay=0.1,
        num_layers=8,
        num_class=problem.num_classes(),
        h=40,
        is_weight_tracker=False,
        down_rank_dim=None,
        width_varyer=None,  
        is_norm=True,
        is_adam_w=True,
        learnable_norm_parameters=True,
    )

    x_val, y_val, val_center_indices = problem.generate_dataset(
        model_config.n_val,
        shuffle=True,
    )
    validation_set = (
        x_val,
        y_val,
        val_center_indices,
    )

    model, tracker, x_train, y_train, train_center_indices = train_once(
        device,
        problem,
        validation_set,
        model_config,
    )


if __name__ == "__main__":
    main()

