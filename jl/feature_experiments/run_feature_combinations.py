from sympy import false
import torch

from jl.feature_experiments.feature_combinations import FeatureCombinations
from jl.config import Config
from jl.single_runner import train_once


def main():
    torch.manual_seed(594732)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    problem = FeatureCombinations(
        num_layers=2,
        random_basis=False,
        device=device,
    )

    n = 100
    model_config = Config(
        model_type="resnet",
        d=problem.d,
        n_val=n,
        n=n,
        batch_size=n // 5,
        lr=0.01,
        epochs=20,
        weight_decay=0.0,
        num_layers=1,
        num_class=problem.num_classes(),
        h=20,
        weight_tracker="full_step",
        down_rank_dim=None,
        width_varyer=None,
        is_norm=True,
        optimizer="reg_adam_w",
        learnable_norm_parameters=False,
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
