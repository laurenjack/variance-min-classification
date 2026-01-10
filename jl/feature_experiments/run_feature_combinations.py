from sympy import false
import torch

from jl.feature_experiments.feature_combinations import FeatureCombinations
from jl.config import Config
from jl.single_runner import train_once


def main():
    torch.manual_seed(659)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    problem = FeatureCombinations(
        num_layers=3,
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
        lr=0.003,
        epochs=30,
        weight_decay=0.0,
        num_layers=2,
        num_class=problem.num_classes(),
        h=40,
        weight_tracker=None,
        width_varyer=None,
        is_norm=True,
        optimizer="adam_w",
        c = 0.01,
        learnable_norm_parameters=True,
        lr_scheduler=None,
    )

    x_val, y_val, val_center_indices, _ = problem.generate_dataset(
        model_config.n_val,
        shuffle=True,
    )
    validation_set = (
        x_val,
        y_val,
        val_center_indices,
    )

    model, tracker, x_train, y_train, train_center_indices, _ = train_once(
        device,
        problem,
        validation_set,
        model_config,
    )

    print(torch.sigmoid(model(x_val[:5])))


if __name__ == "__main__":
    main()
