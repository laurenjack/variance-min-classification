import torch

from jl.config import Config
from jl.single_runner import train_once
from jl.feature_experiments.feature_problem import SingleFeatures


def main():
    # torch.manual_seed(38175)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Problem: SingleFeatures (same as second_descent.py)
    problem = SingleFeatures(
        true_d=2,
        f=2,
        device=device,
        is_orthogonal=False,
        percent_correct_per_f=[0.8] * 2,
        noisy_d=12,
        random_basis=True,
    )
    n = 60
    h = 8

    # Model configuration
    model_config = Config(
        model_type='simple-mlp',
        d=problem.d,
        n_val=1000,
        n=n,
        batch_size=n // 4,
        lr=0.03,
        epochs=200,
        weight_decay=0.001,
        num_layers=1,
        num_class=problem.num_classes(),
        h=h,
        weight_tracker="accuracy",
        width_varyer=None,
        optimizer="adam_w",
        is_norm=False,
    )

    # Generate validation set
    x_val, y_val, center_indices = problem.generate_dataset(
        model_config.n_val,
        shuffle=True,
        clean_mode=True
    )
    validation_set = (x_val.to(device), y_val.to(device), center_indices.to(device))

    # Train (train_once prints final metrics)
    model, tracker, x_train, y_train, train_center_indices = train_once(
        device,
        problem,
        validation_set,
        model_config,
    )


if __name__ == "__main__":
    main()
