import torch

from jl.config import Config
from jl.single_runner import train_once
from jl.feature_experiments.feature_problem import SingleFeatures


def main():
    torch.manual_seed(38175)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    problem = SingleFeatures(
        true_d=4,
        f=4,
        device=device,
        is_orthogonal=False,
        percent_correct_per_f=[0.8] * 4,
        noisy_d=12,
        random_basis=True,
    )
    n = 128

    model_config = Config(
        model_type='resnet',
        d=problem.d,
        n_val=1000,
        n=n,
        batch_size=n // 4,
        lr=0.01,
        epochs=200,
        # adam_betas=(0.9, 0.999),
        weight_decay=0.01,
        num_layers=1,
        num_class=problem.num_classes(),
        h=80,
        weight_tracker="accuracy",
        optimizer="adam_w",
        is_norm=False,
        d_model=8,
    )

    x_val, y_val, center_indices = problem.generate_dataset(
        model_config.n_val,
        shuffle=True,
        clean_mode=True,
    )
    validation_set = (x_val.to(device), y_val.to(device), center_indices.to(device))

    train_once(device, problem, validation_set, model_config, clean_mode=False)


if __name__ == "__main__":
    main()
