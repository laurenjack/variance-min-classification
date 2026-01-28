import torch

from jl.config import Config
from jl.single_runner import train_once
from jl.multi_experiment_grapher import run_list_experiment
from jl.feature_experiments.feature_problem import SingleFeatures


def main():
    torch.manual_seed(38175)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Problem: SingleFeatures
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

    # Experiment parameters
    width_range = list(range(3, 161, 1))
    h = max(width_range)
    num_runs = 20

    # Model configuration
    model_config = Config(
        model_type='simple-mlp',
        d=problem.d,
        n_val=1000,
        n=n,
        batch_size=n // 4,
        lr=0.01,
        epochs=200,
        weight_decay=0.001,
        num_layers=1,
        num_class=problem.num_classes(),
        h=h,
        weight_tracker="accuracy",
        width_varyer="h",
        optimizer="adam_w",
    )

    # Generate validation set with class-balanced sampling
    x_val, y_val, center_indices, _ = problem.generate_dataset(
        model_config.n_val,
        shuffle=True,
        clean_mode=False
    )
    validation_set = (x_val.to(device), y_val.to(device), center_indices.to(device))

    # Run experiments
    run_list_experiment(
        device,
        problem,
        validation_set,
        [model_config],
        width_range,
        num_runs,
        clean_mode=False
    )

    # train_once(device, problem, validation_set, model_config)


if __name__ == "__main__":
    main()
