import torch

from jl.config import Config
from jl.multi_experiment_grapher import run_list_experiment
from jl.posterior_minimizer.dataset_creator import HyperXorNormal


def main():
    torch.manual_seed(38175)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Problem: HyperXorNormal
    true_d = 2
    noisy_d = 12
    problem = HyperXorNormal(
        true_d=true_d,
        percent_correct=0.8,
        noisy_d=noisy_d,
        random_basis=True,
    )
    n = 128

    # Experiment parameters
    width_range = list(range(2, 81, 2))
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
        num_class=2,  # HyperXorNormal is binary classification
        h=h,
        weight_tracker="accuracy",
        width_varyer="h",
        optimizer="adam_w",
    )

    # Generate validation set
    x_val, y_val, _ = problem.generate_dataset(model_config.n_val, clean_mode=False, shuffle=True)
    # Create dummy center_indices (not used by run_list_experiment, but required for tuple format)
    center_indices = torch.zeros(model_config.n_val, dtype=torch.long)
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


if __name__ == "__main__":
    main()
