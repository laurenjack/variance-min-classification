import torch

from jl.variance_experiments.data_generator import SubDirections
from jl.config import Config
from jl.single_runner import train_once


def main():
    torch.manual_seed(4454)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Problem: SubDirections with requested parameters
    percent_correct = 0.8
    clean_mode = True
    problem = SubDirections(
        true_d=16,
        sub_d=4,
        centers=8,
        num_class=2,
        sigma=0.02,
        noisy_d=0,
        random_basis=True,
        percent_correct=percent_correct,
        device=device
    )

    # Model configuration
    model_config = Config(
        model_type='mlp',
        d=problem.d,
        n_val=1000,
        n=128,
        batch_size=16,
        lr=1e-3,
        epochs=300,
        weight_decay=0.001,
        num_layers=2,
        h=None,
        is_weight_tracker=False,
        d_model=20,
        down_rank_dim=5,
        is_norm=False
    )

    # Generate validation set with class-balanced sampling
    x_val, y_val, center_indices = problem.generate_dataset(
        model_config.n_val, 
        shuffle=True, 
        clean_mode=True
    )
    validation_set = x_val.to(device), y_val.to(device), center_indices.to(device)

    # Train the model using single_runner
    train_once(device, problem, validation_set, model_config, clean_mode=False)


if __name__ == "__main__":
    main()
