import torch

from jl.variance_experiments.data_generator import TwoDirections
from jl.config import Config
from jl.single_runner import train_once


def main():
    # torch.manual_seed(38173)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Problem: TwoDirections
    clean_mode = False
    problem = TwoDirections(
        true_d=10,
        noisy_d=20,
        percent_correct=0.8,
        sigma=0.02,
        random_basis=True,
        device=device
    )

    # Model configuration from main_two_directions_mlp
    model_config = Config(
        model_type='mlp',
        d=problem.d,
        n_val=1000,
        n=128,
        batch_size=16,
        lr= 1*1e-3,
        epochs=300,
        weight_decay=0.001,
        num_layers=1,
        num_class=problem.num_classes(),
        h=50,
        weight_tracker=None,
        is_norm=True
    )

    # Generate validation set with class-balanced sampling
    x_val, y_val, center_indices, _ = problem.generate_dataset(
        model_config.n_val, 
        shuffle=True, 
        clean_mode=True
    )
    validation_set = x_val.to(device), y_val.to(device), center_indices.to(device)

    # Train the model using single_runner
    _, _, _, _, _, _ = train_once(device, problem, validation_set, model_config, clean_mode=clean_mode)


if __name__ == "__main__":
    main()

