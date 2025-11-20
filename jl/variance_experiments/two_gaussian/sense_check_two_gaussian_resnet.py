import torch

from jl.variance_experiments.data_generator import TwoGaussians
from jl.config import Config
from jl.single_runner import train_once


def main():
    # torch.manual_seed(38173)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Problem: TwoGaussians
    clean_mode = False
    problem = TwoGaussians(
        true_d=20,
        noisy_d=0,
        percent_correct=0.8,
        device=device
    )

    # Model configuration from main_two_gaussian_resnet
    model_config = Config(
        model_type='resnet',
        d=problem.d,
        n_val=1000,
        n=128,
        batch_size=16,
        lr=1e-3,
        epochs=300,
        weight_decay=0.001,
        num_layers=1,
        num_class=problem.num_classes(),
        h=50,
        d_model=None,
        is_weight_tracker=False,
        down_rank_dim=None,
        is_norm=True
    )

    # Generate validation set with class-balanced sampling
    x_val, y_val, center_indices = problem.generate_dataset(
        model_config.n_val, 
        shuffle=True, 
        clean_mode=True
    )
    validation_set = x_val.to(device), y_val.to(device), center_indices.to(device)

    # Train the model using single_runner
    train_once(device, problem, validation_set, model_config, clean_mode=clean_mode)


if __name__ == "__main__":
    main()

