import torch

from src.learned_dropout.data_generator import SubDirections
from src.learned_dropout.config import Config
from src.learned_dropout.single_runner import train_once


def main():
    torch.manual_seed(38173)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Problem: SubDirections with parameters from main_mlp
    percent_correct = 1.0
    clean_mode = False
    problem = SubDirections(
        true_d=12,
        sub_d=4,
        centers=24,
        num_class=2,
        sigma=0.02,
        noisy_d=0,
        random_basis=True,
        percent_correct=percent_correct,
        device=device
    )

    # Model configuration from main_mlp
    model_config = Config(
        model_type='mlp',
        d=problem.d,
        n_val=1000,
        n=128,
        batch_size=32,
        lr=1e-3,
        epochs=300,
        weight_decay=0.001,
        num_layers=1,
        h=None,
        is_weight_tracker=False,
        d_model=50,
        down_rank_dim=None,
        is_norm=True
    )

    # Generate validation set with class-balanced sampling
    x_val, y_val, center_indices = problem.generate_dataset(
        model_config.n_val, 
        shuffle=True, 
        clean_mode=False
    )
    validation_set = x_val.to(device), y_val.to(device), center_indices.to(device)

    # Train the model using single_runner
    train_once(device, problem, validation_set, model_config, clean_mode=clean_mode)


if __name__ == "__main__":
    main()

