import torch

from src.learned_dropout.data_generator import SubDirections
from src.learned_dropout.config import Config
from src.learned_dropout.sense_check import train_once


def main():
    torch.manual_seed(4454)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Problem: SubDirections with requested parameters
    percent_correct = 1.0
    use_percent_correct = False
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

    # Model configuration
    model_config = Config(
        d=problem.d,
        n_val=1000,
        n=512,
        batch_size=64,
        lr=1e-3,
        epochs=300,
        weight_decay=0.001,
        h=40,
        num_layers=2,
        is_weight_tracker=False,
        l1_final=None,
        d_model=20,
        down_rank_dim=5
    )

    # Generate validation set with class-balanced sampling
    x_val, y_val, center_indices = problem.generate_dataset(
        model_config.n_val, 
        shuffle=True, 
        use_percent_correct=use_percent_correct
    )
    validation_set = x_val.to(device), y_val.to(device), center_indices.to(device)

    # Train the model using sense_check
    train_once(device, problem, validation_set, model_config, use_percent_correct=use_percent_correct)


if __name__ == "__main__":
    main()
