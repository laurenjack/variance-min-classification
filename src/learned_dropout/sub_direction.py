import torch

from src.learned_dropout.data_generator import SubDirections
from src.learned_dropout.config import Config, ModelConfig
from src.learned_dropout.sense_check import train_once


def main():
    torch.manual_seed(38173)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Problem: SubDirections with requested parameters
    percent_correct = 0.8
    d = 12
    problem = SubDirections(
        d=d,
        sub_d=4,
        perms=24,
        num_class=2,
        sigma=0.05,
    )

    # Model configuration
    model_config = ModelConfig(
        d=d,
        n_val=1000,
        n=240,
        batch_size=240,
        layer_norm="rms_norm",
        lr=1e-3,
        epochs=400,
        weight_decay=0.001,
        hidden_sizes=[100],
        is_weight_tracker=False,
        l1_final=None,
        d_model=10
        # down_rank_dim=12
    )

    # Generate validation set with class-balanced sampling
    x_val, y_val = problem.generate_dataset(model_config.n_val, shuffle=True, percent_correct=percent_correct)

    validation_set = x_val.to(device), y_val.to(device)

    # Train the model using sense_check
    train_once(device, problem, validation_set, model_config, percent_correct=percent_correct)


if __name__ == "__main__":
    main()
