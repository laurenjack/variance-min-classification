import torch

from src.learned_dropout.data_generator import HyperXorNormal
from src.learned_dropout.config import Config
from src.learned_dropout.sense_check import train_once


def main():
    torch.manual_seed(38173)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Problem: HyperXorNormal with requested parameters
    percent_correct = 0.8
    true_d = 3
    noisy_d = 7
    d = true_d + noisy_d  # Total dimensionality
    
    problem = HyperXorNormal(
        true_d=true_d,
        noisy_d=noisy_d,
        random_basis=True,  # Apply random orthonormal transformation
        device=device,
    )

    model_config = Config(
        d=d,
        n_val=1000,
        n=512,
        batch_size=128,
        layer_norm="rms_norm",
        lr=1e-3,
        epochs=300,
        weight_decay=0.001,
        h=20,
        num_layers=3,
        is_weight_tracker=False,
        l1_final=None,
        d_model=5
    )

    # Generate validation set with class-balanced sampling
    x_val, y_val, _ = problem.generate_dataset(
        model_config.n_val, 
        shuffle=True, 
        percent_correct=0.8
    )

    validation_set = x_val.to(device), y_val.to(device)

    # Train the model using sense_check
    train_once(device, problem, validation_set, model_config, percent_correct=percent_correct)


if __name__ == "__main__":
    main()
