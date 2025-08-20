import torch

from src.learned_dropout.config import Config, EmpiricalConfig, ModelConfig
from src.learned_dropout.empirical_resnet import run_down_rank_experiment, run_d_model_experiment, three_resnet_experiment
from src.learned_dropout.sense_check import train_once
from src.learned_dropout.data_generator import HyperXorNormal, Gaussian


def main():
    torch.manual_seed(38173)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Problem: HyperXorNormal with requested parameters
    # percent_correct = 0.8
    # true_d = 3
    # noisy_d = 7
    # d = true_d + noisy_d  # Total dimensionality
    
    # problem = HyperXorNormal(
    #     true_d=true_d,
    #     noisy_d=noisy_d,
    #     random_basis=True,  # Apply random orthonormal transformation
    #     device=device,
    # )

    d = 10
    problem = Gaussian(d=d, perfect_class_balance=True)
    
    # Configuration parameters using EmpiricalConfig
    c = EmpiricalConfig(
        d=d,
        n_val=1000,
        n=512,
        batch_size=128,
        layer_norm="rms_norm",
        lr=1e-3,
        epochs=300,
        weight_decay=0.001,
        h_range = list(range(2, 21, 2)),
        #h_range=list(range(1, 11, 1)),  # d_model values from 1 to 21 in increments of 2
        num_runs=20,
        # d_model=20,
        l1_final=None
    )
    
    # Generate validation set with class-balanced sampling
    x_val, y_val, _, _ = problem.generate_dataset(
        c.n_val, 
        shuffle=True, 
        percent_correct=0.8
    )
    validation_set = x_val.to(device), y_val.to(device)
    
    # Use hidden_sizes from original xor.py
    # h_list = [10, 10]
    # run_d_model_experiment(device, problem, validation_set, c, h_list)
    three_resnet_experiment(device, problem, validation_set, c)
    #run_down_rank_experiment(device, problem, validation_set, c, h_list)


if __name__ == "__main__":
    main()
