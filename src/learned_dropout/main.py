import torch

from src.learned_dropout.config import Config
from src.learned_dropout.empirical_resnet import run_list_resnet_experiment
from src.learned_dropout.sense_check import train_once
from src.learned_dropout.data_generator import HyperXorNormal, Gaussian, SubDirections


def main():
    torch.manual_seed(38173)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    percent_correct = 0.8
    
    # Problem: HyperXorNormal with requested parameters
    true_d = 3
    noisy_d = 17
    #d = true_d + noisy_d  # Total dimensionality
    # problem = HyperXorNormal(
    #     true_d=true_d,
    #     noisy_d=noisy_d,
    #     random_basis=True,  # Apply random orthonormal transformation
    #     device=device,
    # )
    problem = SubDirections(
        true_d=12,
        sub_d=4,
        perms=24,
        num_class=2,
        sigma=0.2,
        noisy_d=0,
        random_basis=True,
        device=device
    )   

    # d = 10
    # problem = Gaussian(d=d, perfect_class_balance=True)
    
    # Experiment parameters
    h_range = list(range(4, 21, 4))
    num_runs = 20
    h_max = max(h_range)
    
    # Base configuration parameters
    c = Config(
        d=problem.d,
        n_val=1000,
        n=1280,
        batch_size=128,
        layer_norm="rms_norm",
        lr=1e-3,
        epochs=300,
        weight_decay=0.001,
        h=h_max,
        num_layers=2,
        d_model=20,
        l1_final=None,
        is_weight_tracker=False,
        down_rank_dim=None
    )
    
    # Generate validation set with class-balanced sampling
    x_val, y_val, center_indices = problem.generate_dataset(
        c.n_val, 
        shuffle=True, 
        percent_correct=1.0
    )
    validation_set = x_val.to(device), y_val.to(device), center_indices.to(device)
    

    
    #h_list = [20, 20, 20]
    # run_d_model_experiment(device, problem, validation_set, c, h_list)
    
    # Run ResNet experiments
    run_list_resnet_experiment(device, problem, validation_set, [c], h_range, num_runs, percent_correct)


if __name__ == "__main__":
    main()
