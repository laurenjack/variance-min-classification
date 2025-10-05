import torch

from src.learned_dropout.config import Config
from src.learned_dropout.empirical_resnet import run_list_resnet_experiment
from src.learned_dropout.sense_check import train_once
from src.learned_dropout.data_generator import HyperXorNormal, Gaussian, SubDirections


def main():
    torch.manual_seed(38173)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    percent_correct = 0.8
    use_percent_correct = True
    
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
        centers=24,
        num_class=2,
        sigma=0.2,
        noisy_d=0,
        random_basis=True,
        percent_correct=percent_correct,
        device=device
    )   

    # d = 10
    # problem = Gaussian(d=d, perfect_class_balance=True)
    
    # Experiment parameters
    # h_range = list(range(2, 31, 2))
    # h = max(h_range)
    # d_model = 20
    h = 40
    d_model = 20
    width_range = list(range(2, 21, 2))
    # d_model = max(width_range)
    # width_range = list(range(2, 11, 1))
    down_rank_dim = max(width_range)
    # down_rank_dim = 5
    num_runs = 20
    
    
    # Base configuration parameters
    c = Config(
        d=problem.d,
        n_val=1000,
        n=512,
        batch_size=64,
        lr=1e-3,
        epochs=300,
        weight_decay=0.001,
        h=h,
        num_layers=2,
        d_model=d_model,
        l1_final=None,
        is_weight_tracker=False,
        down_rank_dim=down_rank_dim,
        width_varyer="down_rank_dim"
    )
    
    # Generate validation set with class-balanced sampling
    x_val, y_val, center_indices = problem.generate_dataset(
        c.n_val, 
        shuffle=True, 
        use_percent_correct=False
    )
    validation_set = x_val.to(device), y_val.to(device), center_indices.to(device)
    

    
    #h_list = [20, 20, 20]
    # run_d_model_experiment(device, problem, validation_set, c, h_list)
    
    # Run Resnet experiments
    run_list_resnet_experiment(
        device,
        problem,
        validation_set,
        [c],
        width_range,
        num_runs,
        use_percent_correct,
    )


if __name__ == "__main__":
    main()
