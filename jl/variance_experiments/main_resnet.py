from copy import deepcopy
import torch

from jl.config import Config
from jl.multi_experiment_grapher import run_list_experiment_with_variance
from jl.variance_experiments.data_generator import SubDirections, Gaussian


def main():
    torch.manual_seed(38173)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    percent_correct = 0.8
    clean_mode = False
    
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
        centers=6,
        num_class=2,
        sigma=0.2,
        noisy_d=10,
        random_basis=True,
        percent_correct=percent_correct,
        device=device
    )   

    # d = 10
    # problem = Gaussian(d=20, perfect_class_balance=True)
    
    # Experiment parameters
    # h_range = list(range(2, 31, 2))
    # h = max(h_range)
    # d_model = 20
    # h = 160
    # d_model = 20
    width_range = list(range(2, 41, 2))
    h = max(width_range)
    d_model = 10
    # d_model = max(width_range)
    #width_range = list(range(2, 41, 2))
    # d_model = max(width_range)
    num_runs = 20
    
    
    # Base configuration parameters
    c = Config(
        model_type='resnet',
        d=problem.d,
        n_val=1000,
        n=256,
        adam_eps=1e-8,
        batch_size=64,
        lr=3e-3,
        epochs=100,
        weight_decay=0.001,
        num_layers=2,
        num_class=problem.num_classes(),
        h=h,
        d_model=d_model,
        weight_tracker=None,
        width_varyer="h",
        is_norm=True
    )
    c2 = deepcopy(c)
    c2.num_layer = 3
    c3 = deepcopy(c)
    c3.num_layer = 4
    # Generate validation set with class-balanced sampling
    x_val, y_val, center_indices = problem.generate_dataset(
        c.n_val,
        shuffle=True,
        clean_mode=True
    )
    validation_set = x_val.to(device), y_val.to(device), center_indices.to(device)
    

    
    #h_list = [20, 20, 20]
    # run_d_model_experiment(device, problem, validation_set, c, h_list)
    
    # Run Resnet experiments
    run_list_experiment_with_variance(
        device,
        problem,
        validation_set,
        [c],
        width_range,
        num_runs,
        clean_mode,
    )


if __name__ == "__main__":
    main()
