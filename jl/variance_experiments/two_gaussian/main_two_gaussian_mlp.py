import torch

from jl.variance_experiments.config import Config
from jl.variance_experiments.empirical_runner import run_list_experiment
from jl.variance_experiments.data_generator import TwoGaussians


def main():
    torch.manual_seed(38173)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clean_mode = False
    
    # Problem: TwoGaussians
    problem = TwoGaussians(
        true_d=20,
        noisy_d=0,
        percent_correct=0.8,
        device=device
    )   

    # Experiment parameters for MLP
    # For MLP, d_model represents the hidden dimension size
    width_range = list(range(4, 81, 4))
    d_model = max(width_range)
    num_runs = 20
    
    
    # Base configuration parameters for MLP
    c = Config(
        model_type='mlp',
        d=problem.d,
        n_val=1000,
        n=128,
        batch_size=16,
        lr=1e-3,
        epochs=300,
        weight_decay=0.001,
        num_layers=2,
        d_model=d_model,
        is_weight_tracker=False,
        down_rank_dim=None,
        width_varyer="d_model",
        is_norm=True
    )
    # c2 = deepcopy(c)
    # c2.num_layers = 2
    
    # c3 = deepcopy(c)
    # c3.num_layers = 3
    
    # Generate validation set with class-balanced sampling
    x_val, y_val, center_indices = problem.generate_dataset(
        c.n_val, 
        shuffle=True, 
        clean_mode=True
    )
    validation_set = x_val.to(device), y_val.to(device), center_indices.to(device)
    

    
    # Run MLP experiments
    run_list_experiment(
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

