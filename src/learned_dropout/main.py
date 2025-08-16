import torch

from src import dataset_creator
from src.learned_dropout.config import Config, EmpiricalConfig, ModelConfig
from src.learned_dropout.empirical_resnet import run_down_rank_experiment
from src.learned_dropout.sense_check import train_once
from src.learned_dropout.data_generator import SubDirections


def main():
    # torch.manual_seed(38173)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Problem: SubDirections with same parameters as sub_direction.py
    percent_correct = 0.8
    d = 12
    problem = SubDirections(
        d=d,
        sub_d=4,
        perms=24,
        num_class=2,
        sigma=0.05,
    )
    
    # Configuration parameters for down-rank experiment  
    c = EmpiricalConfig(
        d=d,
        n_val=1000,
        n=100,
        batch_size=100,
        layer_norm="rms_norm",
        lr=3e-3,
        epochs=400,
        weight_decay=0.001,
        h_range=list(range(3, 16, 2)),  # 3, 5, 7, 9, 11, 13, 15 (down_rank_dim values)
        num_runs=20,
        d_model=None,
        l1_final=None
    )
    
    # prepare validation set (same across runs) with percent_correct
    x_val, y_val = problem.generate_dataset(c.n_val, shuffle=True, percent_correct=percent_correct)
    validation_set = x_val.to(device), y_val.to(device)
    h_list = [500]
    run_down_rank_experiment(device, problem, validation_set, c, h_list)


if __name__ == "__main__":
    main()
