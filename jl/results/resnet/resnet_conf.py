from typing import Optional

import torch

from jl.config import Config
from jl.multi_experiment_grapher import run_list_experiment, GraphConfig
from jl.feature_experiments.feature_combinations import FeatureCombinations
from jl.single_runner import train_once


def _get_problem(device: torch.device) -> FeatureCombinations:
    return FeatureCombinations(
        num_layers=4,
        random_basis=True,
        has_favourites=True,
        device=device,
    )


def _get_config(problem: FeatureCombinations, h: int, width_varyer: Optional[str]) -> Config:
    n = 300
    return Config(
        model_type='resnet',
        d=problem.d,
        n_val=1000,
        n=n,
        batch_size=n // 4,
        lr=0.01,
        epochs=100,
        weight_decay=0.001,
        num_layers=3,
        num_class=problem.num_classes(),
        h=h,
        d_model=8,
        weight_tracker="accuracy",
        width_varyer=width_varyer,
        optimizer="adam_w",
        is_norm=False,
        unique_training_set=True,
    )


def _get_validation_set(problem: FeatureCombinations, model_config: Config, device: torch.device):
    x_val, y_val, center_indices_list = problem.generate_dataset(
        model_config.n_val,
        shuffle=True,
        clean_mode=False
    )
    center_indices_on_device = [ci.to(device) for ci in center_indices_list]
    return (x_val.to(device), y_val.to(device), center_indices_on_device)


def run_experiment(width_range: list[int], num_runs: int, graph_config: Optional[GraphConfig] = None):
    torch.manual_seed(38175)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    problem = _get_problem(device)
    model_config = _get_config(problem, h=max(width_range), width_varyer="h")
    validation_set = _get_validation_set(problem, model_config, device)

    # Run experiments
    run_list_experiment(
        device,
        problem,
        validation_set,
        [model_config],
        width_range,
        num_runs,
        clean_mode=False,
        graph_config=graph_config,
    )


def run_single(h: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    problem = _get_problem(device)
    model_config = _get_config(problem, h=h, width_varyer=None)
    validation_set = _get_validation_set(problem, model_config, device)

    # Train (train_once prints final metrics)
    model, tracker, x_train, y_train, train_center_indices = train_once(
        device,
        problem,
        validation_set,
        model_config,
    )
    return model, tracker, x_train, y_train, train_center_indices
