from typing import List, Optional
import copy
from dataclasses import dataclass


@dataclass
class DataParameters:
    percent_correct: float
    n: int
    n_test: int
    d: int
    true_d: int = 0


@dataclass
class HyperParameters:
    batch_size: int
    epochs: int
    learning_rate: float
    momentum: float
    weight_decay: float
    sizes: List[int]
    desired_success_rate: float
    relu_bound: float = 0.25
    do_train: bool = True
    gamma: float = 1.0
    is_adam: bool = False
    all_linear: bool = False
    is_bias: bool = False
    reg_type: Optional[str] = None
    weight_tracker_type: Optional[str] = None
    implementation: str = 'old'
    reg_epsilon: float = 0.0
    print_epoch: bool = False
    print_batch: bool = False


def with_different_weight_decay(hp: HyperParameters, weight_decays: List[float]):
    hps = [copy.deepcopy(hp) for _ in weight_decays]
    for hp, wd in zip(hps, weight_decays):
        hp.weight_decay = wd
    return hps



