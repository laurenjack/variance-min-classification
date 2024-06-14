from typing import List
import copy
from dataclasses import dataclass


@dataclass
class HyperParameters:
    batch_size: int
    epochs: int
    learning_rate: float
    momentum: float
    weight_decay: float
    post_constant: float
    gamma: float = 1.0
    is_adam: bool = False
    all_linear: bool = False
    is_bias = False
    print_epoch: bool = False
    print_batch: bool = False


def with_different_weight_decay(hp: HyperParameters, weight_decays: List[float]):
    hps = [copy.deepcopy(hp) for _ in weight_decays]
    for hp, wd in zip(hps, weight_decays):
        hp.weight_decay = wd
    return hps



