import copy
from dataclasses import dataclass


@dataclass
class HyperParameters:
    batch_size: int
    epochs: int
    learning_rate: float
    momentum: float
    weight_decay: float
    gradient: str = 'xe'
    print_epoch: bool = False
    print_batch: bool = False


def with_different_gradients(hp: HyperParameters):
    gradients = ['xe', 'xe-single','purity-scaled']  #
    learning_rates = [0.02, 0.05, 0.05]  #
    hps = [copy.deepcopy(hp) for _ in gradients]
    for new_hp, gradient, learning_rate in zip(hps, gradients, learning_rates):
        new_hp.learning_rate = learning_rate
        new_hp.gradient = gradient
    return hps