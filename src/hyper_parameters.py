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
    purity_components: str = 'lagging'
    purity_threshold: float = 0.5
    is_bias = False
    print_epoch: bool = False
    print_batch: bool = False
    single_calculator: str = 'auto'


def with_different_gradients(hp: HyperParameters):
    gradients = ['xe', 'xe-single','purity-scaled']  #
    learning_rates = [0.02, 0.05, 0.05]  #
    hps = [copy.deepcopy(hp) for _ in gradients]
    for new_hp, gradient, learning_rate in zip(hps, gradients, learning_rates):
        new_hp.learning_rate = learning_rate
        new_hp.gradient = gradient
    return hps

def with_different_purity_components(hp: HyperParameters):
    purity_components = ['leading', 'lagging']
    hps = []
    for pc in purity_components:
        new_hp = copy.deepcopy(hp)
        new_hp.purity_components = pc
        hps.append(new_hp)
    return hps


def with_different_single_calculators(hp: HyperParameters):
    hps = [copy.deepcopy(hp) for _ in range(2)]
    single_calculators = ['auto', 'manual']
    for new_hp, single_calculator in zip(hps, single_calculators):
        new_hp.learning_rate = 0.05
        new_hp.gradient = 'xe-single'
        new_hp.single_calculator = single_calculator
    return hps



