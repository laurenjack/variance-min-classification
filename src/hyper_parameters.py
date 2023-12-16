from dataclasses import dataclass


@dataclass
class HyperParameters:
    batch_size: int
    epochs: int
    learning_rate: float
    momentum: float
    weight_decay: float
    is_bias = False
    print_epoch: bool = False
    print_batch: bool = False



