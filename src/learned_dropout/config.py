from typing import Optional
from abc import ABC


class Config(ABC):
    """Abstract base configuration parameters for model experiments"""
    
    def __init__(self, d: int, n_val: int, n: int, batch_size: int, 
                 layer_norm: str, lr: float, epochs: int, weight_decay: float,
                 d_model: Optional[int] = None, l1_final: Optional[float] = None):
        self.d = d
        self.n_val = n_val
        self.n = n
        self.batch_size = batch_size
        self.layer_norm = layer_norm
        self.lr = lr
        self.epochs = epochs
        self.weight_decay = weight_decay
        self.d_model = d_model
        self.l1_final = l1_final


class EmpiricalConfig(Config):
    """Configuration parameters for Running Empirical Experiments regarding the variance of models"""
    
    def __init__(self, h_range: list[int], num_runs: int, **kwargs):
        super().__init__(**kwargs)
        self.h_range = h_range
        self.num_runs = num_runs


class ModelConfig(Config):
    """Configuration parameters for model training and architecture"""
    
    def __init__(self, hidden_sizes: list[int], is_weight_tracker: bool = False,
                 down_rank_dim: Optional[int] = None, **kwargs):
        super().__init__(**kwargs)
        self.hidden_sizes = hidden_sizes
        self.is_weight_tracker = is_weight_tracker
        self.down_rank_dim = down_rank_dim
