from typing import Optional


class Config:
    """Configuration parameters for model experiments"""
    
    def __init__(self, d: int, n_val: int, n: int, batch_size: int, 
                 layer_norm: str, lr: float, epochs: int, weight_decay: float,
                 h: int, num_layers: int,
                 d_model: Optional[int] = None, l1_final: Optional[float] = None,
                 is_weight_tracker: bool = False, down_rank_dim: Optional[int] = None):
        self.d = d
        self.n_val = n_val
        self.n = n
        self.batch_size = batch_size
        self.layer_norm = layer_norm
        self.lr = lr
        self.epochs = epochs
        self.weight_decay = weight_decay
        self.h = h
        self.num_layers = num_layers
        self.d_model = d_model
        self.l1_final = l1_final
        self.is_weight_tracker = is_weight_tracker
        self.down_rank_dim = down_rank_dim
