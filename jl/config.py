from typing import Optional


class Config:
    """Configuration parameters for model experiments"""
    
    def __init__(self, model_type: str, d: int, n_val: int, n: int, batch_size: int, 
                 lr: float, epochs: int, weight_decay: float,
                 num_layers: int,
                 h: Optional[int] = None,
                 d_model: Optional[int] = None,
                 is_weight_tracker: bool = False, down_rank_dim: Optional[int] = None,
                 width_varyer: Optional[str] = None, is_norm: bool = True, c: Optional[float] = None,
                 k: Optional[int] = None, adam_eps: float = 1e-8):
        # Validate model_type
        if model_type not in ['resnet', 'mlp', 'k-polynomial']:
            raise ValueError(f"model_type must be either 'resnet', 'mlp', or 'k-polynomial', got '{model_type}'")
        
        # Validate h parameter based on model_type
        if model_type == 'mlp' and h is not None:
            raise ValueError("h parameter cannot be set for 'mlp' model_type. Use d_model for hidden dimension size.")
        if model_type == 'k-polynomial' and h is not None:
            raise ValueError("h parameter cannot be set for 'k-polynomial' model_type.")
        if model_type == 'resnet' and h is None:
            raise ValueError("h parameter is required for 'resnet' model_type")
        
        # Validate k parameter based on model_type
        if model_type == 'k-polynomial':
            if k is None:
                raise ValueError("k parameter is required for 'k-polynomial' model_type")
            if k <= 0:
                raise ValueError("k parameter must be > 0 for 'k-polynomial' model_type")
        else:
            if k is not None:
                raise ValueError(f"k parameter can only be set for 'k-polynomial' model_type, not '{model_type}'")
        
        # Validate width_varyer based on model_type
        if model_type == 'mlp' and width_varyer == 'h':
            raise ValueError("width_varyer cannot be 'h' for 'mlp' model_type")
        
        self.model_type = model_type
        self.d = d
        self.n_val = n_val
        self.n = n
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.weight_decay = weight_decay
        self.h = h
        self.num_layers = num_layers
        self.d_model = d_model
        self.is_weight_tracker = is_weight_tracker
        self.down_rank_dim = down_rank_dim
        self.width_varyer = width_varyer
        self.is_norm = is_norm
        self.c = c
        self.k = k
        self.adam_eps = adam_eps
