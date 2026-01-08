from typing import Optional


class Config:
    """Configuration parameters for model experiments"""
    
    def __init__(self, model_type: str, d: int, n_val: int, n: int, batch_size: int,
                 lr: float, epochs: int, weight_decay: float,
                 num_layers: int,
                 num_class: int,
                 h: Optional[int] = None,
                 d_model: Optional[int] = None,
                 weight_tracker: Optional[str] = "accuracy", down_rank_dim: Optional[int] = None,
                 width_varyer: Optional[str] = None, is_norm: bool = True, c: Optional[float] = None,
                 k: Optional[int] = None, adam_eps: float = 1e-8, optimizer: str = "adam_w",
                 learnable_norm_parameters: bool = True, adam_betas: tuple = (0.9, 0.999),
                 sgd_momentum: float = 0.0, lr_scheduler: Optional[str] = None):
        # Validate model_type
        if model_type not in ['resnet', 'mlp', 'k-polynomial', 'multi-linear']:
            raise ValueError(f"model_type must be either 'resnet', 'mlp', 'k-polynomial', or 'multi-linear', got '{model_type}'")
        
        # Validate num_class
        if num_class < 2:
            raise ValueError(f"num_class must be >= 2, got {num_class}")
        
        # Validate h parameter based on model_type
        if model_type == 'mlp' and d_model is not None:
            raise ValueError("d_model parameter cannot be set for 'mlp' model_type. Use h for hidden dimension size.")
        if model_type == 'multi-linear' and d_model is not None:
            raise ValueError("d_model parameter cannot be set for 'multi-linear' model_type. Use h for hidden dimension size.")
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
        if model_type == 'mlp' and width_varyer == 'd_model':
            raise ValueError("width_varyer cannot be 'd_model' for 'mlp' model_type. Use 'h' instead.")
        if model_type == 'multi-linear' and width_varyer is not None:
            raise ValueError("width_varyer must be None for 'multi-linear' model_type")
        
        # Validate optimizer
        if optimizer not in ['adam_w', 'sgd', 'reg_adam_w']:
            raise ValueError(f"optimizer must be 'adam_w', 'sgd', or 'reg_adam_w', got '{optimizer}'")
        
        # Validate reg_adam_w constraint
        if optimizer == 'reg_adam_w' and learnable_norm_parameters:
            raise ValueError("learnable_norm_parameters must be False when optimizer='reg_adam_w'")
        
        # Validate weight_tracker
        if weight_tracker is not None and weight_tracker not in ['accuracy', 'weight', 'full_step']:
            raise ValueError(f"weight_tracker must be None, 'accuracy', 'weight', or 'full_step', got '{weight_tracker}'")
        
        # Validate adam_betas
        if not isinstance(adam_betas, tuple) or len(adam_betas) != 2:
            raise ValueError(f"adam_betas must be a tuple of length 2, got {adam_betas}")
        if not 0.0 <= adam_betas[0] < 1.0:
            raise ValueError(f"adam_betas[0] must be in [0, 1), got {adam_betas[0]}")
        if not 0.0 <= adam_betas[1] < 1.0:
            raise ValueError(f"adam_betas[1] must be in [0, 1), got {adam_betas[1]}")
        
        # Validate sgd_momentum
        if not isinstance(sgd_momentum, (int, float)) or not 0.0 <= sgd_momentum < 1.0:
            raise ValueError(f"sgd_momentum must be in [0, 1), got {sgd_momentum}")

        # Validate lr_scheduler
        if lr_scheduler is not None and lr_scheduler not in ['sd', 'wsd']:
            raise ValueError(f"lr_scheduler must be None, 'sd', or 'wsd', got '{lr_scheduler}'")

        # Validate scheduler requirements
        if lr_scheduler is not None:
            import math
            training_steps = math.ceil(n / batch_size) * epochs
            if training_steps < 20:
                raise ValueError(f"Learning rate scheduler requires at least 20 training steps, got {training_steps} (calculated as ceil({n}/{batch_size}) * {epochs})")

        self.model_type = model_type
        self.d = d
        self.n_val = n_val
        self.n = n
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.weight_decay = weight_decay
        self.num_class = num_class
        self.h = h
        self.num_layers = num_layers
        self.d_model = d_model
        self.weight_tracker = weight_tracker
        self.down_rank_dim = down_rank_dim
        self.width_varyer = width_varyer
        self.is_norm = is_norm
        self.c = c
        self.k = k
        self.adam_eps = adam_eps
        self.optimizer = optimizer
        self.learnable_norm_parameters = learnable_norm_parameters
        self.adam_betas = adam_betas
        self.sgd_momentum = sgd_momentum
        self.lr_scheduler = lr_scheduler
