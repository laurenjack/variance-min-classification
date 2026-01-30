from typing import Optional


class Config:
    """Configuration parameters for model experiments"""
    
    def __init__(self, model_type: str, d: int, n_val: int, n: int, batch_size: int,
                 lr: float, epochs: int, weight_decay: float,
                 num_layers: int,
                 num_class: int,
                 h: Optional[int] = None,
                 d_model: Optional[int] = None,
                 weight_tracker: Optional[str] = "accuracy",
                 width_varyer: Optional[str] = None, is_norm: bool = True, c: Optional[float] = None,
                 k: Optional[int] = None, adam_eps: float = 1e-8, optimizer: str = "adam_w",
                 learnable_norm_parameters: bool = True, adam_betas: tuple = (0.9, 0.999),
                 sgd_momentum: float = 0.0, lr_scheduler: Optional[str] = None,
                dropout_prob: Optional[float] = None,
                is_hashed_dropout: bool = False,
                prob_weight: float = 1.0,
                num_models: Optional[int] = None):
        # Validate model_type
        if model_type not in ['resnet', 'mlp', 'k-polynomial', 'multi-linear', 'simple-mlp']:
            raise ValueError(f"model_type must be 'resnet', 'mlp', 'k-polynomial', 'multi-linear', or 'simple-mlp', got '{model_type}'")
        
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
        if model_type == 'simple-mlp':
            if d_model is not None:
                raise ValueError("d_model parameter cannot be set for 'simple-mlp' model_type")
            if width_varyer is not None and width_varyer != 'h':
                raise ValueError("width_varyer must be None or 'h' for 'simple-mlp' model_type")
            if num_layers > 0 and h is None:
                raise ValueError("h parameter is required for 'simple-mlp' when num_layers > 0")
            if weight_tracker not in [None, 'accuracy']:
                raise ValueError(f"simple-mlp only supports weight_tracker=None or 'accuracy', got '{weight_tracker}'")
        
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
        if optimizer not in ['adam_w', 'sgd']:
            raise ValueError(f"optimizer must be 'adam_w' or 'sgd', got '{optimizer}'")
        
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

        # Validate dropout_prob
        if dropout_prob is not None:
            if not isinstance(dropout_prob, (int, float)) or not 0.0 <= dropout_prob < 1.0:
                raise ValueError(f"dropout_prob must be in [0, 1), got {dropout_prob}")
            if model_type == 'k-polynomial':
                raise ValueError("dropout_prob cannot be set for 'k-polynomial' model_type")

        # Validate is_hashed_dropout
        if is_hashed_dropout:
            if dropout_prob is None:
                raise ValueError("dropout_prob must be set when is_hashed_dropout=True")
            if model_type != 'resnet':
                raise ValueError(f"is_hashed_dropout=True is only supported for model_type='resnet', got '{model_type}'")
            if width_varyer is not None:
                raise ValueError(f"is_hashed_dropout=True is incompatible with width_varyer={width_varyer}")

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
        self.dropout_prob = dropout_prob
        self.is_hashed_dropout = is_hashed_dropout
        self.prob_weight = prob_weight
        self.num_models = num_models
