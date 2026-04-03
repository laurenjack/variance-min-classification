"""Configuration for RETFound + APTOS-2019 calibration experiment."""

from dataclasses import dataclass


@dataclass
class MedCalConfig:
    # Model
    model_repo: str = "YukunZhou/RETFound_mae_natureCFP"
    model_filename: str = "RETFound_mae_natureCFP.pth"
    num_classes: int = 5
    input_size: int = 224
    global_pool: bool = True
    drop_path: float = 0.2

    # Fine-tuning (paper recipe)
    epochs: int = 50
    batch_size: int = 16
    blr: float = 5e-3  # base learning rate
    layer_decay: float = 0.65
    weight_decay: float = 0.05
    warmup_epochs: int = 5
    label_smoothing: float = 0.1
    mixup: float = 0.0
    cutmix: float = 0.0

    # Augmentation
    reprob: float = 0.25  # random erasing probability
    color_jitter: float = 0.4

    # Data
    split_seed: int = 42
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    num_workers: int = 4

    # Calibration
    l2_lambda: float = 1e-3
    lbfgs_max_steps: int = 100

    # Logging
    log_interval: int = 1  # every epoch
