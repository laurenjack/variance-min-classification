"""Load RETFound checkpoint and build fine-tuning model.

Based on the RETFound_MAE fine-tuning recipe:
- ViT-Large/16 with optional global average pooling
- Pretrained MAE encoder loaded from HuggingFace Hub
- Classifier head reinitialized for target num_classes
"""

import logging
from functools import partial

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download

from jl.double_descent.medical_calibration.config import MedCalConfig

logger = logging.getLogger(__name__)


def build_retfound_model(config: MedCalConfig, device: torch.device) -> nn.Module:
    """Build RETFound ViT-Large model with pretrained weights.

    Downloads checkpoint from HuggingFace Hub, loads encoder weights,
    and reinitializes the classifier head.

    Args:
        config: Experiment config.
        device: Device to load model onto.

    Returns:
        Model ready for fine-tuning.
    """
    import timm

    # Create ViT-Large model via timm
    model = timm.create_model(
        "vit_large_patch16_224",
        num_classes=config.num_classes,
        drop_path_rate=config.drop_path,
        global_pool="avg" if config.global_pool else "token",
    )

    # Download pretrained checkpoint
    logger.info(f"Downloading checkpoint from {config.model_repo}...")
    checkpoint_path = hf_hub_download(
        repo_id=config.model_repo,
        filename=config.model_filename,
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_model = checkpoint["model"]

    # Remove classifier head weights (shape mismatch with our num_classes)
    state_dict = model.state_dict()
    for k in ["head.weight", "head.bias"]:
        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
            logger.info(f"Removing key {k} from checkpoint (shape mismatch)")
            del checkpoint_model[k]

    # Also handle fc_norm if present (for global_pool models)
    for k in list(checkpoint_model.keys()):
        if k.startswith("fc_norm") and k not in state_dict:
            del checkpoint_model[k]

    # Load weights
    msg = model.load_state_dict(checkpoint_model, strict=False)
    logger.info(f"Loaded checkpoint: missing={msg.missing_keys}")

    # Reinitialize classifier head
    nn.init.trunc_normal_(model.head.weight, std=2e-5)
    if model.head.bias is not None:
        nn.init.zeros_(model.head.bias)

    model = model.to(device)
    logger.info(
        f"Model ready: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params"
    )

    return model


def build_layer_decay_param_groups(
    model: nn.Module, config: MedCalConfig, lr: float
) -> list:
    """Build parameter groups with layer-wise learning rate decay.

    Deeper layers get higher learning rates, earlier layers get lower.
    This follows the RETFound fine-tuning recipe.

    Args:
        model: ViT model.
        config: Config with layer_decay and weight_decay.
        lr: Peak learning rate.

    Returns:
        List of param group dicts for the optimizer.
    """
    # Get number of layers
    num_layers = len(model.blocks) + 1  # +1 for patch embed

    layer_scales = {}
    for i in range(num_layers):
        layer_scales[i] = config.layer_decay ** (num_layers - i - 1)

    param_groups = []
    param_group_names = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Determine which layer this parameter belongs to
        if name.startswith("patch_embed") or name.startswith("cls_token") or name.startswith("pos_embed"):
            layer_id = 0
        elif name.startswith("blocks."):
            layer_id = int(name.split(".")[1]) + 1
        else:
            # head, norm, fc_norm
            layer_id = num_layers - 1

        # No weight decay on bias, norm, cls_token, pos_embed
        no_decay = (
            param.ndim == 1
            or name.endswith(".bias")
            or name in ("cls_token", "pos_embed")
        )

        group_name = f"layer_{layer_id}_{'no_decay' if no_decay else 'decay'}"

        if group_name not in param_group_names:
            scale = layer_scales[layer_id]
            param_groups.append({
                "params": [],
                "lr": lr * scale,
                "weight_decay": 0.0 if no_decay else config.weight_decay,
                "group_name": group_name,
            })
            param_group_names[group_name] = len(param_groups) - 1

        idx = param_group_names[group_name]
        param_groups[idx]["params"].append(param)

    return param_groups
