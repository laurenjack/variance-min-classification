import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM


def get_model(c, device):
    """Load and prepare the reward model.

    Args:
        c: RewardConfig with model_name
        device: torch.device for the GPU

    Returns:
        Compiled model ready for training
    """
    # Load the base model (will download if not cached). Use bfloat16 for faster training.
    model = AutoModelForCausalLM.from_pretrained(
        c.model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=False,
        attn_implementation="sdpa"
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.to(device)

    # Add a reward head: a single linear layer to output a scalar from the last token's hidden state
    hidden_size = model.config.hidden_size if hasattr(model.config, "hidden_size") else model.config.hidden_dim
    model.reward_head = nn.Linear(hidden_size, 1, device=device, dtype=torch.bfloat16)

    for param in model.parameters():
        param.requires_grad = True

    model = torch.compile(model)

    return model
