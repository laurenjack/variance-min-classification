import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM


def get_model(c, device):
    # Load the base model (will download if not cached). Use bfloat16 for faster training if available.
    model = AutoModelForCausalLM.from_pretrained(
        c.model_name, 
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=False,
        attn_implementation="sdpa"
    )
    # For training with reduced memory footprint
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.to(device)

    # Add a reward head: a single linear layer to output a scalar from the final hidden state of the last token
    # We'll attach it to the model as an attribute for convenience.
    hidden_size = model.config.hidden_size if hasattr(model.config, "hidden_size") else model.config.hidden_dim
    model.reward_head = nn.Linear(hidden_size, 1, device=device)
    # Ensure the reward head is in bfloat16 as well if model is (to match precision)
    if next(model.parameters()).dtype == torch.bfloat16:
        model.reward_head.to(torch.bfloat16)
        
    # Enable gradient computation for all model parameters (not really needed unless some were frozen by default)
    for param in model.parameters():
        param.requires_grad = True

    # Compile model for faster training (PyTorch 2.x)
    model = torch.compile(model)

    return model
