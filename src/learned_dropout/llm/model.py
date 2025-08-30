from transformers import GPT2Config, GPT2LMHeadModel

# Define preset model size configurations
MODEL_SIZES = {
    "small":  {"n_layer": 12, "n_head": 12, "n_embd": 768},    # ~117M params:contentReference[oaicite:12]{index=12}
    "medium": {"n_layer": 24, "n_head": 16, "n_embd": 1024},   # ~345M params:contentReference[oaicite:13]{index=13}
    "large":  {"n_layer": 36, "n_head": 20, "n_embd": 1280},   # ~774M params:contentReference[oaicite:14]{index=14}
    "xl":     {"n_layer": 48, "n_head": 25, "n_embd": 1600},   # ~1.5B params:contentReference[oaicite:15]{index=15}
    "mini":   {"n_layer": 6,  "n_head": 8,  "n_embd": 512},    # ~50M params (custom smaller model)
    "tiny":   {"n_layer": 4,  "n_head": 4,  "n_embd": 256},    # ~15M params (custom very small model)
}

def create_model(model_size="small", vocab_size=50257, seq_len=64):
    """
    Create a GPT-2 style model with the given size configuration.
    """
    if model_size not in MODEL_SIZES:
        raise ValueError(f"Unknown model size '{model_size}'. Available: {list(MODEL_SIZES.keys())}")
    config_kwargs = MODEL_SIZES[model_size].copy()
    config_kwargs.update({
        "vocab_size": vocab_size,
        "n_positions": seq_len,   # context length
        "n_ctx": seq_len,         # (for GPT2Config, n_ctx is same as n_positions)
        "bos_token_id": 50256,    # GPT2 BOS/EOS token id
        "eos_token_id": 50256,
    })
    config = GPT2Config(**config_kwargs)
    model = GPT2LMHeadModel(config)
    return model
