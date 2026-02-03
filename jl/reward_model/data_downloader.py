import logging
import os
from datasets import load_from_disk
from transformers import AutoTokenizer
import torch
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

# SageMaker mounts S3 input data here
SAGEMAKER_DATA_PATH = "/opt/ml/input/data/training"


def download_data(c):
    # Initialize tokenizer - needed for padding in collate_fn
    tokenizer = AutoTokenizer.from_pretrained(
        c.model_name, 
        cache_dir=c.cache_dir,
        trust_remote_code=False
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # ensure pad_token is defined

    # Load pre-staged tokenized data (must be uploaded to S3 first via prep_data.py)
    if not os.path.exists(SAGEMAKER_DATA_PATH):
        raise FileNotFoundError(
            f"Pre-staged training data not found at {SAGEMAKER_DATA_PATH}. "
            f"Run 'python -m jl.reward_model.prep_data' to prepare and upload data to S3."
        )
    
    logger.info(f"Loading pre-staged data from {SAGEMAKER_DATA_PATH}")
    dataset = load_from_disk(SAGEMAKER_DATA_PATH)
    train_data = dataset["train"]
    val_data = dataset.get("test", dataset.get("validation", None))

    # Create PyTorch DataLoaders with padding in collate function
    from torch.utils.data import DataLoader

    def collate_fn(batch):
        # Batch is a list of examples (dicts) with keys: chosen_ids, chosen_attention_mask, rejected_ids, rejected_attention_mask
        # Pad all sequences to max_length for consistent tensor shapes
        chosen_list = [torch.tensor(ex["chosen_ids"], dtype=torch.long) for ex in batch]
        reject_list = [torch.tensor(ex["rejected_ids"], dtype=torch.long) for ex in batch]
        
        # Pad all sequences to the configured max_length
        def pad_to_length(tensor, target_len, pad_value):
            if len(tensor) >= target_len:
                return tensor[:target_len]
            else:
                padding = torch.full((target_len - len(tensor),), pad_value, dtype=tensor.dtype)
                return torch.cat([tensor, padding])
        
        chosen_padded = torch.stack([pad_to_length(seq, c.max_length, tokenizer.pad_token_id) for seq in chosen_list])
        reject_padded = torch.stack([pad_to_length(seq, c.max_length, tokenizer.pad_token_id) for seq in reject_list])
        
        # Create attention masks (1 for real tokens, 0 for padding)
        chosen_mask = torch.stack([pad_to_length(torch.tensor(ex["chosen_attention_mask"], dtype=torch.long), c.max_length, 0) for ex in batch])
        reject_mask = torch.stack([pad_to_length(torch.tensor(ex["rejected_attention_mask"], dtype=torch.long), c.max_length, 0) for ex in batch])
        
        # Stack chosen and rejected together into one batch dimension of size 2*batch_size for efficient forward pass
        input_ids = torch.cat([chosen_padded, reject_padded], dim=0)
        attention_mask = torch.cat([chosen_mask, reject_mask], dim=0)
        return input_ids, attention_mask

    train_loader = DataLoader(train_data, batch_size=c.train_batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_data, batch_size=c.eval_batch_size, shuffle=False, collate_fn=collate_fn) if val_data else None

    logger.info(f"Loaded dataset: {len(train_data)} train, {len(val_data) if val_data else 0} val examples")
    return train_loader, val_loader
