"""Load tokenized training data and create PyTorch DataLoaders."""

import logging
import os

from datasets import load_from_disk
from transformers import AutoTokenizer
import torch
from torch.utils.data import DataLoader

from jl.reward_model.prep_data import prepare_dataset

logger = logging.getLogger(__name__)


def load_data(config, train_path: str):
    """Load tokenized data and create DataLoaders.

    If the data doesn't exist at train_path, it will be prepared automatically
    using prep_data.prepare_dataset().

    Args:
        config: RewardConfig with model_name, max_length, batch sizes
        train_path: Path to the tokenized dataset directory

    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Initialize tokenizer - needed for padding in collate_fn
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        trust_remote_code=False
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Check if data exists, if not prepare it
    if not os.path.exists(train_path):
        logger.warning(
            f"Training data not found at {train_path}. "
            "Preparing dataset locally (this is slower than using pre-staged S3 data via launch_sagemaker.py)."
        )
        prepare_dataset(config, train_path)

    logger.info(f"Loading tokenized data from {train_path}")
    dataset = load_from_disk(train_path)
    train_data = dataset["train"]
    val_data = dataset.get("test", dataset.get("validation", None))

    def collate_fn(batch):
        """Collate function that pads sequences and combines chosen/rejected pairs."""
        chosen_list = [torch.tensor(ex["chosen_ids"], dtype=torch.long) for ex in batch]
        reject_list = [torch.tensor(ex["rejected_ids"], dtype=torch.long) for ex in batch]

        def pad_to_length(tensor, target_len, pad_value):
            if len(tensor) >= target_len:
                return tensor[:target_len]
            else:
                padding = torch.full((target_len - len(tensor),), pad_value, dtype=tensor.dtype)
                return torch.cat([tensor, padding])

        chosen_padded = torch.stack([pad_to_length(seq, config.max_length, tokenizer.pad_token_id) for seq in chosen_list])
        reject_padded = torch.stack([pad_to_length(seq, config.max_length, tokenizer.pad_token_id) for seq in reject_list])

        chosen_mask = torch.stack([pad_to_length(torch.tensor(ex["chosen_attention_mask"], dtype=torch.long), config.max_length, 0) for ex in batch])
        reject_mask = torch.stack([pad_to_length(torch.tensor(ex["rejected_attention_mask"], dtype=torch.long), config.max_length, 0) for ex in batch])

        # Stack chosen and rejected together: [chosen_0..chosen_n, rejected_0..rejected_n]
        input_ids = torch.cat([chosen_padded, reject_padded], dim=0)
        attention_mask = torch.cat([chosen_mask, reject_mask], dim=0)
        return input_ids, attention_mask

    train_loader = DataLoader(
        train_data,
        batch_size=config.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    # Validation loader
    val_loader = DataLoader(val_data, batch_size=config.eval_batch_size, shuffle=False, collate_fn=collate_fn) if val_data else None

    logger.info(f"Loaded dataset: {len(train_data)} train, {len(val_data) if val_data else 0} val examples")

    return train_loader, val_loader
