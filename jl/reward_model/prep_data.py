"""Data preparation for reward model training."""

import logging
from pathlib import Path

from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer

from jl.reward_model.reward_config import RewardConfig

logger = logging.getLogger(__name__)


def prepare_dataset(config: RewardConfig, output_path: str) -> None:
    """Download dataset from HuggingFace, tokenize it, and save to disk.
    
    Args:
        config: RewardConfig with model_name, hf_dataset, and max_length
        output_path: Local directory path to save the tokenized dataset
    
    Example:
        from jl.reward_model.prep_data import prepare_dataset
        from jl.reward_model.reward_config import RewardConfig
        
        config = RewardConfig()
        prepare_dataset(config, "./data/tokenized")
    """
    output_dir = Path(output_path)
    
    logger.info(f"Preparing dataset: {config.hf_dataset}")
    logger.info(f"Tokenizer model: {config.model_name}")
    logger.info(f"Max length: {config.max_length}")
    logger.info(f"Output path: {output_dir}")

    # Download dataset
    logger.info("Downloading dataset from HuggingFace...")
    dataset = load_dataset(config.hf_dataset, name=None)
    train_data = dataset["train"]
    val_data = dataset.get("test", dataset.get("validation", None))
    logger.info(f"Downloaded: {len(train_data)} train, {len(val_data) if val_data else 0} val examples")

    # Initialize tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Tokenize dataset
    logger.info("Tokenizing dataset...")
    
    def tokenize_pair(example):
        chosen = example["chosen"]
        rejected = example["rejected"]
        chosen_text = chosen + (tokenizer.eos_token or "")
        rejected_text = rejected + (tokenizer.eos_token or "")
        chosen_enc = tokenizer(chosen_text, max_length=config.max_length, padding=False, truncation=True)
        rejected_enc = tokenizer(rejected_text, max_length=config.max_length, padding=False, truncation=True)
        return {
            "chosen_ids": chosen_enc["input_ids"],
            "chosen_attention_mask": chosen_enc["attention_mask"],
            "rejected_ids": rejected_enc["input_ids"],
            "rejected_attention_mask": rejected_enc["attention_mask"],
        }

    train_tokenized = train_data.map(
        tokenize_pair, 
        batched=False, 
        remove_columns=train_data.column_names,
        desc="Tokenizing train"
    )
    
    if val_data:
        val_tokenized = val_data.map(
            tokenize_pair, 
            batched=False, 
            remove_columns=val_data.column_names,
            desc="Tokenizing val"
        )
        tokenized_dataset = DatasetDict({"train": train_tokenized, "test": val_tokenized})
    else:
        tokenized_dataset = DatasetDict({"train": train_tokenized})

    # Save to disk
    logger.info(f"Saving tokenized dataset to {output_dir}...")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    tokenized_dataset.save_to_disk(str(output_dir))
    
    logger.info("Data preparation complete")
