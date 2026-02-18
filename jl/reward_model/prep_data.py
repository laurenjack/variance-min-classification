"""Data preparation for reward model training.

Downloads a dataset from HuggingFace, normalizes it to chosen/rejected text
columns, tokenizes, and saves to disk. Dataset-specific download and
preprocessing logic lives in the _load_* functions.
"""

import logging
from pathlib import Path

from datasets import load_dataset, DatasetDict, Dataset
from transformers import AutoTokenizer

from jl.reward_model.reward_config import RewardConfig

logger = logging.getLogger(__name__)


def _load_anthropic_hh(config: RewardConfig) -> tuple[Dataset, Dataset | None]:
    """Load Anthropic/hh-rlhf — already has chosen/rejected columns."""
    dataset = load_dataset("Anthropic/hh-rlhf", name=None)
    train_data = dataset["train"]
    val_data = dataset.get("test", dataset.get("validation", None))
    return train_data, val_data


def _load_helpsteer2(config: RewardConfig) -> tuple[Dataset, Dataset | None]:
    """Load nvidia/HelpSteer2 preference annotations and map to chosen/rejected."""
    raw = load_dataset("nvidia/HelpSteer2", data_dir="preference", split="train")
    logger.info(f"Downloaded {len(raw)} total examples")

    train_data = raw.filter(lambda x: x["split"] == "train")
    val_data = raw.filter(lambda x: x["split"] == "val")
    logger.info(f"Split: {len(train_data)} train, {len(val_data)} val (before filtering ties)")

    train_data = train_data.filter(lambda x: x["preference_strength"] != 0)
    val_data = val_data.filter(lambda x: x["preference_strength"] != 0)
    logger.info(f"After dropping ties: {len(train_data)} train, {len(val_data)} val")

    def to_bt_pair(ex):
        if ex["preference_strength"] < 0:
            chosen_response = ex["response_1"]
            rejected_response = ex["response_2"]
        else:
            chosen_response = ex["response_2"]
            rejected_response = ex["response_1"]
        return {
            "chosen": ex["prompt"] + chosen_response,
            "rejected": ex["prompt"] + rejected_response,
        }

    train_data = train_data.map(to_bt_pair, remove_columns=train_data.column_names, desc="Mapping to BT pairs (train)")
    val_data = val_data.map(to_bt_pair, remove_columns=val_data.column_names, desc="Mapping to BT pairs (val)")
    return train_data, val_data


_LOADERS = {
    "anthropic-hh": _load_anthropic_hh,
    "helpsteer2": _load_helpsteer2,
}


def prepare_dataset(config: RewardConfig, output_path: str) -> None:
    """Download dataset from HuggingFace, tokenize it, and save to disk.

    Dispatches to a dataset-specific loader based on config.dataset, then
    tokenizes and saves in a common format.

    Args:
        config: RewardConfig with dataset, model_name, and max_length
        output_path: Local directory path to save the tokenized dataset

    Example:
        from jl.reward_model.prep_data import prepare_dataset
        from jl.reward_model.reward_config import RewardConfig

        config = RewardConfig()
        prepare_dataset(config, "./data/tokenized")
    """
    output_dir = Path(output_path)

    loader = _LOADERS.get(config.dataset)
    if loader is None:
        raise ValueError(f"Unknown dataset: {config.dataset!r}. Expected one of {list(_LOADERS)}")

    logger.info(f"Preparing dataset: {config.dataset}")
    logger.info(f"Tokenizer model: {config.model_name}")
    logger.info(f"Max length: {config.max_length}")
    logger.info(f"Output path: {output_dir}")

    logger.info("Downloading dataset from HuggingFace...")
    train_data, val_data = loader(config)
    logger.info(f"Downloaded: {len(train_data)} train, {len(val_data) if val_data else 0} val examples")

    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Tokenizing dataset...")

    def tokenize_pair(example):
        chosen_text = example["chosen"] + (tokenizer.eos_token or "")
        rejected_text = example["rejected"] + (tokenizer.eos_token or "")
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

    logger.info(f"Saving tokenized dataset to {output_dir}...")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    tokenized_dataset.save_to_disk(str(output_dir))

    logger.info("Data preparation complete")
