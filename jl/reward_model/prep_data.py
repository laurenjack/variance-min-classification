#!/usr/bin/env python3
"""Prepare and upload tokenized training data to S3.

One-time setup before running SageMaker jobs:
    source ~/.cursor_bootstrap.sh && source venv/bin/activate
    python -m jl.reward_model.prep_data

This downloads the dataset, tokenizes it, and uploads to S3.
Subsequent SageMaker jobs will load the pre-staged data directly,
avoiding the HuggingFace download overhead.
"""

import shutil
import subprocess
import tempfile
from pathlib import Path

from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer

# Configuration - must match main.py Config defaults
MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
HF_DATASET = "Anthropic/hh-rlhf"
SUBSET_NAME = None
MAX_LENGTH = 1024

# S3 destination
S3_BUCKET = "sagemaker-reward-model-100611042793"
S3_PREFIX = "data/hh-rlhf-tokenized"


def main():
    print("=" * 60)
    print("Reward Model Data Preparation")
    print("=" * 60)
    print(f"Dataset: {HF_DATASET}")
    print(f"Model (for tokenizer): {MODEL_NAME}")
    print(f"Max length: {MAX_LENGTH}")
    print(f"S3 destination: s3://{S3_BUCKET}/{S3_PREFIX}/")
    print()

    # Create temp directory for local processing
    tmp_dir = Path(tempfile.mkdtemp(prefix="reward_data_"))
    local_output = tmp_dir / "hh-rlhf-tokenized"
    print(f"Working directory: {tmp_dir}")
    print()

    try:
        # Step 1: Download dataset
        print("Step 1/4: Downloading dataset from HuggingFace...")
        dataset = load_dataset(HF_DATASET, name=SUBSET_NAME)
        train_data = dataset["train"]
        val_data = dataset.get("test", dataset.get("validation", None))
        print(f"  Train: {len(train_data)} examples")
        print(f"  Val: {len(val_data) if val_data else 0} examples")
        print()

        # Step 2: Initialize tokenizer
        print("Step 2/4: Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=False)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print(f"  Tokenizer loaded: {tokenizer.__class__.__name__}")
        print()

        # Step 3: Tokenize dataset
        print("Step 3/4: Tokenizing dataset...")
        
        def tokenize_pair(example):
            chosen = example["chosen"]
            rejected = example["rejected"]
            chosen_text = chosen + (tokenizer.eos_token or "")
            rejected_text = rejected + (tokenizer.eos_token or "")
            chosen_enc = tokenizer(chosen_text, max_length=MAX_LENGTH, padding=False, truncation=True)
            rejected_enc = tokenizer(rejected_text, max_length=MAX_LENGTH, padding=False, truncation=True)
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
        
        print(f"  Tokenization complete")
        print()

        # Step 4: Save and upload to S3
        print("Step 4/4: Saving and uploading to S3...")
        tokenized_dataset.save_to_disk(str(local_output))
        print(f"  Saved locally to {local_output}")

        s3_uri = f"s3://{S3_BUCKET}/{S3_PREFIX}/"
        print(f"  Uploading to {s3_uri}...")
        subprocess.run(
            ["aws", "s3", "sync", str(local_output), s3_uri, "--quiet"],
            check=True
        )
        print(f"  Upload complete!")
        print()

        print("=" * 60)
        print("SUCCESS! Data is ready for SageMaker jobs.")
        print(f"S3 location: {s3_uri}")
        print()
        print("Now run:")
        print("  python -m jl.reward_model.launch_sagemaker")
        print("=" * 60)

    finally:
        # Cleanup
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()

