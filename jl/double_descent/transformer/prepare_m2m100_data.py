"""Preprocess IWSLT'14 de-en data using M2M100's SentencePiece tokenizer.

Downloads the dataset from HuggingFace, tokenizes with M2M100's tokenizer,
builds a compact vocabulary mapping (subset of M2M100's 128K vocab that appears
in the training data), and saves pre-tokenized integer sequences.

Output directory: data/iwslt14.m2m100.de-en/
  - {split}.{lang}.ids   : space-separated compact token IDs (content only, no BOS/EOS)
  - {split}.{lang}.txt   : original mixed-case text
  - vocab_mapping.json    : compact <-> M2M100 ID mapping + special token indices

Usage:
    python -m jl.double_descent.transformer.prepare_m2m100_data [--output-dir DIR]
"""

import argparse
import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Preprocess IWSLT14 with M2M100 tokenizer")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/iwslt14.m2m100.de-en",
        help="Output directory for preprocessed data",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if already preprocessed
    if (output_dir / "vocab_mapping.json").exists():
        logger.info(f"Already preprocessed: {output_dir / 'vocab_mapping.json'} exists")
        return

    # Import dependencies
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("Install datasets: pip install datasets")
        sys.exit(1)

    try:
        from transformers import AutoTokenizer
    except ImportError:
        logger.error("Install transformers: pip install transformers")
        sys.exit(1)

    # Load M2M100 tokenizer (downloads tokenizer files only, not the model)
    logger.info("Loading M2M100 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("facebook/m2m100-12B-last-ckpt")

    # Download IWSLT14 dataset
    logger.info("Downloading IWSLT14 de-en from HuggingFace...")
    dataset = load_dataset("bbaaaa/iwslt14-de-en", revision="refs/convert/parquet")

    # Extract text and tokenize all splits
    splits = [("train", "train"), ("valid", "validation"), ("test", "test")]
    all_data = {}  # split_name -> {"de_text": [...], "en_text": [...], "de_ids": [...], "en_ids": [...]}

    for our_name, hf_name in splits:
        split_data = dataset[hf_name]
        de_texts = []
        en_texts = []
        de_ids_list = []
        en_ids_list = []

        for example in split_data:
            de_text = example["translation"]["de"].strip()
            en_text = example["translation"]["en"].strip()
            if not de_text or not en_text:
                continue

            de_texts.append(de_text)
            en_texts.append(en_text)

            # Tokenize without special tokens (content only)
            de_ids = tokenizer.encode(de_text, add_special_tokens=False)
            en_ids = tokenizer.encode(en_text, add_special_tokens=False)
            de_ids_list.append(de_ids)
            en_ids_list.append(en_ids)

        all_data[our_name] = {
            "de_text": de_texts,
            "en_text": en_texts,
            "de_ids": de_ids_list,
            "en_ids": en_ids_list,
        }
        logger.info(f"  {our_name}: {len(de_texts)} sentence pairs")

    # Collect unique M2M100 token IDs from training data
    logger.info("Building compact vocabulary from training data...")
    train_token_ids = set()
    for ids_list in [all_data["train"]["de_ids"], all_data["train"]["en_ids"]]:
        for ids in ids_list:
            train_token_ids.update(ids)

    # Add special tokens
    special_m2m100_ids = {
        tokenizer.pad_token_id,     # 1
        tokenizer.bos_token_id,     # 0 (<s>)
        tokenizer.eos_token_id,     # 2 (</s>)
        tokenizer.unk_token_id,     # 3
        tokenizer.convert_tokens_to_ids("__de__"),  # 128020
        tokenizer.convert_tokens_to_ids("__en__"),  # 128022
    }
    all_m2m100_ids = train_token_ids | special_m2m100_ids

    logger.info(f"  Training content tokens: {len(train_token_ids)}")
    logger.info(f"  + special tokens: {len(special_m2m100_ids)}")
    logger.info(f"  Total unique M2M100 IDs: {len(all_m2m100_ids)}")

    # Build compact mapping: sort M2M100 IDs, assign compact IDs 0..N-1
    # Force PAD (m2m100 id=1) to compact ID 0
    sorted_m2m100_ids = sorted(all_m2m100_ids)

    # Remove PAD from sorted list and prepend it
    pad_m2m100_id = tokenizer.pad_token_id  # 1
    sorted_m2m100_ids.remove(pad_m2m100_id)
    sorted_m2m100_ids = [pad_m2m100_id] + sorted_m2m100_ids

    # compact_to_m2m100[compact_id] = m2m100_id
    compact_to_m2m100 = sorted_m2m100_ids
    m2m100_to_compact = {m2m100_id: compact_id for compact_id, m2m100_id in enumerate(compact_to_m2m100)}

    # Identify special token compact IDs
    pad_idx = m2m100_to_compact[tokenizer.pad_token_id]      # 0 (forced)
    bos_idx = m2m100_to_compact[tokenizer.bos_token_id]
    eos_idx = m2m100_to_compact[tokenizer.eos_token_id]
    unk_idx = m2m100_to_compact[tokenizer.unk_token_id]
    de_lang_idx = m2m100_to_compact[tokenizer.convert_tokens_to_ids("__de__")]
    en_lang_idx = m2m100_to_compact[tokenizer.convert_tokens_to_ids("__en__")]

    vocab_size = len(compact_to_m2m100)
    logger.info(f"  Compact vocab size: {vocab_size}")
    logger.info(f"  PAD={pad_idx}, BOS={bos_idx}, EOS={eos_idx}, UNK={unk_idx}")
    logger.info(f"  __de__={de_lang_idx}, __en__={en_lang_idx}")

    # Check for test/valid tokens not in training vocab
    for split_name in ["valid", "test"]:
        split_ids = set()
        for ids_list in [all_data[split_name]["de_ids"], all_data[split_name]["en_ids"]]:
            for ids in ids_list:
                split_ids.update(ids)
        oov = split_ids - train_token_ids
        if oov:
            logger.warning(f"  {split_name} has {len(oov)} token IDs not in training vocab (will map to UNK)")

    # Save files
    logger.info(f"Saving to {output_dir}...")

    for split_name in ["train", "valid", "test"]:
        data = all_data[split_name]

        for lang in ["de", "en"]:
            # Save original text
            txt_path = output_dir / f"{split_name}.{lang}.txt"
            with open(txt_path, "w", encoding="utf-8") as f:
                for text in data[f"{lang}_text"]:
                    f.write(text + "\n")

            # Save compact IDs
            ids_path = output_dir / f"{split_name}.{lang}.ids"
            with open(ids_path, "w", encoding="utf-8") as f:
                for m2m100_ids in data[f"{lang}_ids"]:
                    compact_ids = [m2m100_to_compact.get(mid, unk_idx) for mid in m2m100_ids]
                    f.write(" ".join(str(cid) for cid in compact_ids) + "\n")

    # Save vocab mapping
    mapping = {
        "compact_to_m2m100": compact_to_m2m100,
        "pad_idx": pad_idx,
        "bos_idx": bos_idx,
        "eos_idx": eos_idx,
        "unk_idx": unk_idx,
        "de_lang_idx": de_lang_idx,
        "en_lang_idx": en_lang_idx,
        "vocab_size": vocab_size,
        "m2m100_model": "facebook/m2m100-12B-last-ckpt",
    }
    mapping_path = output_dir / "vocab_mapping.json"
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2)

    # Print summary
    logger.info("\nPreprocessing complete!")
    logger.info(f"Output: {output_dir}")
    for split_name in ["train", "valid", "test"]:
        n = len(all_data[split_name]["de_text"])
        logger.info(f"  {split_name}: {n} pairs")
    logger.info(f"  Vocab size: {vocab_size} (compact) from {tokenizer.vocab_size} (M2M100)")


if __name__ == "__main__":
    main()
