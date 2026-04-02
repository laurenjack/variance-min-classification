"""Extract M2M100-12B reference logits on the IWSLT test set.

One-time script that runs on an H100. Teacher-forces the M2M100-12B model
on each test sentence (German source → English target), extracts next-token
distributions renormalized to the compact IWSLT vocab subset (~18K tokens),
and stores the full log-probability distribution per position.

Also computes per-position entropy H(p) from the renormalized distribution.

Output: reference_logits.pt with:
  - sentence_offsets: cumulative position counts
  - target_ids: compact IDs of actual target tokens
  - log_probs: [total_positions, vocab_size] full log-probabilities (fp16)
  - entropy: [total_positions] H(p) per position

Usage:
    python -m jl.double_descent.transformer.extract_m2m100_reference \\
        --data-path ./data/iwslt14.m2m100.de-en \\
        --output-path ./data/iwslt14.m2m100.de-en/reference_logits.pt \\
        --batch-size 128
"""

import argparse
import json
import logging
from pathlib import Path

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Extract M2M100-12B reference logits on IWSLT test set"
    )
    parser.add_argument(
        "--data-path", type=str, required=True,
        help="Directory with M2M100-preprocessed IWSLT data (vocab_mapping.json, test.*.txt)",
    )
    parser.add_argument(
        "--output-path", type=str, required=True,
        help="Path to save reference_logits.pt",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    data_path = Path(args.data_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load vocab mapping
    with open(data_path / "vocab_mapping.json") as f:
        mapping = json.load(f)

    compact_to_m2m100 = mapping["compact_to_m2m100"]
    m2m100_to_compact = {mid: cid for cid, mid in enumerate(compact_to_m2m100)}
    vocab_size = mapping["vocab_size"]
    unk_idx = mapping["unk_idx"]

    logger.info(f"Compact vocab size: {vocab_size}")

    # Load M2M100-12B model and tokenizer
    from transformers import AutoTokenizer, M2M100ForConditionalGeneration

    logger.info("Loading M2M100-12B model...")
    model_name = "facebook/m2m100-12B-last-ckpt"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.src_lang = "de"
    tokenizer.tgt_lang = "en"

    model = M2M100ForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=torch.float16,
    ).to(device)
    model.eval()
    logger.info("Model loaded.")

    # Load test data (original text for M2M100's own tokenizer)
    with open(data_path / "test.de.txt", "r", encoding="utf-8") as f:
        test_de = [line.strip() for line in f]
    with open(data_path / "test.en.txt", "r", encoding="utf-8") as f:
        test_en = [line.strip() for line in f]

    assert len(test_de) == len(test_en)
    logger.info(f"Test set: {len(test_de)} sentence pairs")

    # Index tensor for extracting compact vocab columns from M2M100 logits
    extract_indices = torch.tensor(compact_to_m2m100, dtype=torch.long, device=device)

    # Process test set in batches
    all_log_probs = []
    all_entropy = []
    all_target_ids = []
    sentence_offsets = [0]

    num_batches = (len(test_de) + args.batch_size - 1) // args.batch_size

    with torch.no_grad():
        for batch_idx in range(num_batches):
            start = batch_idx * args.batch_size
            end = min(start + args.batch_size, len(test_de))
            batch_de = test_de[start:end]
            batch_en = test_en[start:end]

            # Tokenize source (German) with M2M100 tokenizer
            src_encoded = tokenizer(
                batch_de, return_tensors="pt", padding=True, truncation=True,
                max_length=512,
            ).to(device)

            # Tokenize target (English) for teacher-forcing
            tgt_encoded = tokenizer(
                text_target=batch_en, return_tensors="pt", padding=True,
                truncation=True, max_length=512,
            )
            tgt_input_ids = tgt_encoded["input_ids"].to(device)
            tgt_attention_mask = tgt_encoded["attention_mask"].to(device)

            # Teacher-forced forward pass
            decoder_input_ids = tgt_input_ids[:, :-1]

            outputs = model(
                input_ids=src_encoded["input_ids"],
                attention_mask=src_encoded["attention_mask"],
                decoder_input_ids=decoder_input_ids,
            )

            # Extract compact vocab columns before upcasting (128K fp16 -> 18K fp16 -> 18K fp32)
            compact_logits = outputs.logits[:, :, extract_indices].float()
            del outputs

            # Target tokens (what each position is predicting)
            target_native_ids = tgt_input_ids[:, 1:]  # [batch, tgt_len-1]
            target_mask = tgt_attention_mask[:, 1:]    # [batch, tgt_len-1]

            # Renormalize over compact vocab (log_softmax)
            log_probs = F.log_softmax(compact_logits, dim=-1)
            del compact_logits

            # Compute entropy from full renormalized distribution
            probs = log_probs.exp()
            entropy = -(probs * log_probs).sum(dim=-1)  # [batch, tgt_len-1]
            del probs

            # Process each sentence in batch, skipping padding positions
            for i in range(len(batch_de)):
                mask = target_mask[i].bool()

                # M2M100 target is [</s>, __en__, tok1, tok2, ..., </s>]
                # After shifting: predicting [__en__, tok1, tok2, ..., </s>]
                # Skip position 0 (predicting __en__), keep positions 1+ (predicting content)
                if mask.sum() > 1:
                    content_mask = mask.clone()
                    content_mask[0] = False
                    n_positions = content_mask.sum().item()

                    all_log_probs.append(log_probs[i][content_mask].cpu())
                    all_entropy.append(entropy[i][content_mask].cpu())

                    # Map target native IDs to compact IDs
                    target_native = target_native_ids[i][content_mask].cpu().tolist()
                    target_compact = [m2m100_to_compact.get(tid, unk_idx) for tid in target_native]
                    all_target_ids.append(torch.tensor(target_compact, dtype=torch.short))
                else:
                    n_positions = 0

                sentence_offsets.append(sentence_offsets[-1] + n_positions)

            if (batch_idx + 1) % 10 == 0 or batch_idx == num_batches - 1:
                total_positions = sentence_offsets[-1]
                logger.info(
                    f"Batch {batch_idx + 1}/{num_batches}: "
                    f"{total_positions} total positions so far"
                )

    # Concatenate all results
    all_log_probs = torch.cat(all_log_probs, dim=0)      # [total_positions, vocab_size]
    all_entropy = torch.cat(all_entropy, dim=0)           # [total_positions]
    all_target_ids = torch.cat(all_target_ids, dim=0)     # [total_positions]
    sentence_offsets = torch.tensor(sentence_offsets, dtype=torch.long)

    total_positions = all_log_probs.shape[0]
    logger.info(f"\nTotal positions: {total_positions}")
    logger.info(f"Vocab size: {vocab_size}")
    logger.info(f"Mean entropy: {all_entropy.mean().item():.4f}")

    # Save
    output = {
        'sentence_offsets': sentence_offsets,
        'target_ids': all_target_ids.short(),
        'log_probs': all_log_probs.half(),
        'entropy': all_entropy.half(),
        'vocab_size': vocab_size,
    }

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, args.output_path)

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(f"Saved to {args.output_path} ({file_size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
