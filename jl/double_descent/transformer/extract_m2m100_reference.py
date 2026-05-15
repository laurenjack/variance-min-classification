"""Extract M2M100-12B reference log p(y_i | context_i) for IWSLT splits.

Slim version: at each non-padding decoder position, stores only the log-probability
of the actual target token y_i under M2M100-12B, renormalized over the compact
(~18K) IWSLT vocabulary subset that the small transformer was trained on.

Used to measure how surprising each token is under the oracle, for correlating
against per-token influence scores AND for the variance-mode distributional
bias-variance decomposition on the held-out in-distribution test chunk.

Index conventions for `--split train` with `--variance-split-id`:
  - split_id in 0..num_splits-1 → training chunk (matches
    `load_m2m100_iwslt14_variance_split`).
  - split_id == num_splits      → the held-out in-distribution test chunk
    used by transformer variance mode (matches the test_dataset returned by
    `load_m2m100_iwslt14_variance_split`).

Output (.pt):
  - log_probs:        1D fp16, log p_oracle(y_i | context_i), one per non-pad token
  - sentence_offsets: 1D long, cumulative position counts (len = n_sentences + 1)
  - target_ids:       1D short, compact ID of y_i at each position
  - split:            "train" | "valid" | "test"
  - split_id:         int or None
  - num_splits, samples_per_split, subsample_seed: metadata

Usage:
    # Train split 0 (matches the d_model=*, split=0 small transformers)
    python -m jl.double_descent.transformer.extract_m2m100_reference \\
        --data-path ./data/iwslt14.m2m100.de-en \\
        --output-path ./data/iwslt14.m2m100.de-en/train_split0_log_probs.pt \\
        --split train --variance-split-id 0

    # Held-out in-distribution test chunk (variance mode)
    python -m jl.double_descent.transformer.extract_m2m100_reference \\
        --data-path ./data/iwslt14.m2m100.de-en \\
        --output-path ./data/iwslt14.m2m100.de-en/holdout_log_probs.pt \\
        --split train --variance-split-id 4 --num-splits 4

    # Standalone test set (no variance carve)
    python -m jl.double_descent.transformer.extract_m2m100_reference \\
        --data-path ./data/iwslt14.m2m100.de-en \\
        --output-path ./data/iwslt14.m2m100.de-en/test_log_probs.pt \\
        --split test
"""

import argparse
import json
import logging
import random
from pathlib import Path

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def get_variance_split_indices(
    n_total: int,
    split_id: int,
    num_splits: int = 4,
    samples_per_split: int = 36000,
    seed: int = 42,
):
    """Reproduce the index slice used by load_m2m100_iwslt14_variance_split.

    Shuffle [0..n_total) with random.Random(seed), then take the contiguous
    [split_id * samples_per_split, (split_id+1) * samples_per_split) chunk.

    split_id in [0, num_splits-1] → training chunk.
    split_id == num_splits        → held-out in-distribution test chunk
                                    (matches load_m2m100_iwslt14_variance_split's
                                     test_dataset).
    """
    if split_id < 0 or split_id > num_splits:
        raise ValueError(
            f"split_id must be in [0, {num_splits}] (held-out chunk = "
            f"{num_splits}), got {split_id}"
        )
    # When extracting the held-out chunk we need (num_splits + 1) *
    # samples_per_split available; otherwise num_splits * samples_per_split
    # is enough.
    chunks_required = num_splits + 1 if split_id == num_splits else num_splits
    required = chunks_required * samples_per_split
    if n_total < required:
        raise ValueError(
            f"Not enough data for {chunks_required} chunks of "
            f"{samples_per_split} samples. Need {required}, have {n_total}."
        )
    rng = random.Random(seed)
    indices = list(range(n_total))
    rng.shuffle(indices)
    start = split_id * samples_per_split
    return indices[start : start + samples_per_split]


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Extract M2M100-12B log p(y_i | context_i) on an IWSLT split."
    )
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument(
        "--split", type=str, required=True,
        choices=["train", "valid", "test"],
    )
    parser.add_argument(
        "--variance-split-id", type=int, default=None,
        help="Train only: disjoint subset index. 0..num_splits-1 selects a "
             "training chunk; num_splits selects the held-out in-distribution "
             "test chunk used by variance mode.",
    )
    parser.add_argument("--num-splits", type=int, default=4)
    parser.add_argument("--samples-per-split", type=int, default=36000)
    parser.add_argument("--subsample-seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    data_path = Path(args.data_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(data_path / "vocab_mapping.json") as f:
        mapping = json.load(f)

    compact_to_m2m100 = mapping["compact_to_m2m100"]
    m2m100_to_compact = {mid: cid for cid, mid in enumerate(compact_to_m2m100)}
    vocab_size = mapping["vocab_size"]
    unk_idx = mapping["unk_idx"]

    logger.info(f"Compact vocab size: {vocab_size}")

    from transformers import AutoTokenizer, M2M100ForConditionalGeneration

    logger.info("Loading M2M100-12B model (fp16)...")
    model_name = "facebook/m2m100-12B-last-ckpt"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.src_lang = "de"
    tokenizer.tgt_lang = "en"

    model = M2M100ForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=torch.float16,
    ).to(device)
    model.eval()
    logger.info("Model loaded.")

    with open(data_path / f"{args.split}.de.txt", "r", encoding="utf-8") as f:
        src_texts = [line.strip() for line in f]
    with open(data_path / f"{args.split}.en.txt", "r", encoding="utf-8") as f:
        tgt_texts = [line.strip() for line in f]
    assert len(src_texts) == len(tgt_texts)
    logger.info(f"{args.split}: {len(src_texts)} sentence pairs available")

    if args.split == "train" and args.variance_split_id is not None:
        idx = get_variance_split_indices(
            n_total=len(src_texts),
            split_id=args.variance_split_id,
            num_splits=args.num_splits,
            samples_per_split=args.samples_per_split,
            seed=args.subsample_seed,
        )
        src_texts = [src_texts[i] for i in idx]
        tgt_texts = [tgt_texts[i] for i in idx]
        logger.info(
            f"Selected variance split {args.variance_split_id}/{args.num_splits}: "
            f"{len(src_texts)} sentence pairs"
        )

    extract_indices = torch.tensor(compact_to_m2m100, dtype=torch.long, device=device)

    all_log_probs = []
    all_target_ids = []
    sentence_offsets = [0]

    num_batches = (len(src_texts) + args.batch_size - 1) // args.batch_size

    with torch.no_grad():
        for batch_idx in range(num_batches):
            start = batch_idx * args.batch_size
            end = min(start + args.batch_size, len(src_texts))
            batch_de = src_texts[start:end]
            batch_en = tgt_texts[start:end]

            src_encoded = tokenizer(
                batch_de, return_tensors="pt", padding=True, truncation=True,
                max_length=512,
            ).to(device)

            tgt_encoded = tokenizer(
                text_target=batch_en, return_tensors="pt", padding=True,
                truncation=True, max_length=512,
            )
            tgt_input_ids = tgt_encoded["input_ids"].to(device)
            tgt_attention_mask = tgt_encoded["attention_mask"].to(device)

            # Prepend decoder_start_token_id (M2M100 was trained with </s>=2 at
            # decoder position 0). Without this prefix the decoder is fed
            # __en__ at position 0 — out-of-distribution — and per-token NLL
            # is inflated 4-9x. See oracle_diagnostic.py which uses labels= to
            # let HF auto-shift.
            bsz = tgt_input_ids.size(0)
            start_id = model.config.decoder_start_token_id
            start_col = torch.full(
                (bsz, 1), start_id, dtype=tgt_input_ids.dtype, device=device,
            )
            decoder_input_ids = torch.cat([start_col, tgt_input_ids[:, :-1]], dim=1)

            outputs = model(
                input_ids=src_encoded["input_ids"],
                attention_mask=src_encoded["attention_mask"],
                decoder_input_ids=decoder_input_ids,
            )

            # Slice compact columns from fp16 logits before upcasting.
            compact_logits = outputs.logits[:, :, extract_indices].float()
            del outputs

            # logits[i] now predicts tgt_input_ids[i] (HF-shift convention).
            # Drop position 0 (predicts the leading __en__ language tag) so
            # we keep only content-token predictions.
            target_native_ids = tgt_input_ids                # [batch, tgt_len]
            target_mask = tgt_attention_mask                  # [batch, tgt_len]

            log_probs = F.log_softmax(compact_logits, dim=-1)
            del compact_logits

            # Map M2M100 native target IDs to compact IDs (vectorized).
            # target_native -> compact via lookup; OOV -> unk_idx.
            # Build a flat dense lookup once on device for speed.
            # (Lazy: per-batch dict lookup, ~100s of tokens — fine.)
            # Use gather to extract log p at the target compact ID.
            target_native_flat = target_native_ids.cpu().tolist()
            target_compact = torch.tensor(
                [
                    [m2m100_to_compact.get(tid, unk_idx) for tid in row]
                    for row in target_native_flat
                ],
                dtype=torch.long, device=device,
            )

            # log p(y_i) at every position: [batch, tgt_len-1]
            log_p_target = log_probs.gather(-1, target_compact.unsqueeze(-1)).squeeze(-1)
            del log_probs

            for i in range(len(batch_de)):
                # Position 0 predicts __en__ (language tag); skip it.
                content_mask = target_mask[i].bool().clone()
                content_mask[0] = False
                n_positions = int(content_mask.sum().item())

                if n_positions > 0:
                    all_log_probs.append(log_p_target[i][content_mask].cpu())
                    all_target_ids.append(target_compact[i][content_mask].cpu().short())

                sentence_offsets.append(sentence_offsets[-1] + n_positions)

            if (batch_idx + 1) % 10 == 0 or batch_idx == num_batches - 1:
                total_positions = sentence_offsets[-1]
                logger.info(
                    f"Batch {batch_idx + 1}/{num_batches}: "
                    f"{total_positions} positions cumulative"
                )

    all_log_probs = torch.cat(all_log_probs, dim=0)
    all_target_ids = torch.cat(all_target_ids, dim=0)
    sentence_offsets = torch.tensor(sentence_offsets, dtype=torch.long)

    total_positions = all_log_probs.shape[0]
    logger.info(f"Total positions: {total_positions}")
    logger.info(f"Mean log p(y_i): {all_log_probs.mean().item():.4f}")
    logger.info(f"NLL (per token): {-all_log_probs.mean().item():.4f}")

    output = {
        "log_probs": all_log_probs.half(),
        "sentence_offsets": sentence_offsets,
        "target_ids": all_target_ids.short(),
        "split": args.split,
        "split_id": args.variance_split_id,
        "num_splits": args.num_splits,
        "samples_per_split": args.samples_per_split,
        "subsample_seed": args.subsample_seed,
        "vocab_size": vocab_size,
    }

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, args.output_path)

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(f"Saved to {args.output_path} ({file_size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
