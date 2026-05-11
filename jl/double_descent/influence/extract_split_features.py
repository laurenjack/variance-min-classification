#!/usr/bin/env python3
"""Extract decoder features for a held-out variance split, using a trained
transformer checkpoint.  Used to produce an in-distribution "test set" for
the influence-share analysis (since the actual IWSLT test split has a much
higher mean oracle surprise than train, biasing the 85/15 metric).

Usage:
    python -m jl.double_descent.influence.extract_split_features \\
        --model-path data/transformer_m2m100/04-28-1534/model_d112_36k.pt \\
        --d-model 112 \\
        --variance-split-id 1 \\
        --data-path data/iwslt14.m2m100.de-en \\
        --output-features data/transformer_m2m100/04-28-1534/distill_lam1e-5_lr3e-3/d112_split0/features_test_indist.pt

Notes:
- Uses variance_split with subsample_seed=42, samples_per_split=36000.
- Saves a dict matching the format of features_test.pt:
    {features (fp16), target_ids (int16), sentence_offsets (int64)}
"""
import argparse
import logging
from pathlib import Path

import torch

from jl.double_descent.transformer.transformer_config import TDDConfig
from jl.double_descent.transformer.transformer_data import (
    M2M100Vocab,
    load_m2m100_iwslt14_variance_split,
)
from jl.double_descent.transformer.transformer_model import TransformerModel
from jl.double_descent.influence.transformer_main import (
    extract_decoder_features,
    untie_output_proj,
)

logger = logging.getLogger(__name__)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", required=True)
    p.add_argument("--d-model", type=int, required=True)
    p.add_argument("--variance-split-id", type=int, required=True,
                   help="Index 1, 2, or 3 (0 was used for training).")
    p.add_argument("--data-path", required=True)
    p.add_argument("--output-features", required=True,
                   help="Where to write features_*.pt")
    p.add_argument("--num-splits", type=int, default=4)
    p.add_argument("--samples-per-split", type=int, default=36000)
    p.add_argument("--subsample-seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(name)s] %(message)s",
                        datefmt="%H:%M:%S")
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logger.info(f"device = {device}")

    # Load the requested variance split as a torch Dataset.  This reuses the
    # same shuffle/seed as training; split_id != 0 is unseen by the model.
    train_split, _, _, vocab = load_m2m100_iwslt14_variance_split(
        args.data_path, args.variance_split_id, args.num_splits,
        args.samples_per_split,
    )
    # NOTE: variance_split assigns split_id's slice to `train`, but we use
    # the split as the held-out evaluation set here.
    eval_dataset = train_split
    logger.info(
        f"Loaded variance split {args.variance_split_id}: "
        f"{len(eval_dataset)} sentences (seed={args.subsample_seed}, "
        f"num_splits={args.num_splits}, samples_per_split={args.samples_per_split})"
    )

    # Build model + load checkpoint
    cfg = TDDConfig()
    model = TransformerModel(
        vocab_size=len(vocab), d_model=args.d_model,
        n_layers=cfg.n_layers, n_heads=cfg.n_heads,
        d_ff_multiplier=cfg.d_ff_multiplier, pad_idx=vocab.pad_idx,
    ).to(device)
    state = torch.load(args.model_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    untie_output_proj(model)
    model.eval()
    logger.info(f"Loaded model from {args.model_path}; d_model={args.d_model}")

    # Forward pass through the model, capture pre-output-proj hidden states.
    logger.info("Extracting decoder features...")
    features, targets, offsets = extract_decoder_features(
        model, eval_dataset, vocab, device, args.batch_size,
    )
    logger.info(
        f"Extracted: {features.shape[0]} tokens, d={features.shape[1]}, "
        f"sentences={len(offsets) - 1}"
    )

    out_path = Path(args.output_features)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "features": features.half(),
            "target_ids": targets.short(),
            "sentence_offsets": offsets,
        },
        out_path,
    )
    logger.info(f"Wrote {out_path}  ({out_path.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
