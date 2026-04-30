#!/usr/bin/env python3
"""Re-run decomposition validation on cached distill outputs.

Loads features_train, untied_output_proj from a prior `transformer_main --distill`
run, recomputes W_orig from the source model checkpoint, and validates the
Yeh-Kim §3.2 identity using the *correct* distillation residual
(R = sigma_new − sigma_orig). Writes an updated validation_distill.json.

Usage:
    python -m jl.double_descent.influence.transformer_revalidate \\
        --source-dir <distill run dir, e.g. .../d40_split0> \\
        --orig-model-path <model_d{N}_36k.pt> \\
        --d-model 40 \\
        --data-path ./data/iwslt14.m2m100.de-en \\
        --lambda-l2 1e-6
"""

import argparse
import json
import logging
from pathlib import Path

import torch
import torch.nn as nn

from jl.double_descent.transformer.transformer_config import TDDConfig
from jl.double_descent.transformer.transformer_data import M2M100Vocab
from jl.double_descent.transformer.transformer_model import TransformerModel
from jl.double_descent.influence.transformer_main import (
    untie_output_proj,
    validate_decomposition_chunked,
)

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True,
                        help="Directory with features_train.pt + untied_output_proj.pt")
    parser.add_argument("--orig-model-path", required=True,
                        help="Path to the original trained model (the distill source)")
    parser.add_argument("--d-model", type=int, required=True)
    parser.add_argument("--data-path", required=True,
                        help="IWSLT M2M100 data dir (for vocab_mapping.json)")
    parser.add_argument("--lambda-l2", type=float, required=True)
    parser.add_argument("--feature-chunk", type=int, default=4096)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(name)s] %(message)s",
                        datefmt="%H:%M:%S")

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")

    src = Path(args.source_dir)
    vocab = M2M100Vocab(str(Path(args.data_path) / "vocab_mapping.json"))

    # 1) Load cached features + untied output_proj
    train_blob = torch.load(src / "features_train.pt", map_location="cpu", weights_only=False)
    proj_state = torch.load(src / "untied_output_proj.pt", map_location="cpu", weights_only=True)
    f_train = train_blob["features"].float().to(device)
    y_train = train_blob["target_ids"].long().to(device)

    d_model = f_train.size(1)
    vocab_size = proj_state["weight"].size(0)
    assert d_model == args.d_model, f"feature dim {d_model} != --d-model {args.d_model}"

    output_proj = nn.Linear(d_model, vocab_size, bias=True).to(device)
    output_proj.load_state_dict({k: v.to(device) for k, v in proj_state.items()})
    output_proj.eval()
    logger.info(f"Loaded fitted output_proj from {src}/untied_output_proj.pt")

    # 2) Recover W_orig + b_orig from the original trained model.
    # Reproduce what `untie_output_proj` did at training time: copy the embedding
    # weight into a new bias=True linear with bias=0.
    config = TDDConfig()
    model = TransformerModel(
        vocab_size=vocab_size, d_model=args.d_model,
        n_layers=config.n_layers, n_heads=config.n_heads,
        d_ff_multiplier=config.d_ff_multiplier, pad_idx=vocab.pad_idx,
    ).to(device)
    state = torch.load(args.orig_model_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    untie_output_proj(model)
    distill_W_orig = model.output_proj.weight.detach().clone()
    distill_b_orig = model.output_proj.bias.detach().clone()
    logger.info(f"Recovered W_orig from {args.orig_model_path}")
    del model

    # 3) Run validate with the *distillation* residual
    val_stats = validate_decomposition_chunked(
        f_train, y_train, output_proj,
        lambda_l2=args.lambda_l2, chunk_size=args.feature_chunk,
        distill_W_orig=distill_W_orig, distill_b_orig=distill_b_orig,
    )

    # 4) Save
    out_path = src / "validation_distill.json"
    out_path.write_text(json.dumps(val_stats, indent=2))
    logger.info(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
