"""Fine-tune Transformer final layer with L-BFGS + L2 regularization.

Loads trained Transformer models, unties the output projection from the
embedding, extracts decoder features, and fine-tunes only the output
projection layer to reach a stationary point.

Usage:
    python -m jl.double_descent.transformer.fine_tune \
        --model-path ./output/transformer/03-01-1010 \
        --data-path ./data/iwslt14.tokenized.de-en \
        --l2-lambda 1e-5 --max-steps 100
"""

import argparse
import json
import logging
from functools import partial
from pathlib import Path

import torch
import torch.multiprocessing as mp
import torch.nn as nn
from torch.utils.data import DataLoader

from jl.double_descent.fine_tune_lib import fine_tune_final_layer
from jl.double_descent.transformer.evaluation import discover_models
from jl.double_descent.transformer.transformer_config import TDDConfig
from jl.double_descent.transformer.transformer_data import (
    build_vocab,
    collate_fn,
    load_iwslt14,
)
from jl.double_descent.transformer.transformer_model import TransformerModel

logger = logging.getLogger(__name__)


def fine_tune_worker(
    gpu_id: int,
    d_model: int,
    train_samples: int,
    model_path: str,
    data_path: str,
    output_dir: str,
    l2_lambda: float,
    max_steps: int,
) -> None:
    """Fine-tune a single Transformer model on one GPU."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    device = torch.device(f"cuda:{gpu_id}")
    samples_k = train_samples // 1000
    logger.info(f"[d={d_model}] Starting fine-tuning on GPU {gpu_id}")

    config = TDDConfig()

    # Build vocab
    vocab = build_vocab(data_path)
    vocab_size = len(vocab)

    # Load model
    model = TransformerModel(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        d_ff_multiplier=config.d_ff_multiplier,
        pad_idx=vocab.pad_idx,
    ).to(device)
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # Create untied output projection with copied weights
    new_linear = nn.Linear(d_model, vocab_size, bias=False).to(device)
    new_linear.weight.data.copy_(model.output_proj.weight.data)

    # Load training data (same subsample as original training)
    train_dataset, _, _, _ = load_iwslt14(
        data_path, train_samples=train_samples, subsample_seed=config.subsample_seed
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=False,
        collate_fn=partial(collate_fn, pad_idx=vocab.pad_idx),
    )

    # Extract features using forward hook on decoder_norm
    features_cache = []

    def hook_fn(module, input, output):
        features_cache.append(output.detach())

    handle = model.decoder_norm.register_forward_hook(hook_fn)

    all_features = []
    all_targets = []

    with torch.no_grad():
        for src, tgt in train_loader:
            src = src.to(device)
            tgt = tgt.to(device)

            features_cache.clear()
            _ = model(src, tgt[:, :-1])

            feats = features_cache[0]  # [B, tgt_len-1, d_model]
            target = tgt[:, 1:].contiguous()  # [B, tgt_len-1]
            mask = target != vocab.pad_idx  # [B, tgt_len-1]

            all_features.append(feats[mask])  # [num_valid_tokens, d_model]
            all_targets.append(target[mask])  # [num_valid_tokens]

    handle.remove()

    features = torch.cat(all_features, dim=0)
    targets = torch.cat(all_targets, dim=0)
    logger.info(f"[d={d_model}] Extracted features: {features.shape}")

    # Free model memory
    del model
    torch.cuda.empty_cache()

    # Fine-tune
    metadata = fine_tune_final_layer(
        features=features,
        targets=targets,
        linear_layer=new_linear,
        l2_lambda=l2_lambda,
        max_steps=max_steps,
        device=device,
    )
    metadata["d_model"] = d_model
    metadata["train_samples"] = train_samples

    # Save fine-tuned layer
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    layer_path = out_path / f"layer_d{d_model}_{samples_k}k.pt"
    torch.save(new_linear.state_dict(), layer_path)
    logger.info(f"[d={d_model}] Saved fine-tuned layer to {layer_path}")

    # Append metadata
    metadata_path = out_path / "fine_tune_metadata.jsonl"
    with open(metadata_path, "a") as f:
        f.write(json.dumps(metadata) + "\n")
    logger.info(
        f"[d={d_model}] Done: loss={metadata['final_loss']:.6f}, "
        f"grad_norm={metadata['final_grad_norm']:.2e}"
    )


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Fine-tune Transformer final layers with L-BFGS + L2"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Directory containing model_d*_*k.pt files",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Directory containing preprocessed IWSLT data",
    )
    parser.add_argument(
        "--l2-lambda",
        type=float,
        default=1e-5,
        help="L2 regularization strength (default: 1e-5)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=100,
        help="Number of L-BFGS steps (default: 100)",
    )
    args = parser.parse_args()

    # Discover models
    models = discover_models(args.model_path)
    if not models:
        raise FileNotFoundError(
            f"No model_d*_*k.pt files found in {args.model_path}"
        )
    logger.info(f"Found models: {list(models.keys())}")

    output_dir = str(Path(args.model_path) / "fine_tuned")

    # Clear existing metadata file
    metadata_path = Path(output_dir) / "fine_tune_metadata.jsonl"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    if metadata_path.exists():
        metadata_path.unlink()

    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("No GPUs available. Fine-tuning requires at least one GPU.")
    logger.info(f"Using {num_gpus} GPUs")

    model_keys = list(models.keys())

    # Process in batches of num_gpus
    mp.set_start_method("spawn", force=True)
    for batch_start in range(0, len(model_keys), num_gpus):
        batch = model_keys[batch_start : batch_start + num_gpus]
        batch_num = batch_start // num_gpus + 1
        total_batches = (len(model_keys) + num_gpus - 1) // num_gpus
        logger.info(f"Batch {batch_num}/{total_batches}: {batch}")

        processes = []
        for gpu_id, (d_model, train_samples) in enumerate(batch):
            p = mp.Process(
                target=fine_tune_worker,
                args=(
                    gpu_id,
                    d_model,
                    train_samples,
                    str(models[(d_model, train_samples)]),
                    args.data_path,
                    output_dir,
                    args.l2_lambda,
                    args.max_steps,
                ),
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
            if p.exitcode != 0:
                logger.error(f"Worker exited with code {p.exitcode}")

        logger.info(f"Batch {batch_num}/{total_batches} complete")

    logger.info(f"Fine-tuning complete. Results in {output_dir}")


if __name__ == "__main__":
    main()
