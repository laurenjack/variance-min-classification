"""Calibration sweep for Transformer models on IWSLT'14 de-en.

Runs all calibration baselines (temperature scaling, vector scaling) plus
an L2 lambda sweep on the output projection layer. Skips histogram binning
and Dirichlet L2 (infeasible with ~10K vocab). After the token-level sweep,
runs multi-GPU BLEU evaluation for each calibrated head.

Usage:
    python -m jl.double_descent.calibration.calibrate_transformer \
        --model-path ./output/transformer/03-01-1010 \
        --data-path ./data/iwslt14.tokenized.de-en \
        --d-model 128
"""

import argparse
import json
import logging
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.multiprocessing as mp
import torch.nn as nn
from torch.utils.data import DataLoader

from jl.double_descent.calibration.evaluate import evaluate_logits_lightweight
from jl.double_descent.calibration.sweep import DEFAULT_LAMBDAS, run_calibration_sweep
from jl.double_descent.transformer.bleu import compute_bleu
from jl.double_descent.transformer.evaluation import discover_models
from jl.double_descent.transformer.transformer_config import TDDConfig
from jl.double_descent.transformer.transformer_data import (
    build_vocab,
    collate_fn,
    load_iwslt14,
)
from jl.double_descent.transformer.transformer_model import TransformerModel

logger = logging.getLogger(__name__)


def _extract_features(
    model: TransformerModel,
    dataset,
    vocab,
    device: torch.device,
    batch_size: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract decoder features using forward hook on decoder_norm.

    Returns:
        (features [num_valid_tokens, d_model], targets [num_valid_tokens])
    """
    features_cache = []

    def hook_fn(module, input, output):
        features_cache.append(output.detach())

    handle = model.decoder_norm.register_forward_hook(hook_fn)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=partial(collate_fn, pad_idx=vocab.pad_idx),
    )

    all_features = []
    all_targets = []

    with torch.no_grad():
        for src, tgt in loader:
            src = src.to(device)
            tgt = tgt.to(device)

            features_cache.clear()
            _ = model(src, tgt[:, :-1])

            feats = features_cache[0]  # [B, tgt_len-1, d_model]
            target = tgt[:, 1:].contiguous()  # [B, tgt_len-1]
            mask = target != vocab.pad_idx  # [B, tgt_len-1]

            all_features.append(feats[mask].cpu())  # [num_valid_tokens, d_model]
            all_targets.append(target[mask].cpu())  # [num_valid_tokens]

    handle.remove()

    features = torch.cat(all_features, dim=0)
    targets = torch.cat(all_targets, dim=0)
    return features, targets


def _bleu_worker(
    gpu_id: int,
    method_key: str,
    head_state: dict,
    head_has_bias: bool,
    d_model: int,
    vocab_size: int,
    model_file: str,
    data_path: str,
    train_samples: int,
    config: TDDConfig,
    dataset_split: str,
    result_dict: dict,
) -> None:
    """Compute BLEU for one calibrated head on one GPU."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    device = torch.device(f"cuda:{gpu_id}")

    # Build vocab
    vocab = build_vocab(data_path)

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
        torch.load(model_file, map_location=device, weights_only=True)
    )
    model.eval()

    # Untie output_proj and load calibrated weights
    model.output_proj = nn.Linear(d_model, vocab_size, bias=head_has_bias).to(device)
    model.output_proj.load_state_dict(head_state)
    model.output_proj.eval()

    # Load dataset
    train_dataset, val_dataset, test_dataset, _ = load_iwslt14(
        data_path, train_samples=train_samples, subsample_seed=config.subsample_seed
    )
    dataset = test_dataset if dataset_split == "test" else val_dataset

    bleu = compute_bleu(model, dataset, vocab, device, max_len=128, batch_size=32)
    logger.info(f"[{method_key}] BLEU = {bleu:.2f} (GPU {gpu_id})")
    result_dict[method_key] = bleu


def _build_temperature_head_state(
    original_state: dict, T: float
) -> Tuple[dict, bool]:
    """Build head state with temperature baked into weights.

    logits / T = (W / T) @ x, so new_weight = W / T.
    """
    state = {"weight": original_state["weight"] / T}
    return state, False


def _build_vector_scaled_head_state(
    original_state: dict,
    vs_weights: torch.Tensor,
    vs_biases: torch.Tensor,
) -> Tuple[dict, bool]:
    """Build head state with vector scaling baked in.

    scaled_logits = w * (W @ x) + b = diag(w) @ W @ x + b
    """
    W = original_state["weight"]  # [vocab_size, d_model]
    # diag(w) @ W: multiply each row i of W by w[i]
    new_W = vs_weights.unsqueeze(1) * W
    state = {"weight": new_W, "bias": vs_biases}
    return state, True


def _run_bleu_evaluation(
    methods: Dict[str, Tuple[dict, bool]],
    d_model: int,
    vocab_size: int,
    model_file: str,
    data_path: str,
    train_samples: int,
    config: TDDConfig,
    dataset_split: str = "test",
) -> Dict[str, float]:
    """Run BLEU evaluation for multiple methods, parallelizing across GPUs.

    Args:
        methods: Dict mapping method_key -> (head_state_dict, has_bias).
        dataset_split: "test" or "val".

    Returns:
        Dict mapping method_key -> BLEU score.
    """
    num_gpus = max(torch.cuda.device_count(), 1) if torch.cuda.is_available() else 1
    method_keys = list(methods.keys())

    if not torch.cuda.is_available() or num_gpus <= 1:
        # Sequential fallback
        result_dict = {}
        for method_key in method_keys:
            head_state, has_bias = methods[method_key]
            _bleu_worker(
                gpu_id=0,
                method_key=method_key,
                head_state=head_state,
                head_has_bias=has_bias,
                d_model=d_model,
                vocab_size=vocab_size,
                model_file=model_file,
                data_path=data_path,
                train_samples=train_samples,
                config=config,
                dataset_split=dataset_split,
                result_dict=result_dict,
            )
        return result_dict

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    manager = mp.Manager()
    result_dict = manager.dict()

    for batch_start in range(0, len(method_keys), num_gpus):
        batch = method_keys[batch_start: batch_start + num_gpus]
        batch_num = batch_start // num_gpus + 1
        total_batches = (len(method_keys) + num_gpus - 1) // num_gpus
        logger.info(
            f"BLEU batch {batch_num}/{total_batches} ({dataset_split}): "
            f"{batch}"
        )

        processes = []
        for gpu_id, method_key in enumerate(batch):
            head_state, has_bias = methods[method_key]
            p = mp.Process(
                target=_bleu_worker,
                args=(
                    gpu_id,
                    method_key,
                    head_state,
                    has_bias,
                    d_model,
                    vocab_size,
                    model_file,
                    data_path,
                    train_samples,
                    config,
                    dataset_split,
                    result_dict,
                ),
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
            if p.exitcode != 0:
                logger.error(f"BLEU worker exited with code {p.exitcode}")

    return dict(result_dict)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Calibration sweep for Transformer on IWSLT'14"
    )
    parser.add_argument(
        "--model-path", type=str, required=True,
        help="Directory containing model_d*_*k.pt files",
    )
    parser.add_argument(
        "--data-path", type=str, required=True,
        help="Directory containing preprocessed IWSLT data",
    )
    parser.add_argument(
        "--d-model", type=int, required=True,
        help="Model embedding dimension to calibrate",
    )
    parser.add_argument(
        "--output-path", type=str, default=None,
        help="Output directory (default: <model-path>/calibration_d<d_model>)",
    )
    parser.add_argument(
        "--max-steps", type=int, default=100,
        help="L-BFGS steps per lambda (default: 100)",
    )
    parser.add_argument(
        "--sweep-metric", type=str, default="ece",
        choices=["ece", "nll", "bleu"],
        help="Metric to select best lambda (default: ece)",
    )
    parser.add_argument(
        "--train-samples", type=int, default=36000,
        help="Training samples (must match original training, default: 36000)",
    )
    args = parser.parse_args()

    config = TDDConfig()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # === Discover model ===
    models = discover_models(args.model_path)
    if not models:
        raise FileNotFoundError(
            f"No model_d*_*k.pt files found in {args.model_path}"
        )

    model_key = (args.d_model, args.train_samples)
    if model_key not in models:
        available = sorted(models.keys())
        raise ValueError(
            f"No model found for d_model={args.d_model}, "
            f"train_samples={args.train_samples}. Available: {available}"
        )
    model_file = models[model_key]
    logger.info(f"Using model: {model_file}")

    # === Build vocab and load data ===
    vocab = build_vocab(args.data_path)
    vocab_size = len(vocab)
    logger.info(f"Vocab size: {vocab_size}")

    train_dataset, val_dataset, test_dataset, _ = load_iwslt14(
        args.data_path,
        train_samples=args.train_samples,
        subsample_seed=config.subsample_seed,
    )
    logger.info(
        f"Data: train={len(train_dataset)}, val={len(val_dataset)}, "
        f"test={len(test_dataset)}"
    )

    # === Load model ===
    model = TransformerModel(
        vocab_size=vocab_size,
        d_model=args.d_model,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        d_ff_multiplier=config.d_ff_multiplier,
        pad_idx=vocab.pad_idx,
    ).to(device)
    model.load_state_dict(
        torch.load(str(model_file), map_location=device, weights_only=True)
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # === Untie output projection ===
    new_linear = nn.Linear(args.d_model, vocab_size, bias=False).to(device)
    new_linear.weight.data.copy_(model.output_proj.weight.data)
    original_head_state = {k: v.cpu() for k, v in new_linear.state_dict().items()}

    # === Extract features ===
    logger.info("Extracting training features...")
    train_features, train_targets = _extract_features(
        model, train_dataset, vocab, device
    )
    logger.info(f"Train features: {train_features.shape}")

    logger.info("Extracting validation features...")
    val_features, val_targets = _extract_features(
        model, val_dataset, vocab, device
    )
    logger.info(f"Val features: {val_features.shape}")

    logger.info("Extracting test features...")
    test_features, test_targets = _extract_features(
        model, test_dataset, vocab, device
    )
    logger.info(f"Test features: {test_features.shape}")

    # Free model GPU memory (will reload for BLEU later)
    del model
    torch.cuda.empty_cache()

    # === Phase 1: Token-level sweep via sweep.py ===
    output_dir = args.output_path
    if output_dir is None:
        output_dir = str(
            Path(args.model_path) / f"calibration_d{args.d_model}"
        )
    output_dir = Path(output_dir)

    # Use ece/nll for Phase 1 selection; bleu selection happens in Phase 2
    phase1_metric = args.sweep_metric if args.sweep_metric != "bleu" else "ece"

    results = run_calibration_sweep(
        train_features=train_features,
        train_labels=train_targets,
        val_features=val_features,
        val_labels=val_targets,
        test_features=test_features,
        test_labels=test_targets,
        original_head_state=original_head_state,
        num_classes=vocab_size,
        feature_dim=args.d_model,
        lambdas=DEFAULT_LAMBDAS,
        max_steps=args.max_steps,
        sweep_metric=phase1_metric,
        device=device,
        output_dir=output_dir,
        skip_baselines=["histogram_binning", "dirichlet_l2"],
        evaluate_fn=evaluate_logits_lightweight,
    )

    # === Phase 2: Multi-GPU BLEU evaluation ===
    logger.info("=== Phase 2: BLEU evaluation ===")

    # Read back sweep results to get all lambda states
    sweep_data = json.loads((output_dir / "sweep_results.json").read_text())
    all_lambda_states = {}
    for entry in sweep_data:
        lam = entry["l2_lambda"]
        lam_key = f"l2_lambda_{lam:.0e}"
        # Re-fit to get the state: load from the sweep
        # Actually we need the states. Let's re-derive them from the sweep.
        # The sweep only saved the best state. We need to re-run or save all.
        # For now, let's load the best from the saved file.

    # We need all lambda states for BLEU. The sweep function only saves the best.
    # Re-run a lightweight lambda sweep to get all states.
    # Actually, let's re-do this more efficiently: just re-calibrate all lambdas
    # and keep the states in memory. We already have the features.

    logger.info("Re-calibrating all lambdas to collect head states for BLEU...")
    from jl.double_descent.l2_calibrate_lib import l2_calibrate_final_layer

    lambda_states = {}
    for lam in DEFAULT_LAMBDAS:
        linear = nn.Linear(args.d_model, vocab_size, bias=False).to(device)
        linear.load_state_dict(original_head_state)
        l2_calibrate_final_layer(
            features=train_features,
            targets=train_targets,
            linear_layer=linear,
            l2_lambda=lam,
            max_steps=args.max_steps,
            device=device,
        )
        lambda_states[lam] = {k: v.cpu() for k, v in linear.state_dict().items()}
        del linear
    torch.cuda.empty_cache()

    # Build temperature and vector scaling head states
    T = results["temperature_scaled"]["temperature"]
    ts_head_state, ts_has_bias = _build_temperature_head_state(
        original_head_state, T
    )

    # Re-fit vector scaling to get the weights/biases
    # (they weren't returned by sweep.py, so re-fit on val logits)
    head = nn.Linear(args.d_model, vocab_size, bias=False)
    head.load_state_dict(original_head_state)
    head.eval()
    with torch.no_grad():
        val_logits = head(val_features).detach()

    from jl.double_descent.calibration.baselines import fit_vector_scaling
    vs_weights, vs_biases = fit_vector_scaling(val_logits, val_targets)
    vs_head_state, vs_has_bias = _build_vector_scaled_head_state(
        original_head_state, vs_weights, vs_biases
    )
    del head, val_logits

    # Assemble all methods for BLEU
    bleu_methods: Dict[str, Tuple[dict, bool]] = {
        "uncalibrated": (original_head_state, False),
        "temperature_scaled": (ts_head_state, ts_has_bias),
        "vector_scaled": (vs_head_state, vs_has_bias),
    }
    for lam in DEFAULT_LAMBDAS:
        bleu_methods[f"l2_lambda_{lam:.0e}"] = (lambda_states[lam], False)

    # Determine which splits need BLEU
    splits_to_eval = ["test"]
    if args.sweep_metric == "bleu":
        splits_to_eval.insert(0, "val")

    bleu_results: Dict[str, Dict[str, float]] = {}  # method -> {split: bleu}
    for split in splits_to_eval:
        logger.info(f"Computing BLEU on {split} set...")
        split_bleu = _run_bleu_evaluation(
            methods=bleu_methods,
            d_model=args.d_model,
            vocab_size=vocab_size,
            model_file=str(model_file),
            data_path=args.data_path,
            train_samples=args.train_samples,
            config=config,
            dataset_split=split,
        )
        for method_key, bleu in split_bleu.items():
            if method_key not in bleu_results:
                bleu_results[method_key] = {}
            bleu_results[method_key][split] = bleu

    # If sweep_metric is bleu, re-select best lambda by val BLEU
    best_lambda = results["l2_calibrated"]["l2_lambda"]
    if args.sweep_metric == "bleu":
        best_val_bleu = -1.0
        for lam in DEFAULT_LAMBDAS:
            key = f"l2_lambda_{lam:.0e}"
            val_bleu = bleu_results.get(key, {}).get("val", 0.0)
            if val_bleu > best_val_bleu:
                best_val_bleu = val_bleu
                best_lambda = lam
        logger.info(
            f"Best lambda by val BLEU: {best_lambda:.0e} "
            f"(val BLEU={best_val_bleu:.2f})"
        )

    # Add BLEU to results
    for method in ["uncalibrated", "temperature_scaled", "vector_scaled"]:
        test_bleu = bleu_results.get(method, {}).get("test", 0.0)
        results[method]["bleu"] = round(test_bleu, 2)

    best_key = f"l2_lambda_{best_lambda:.0e}"
    test_bleu = bleu_results.get(best_key, {}).get("test", 0.0)
    results["l2_calibrated"]["bleu"] = round(test_bleu, 2)
    results["l2_calibrated"]["l2_lambda"] = best_lambda

    # Build full sweep results with BLEU
    sweep_with_bleu = []
    for entry in sweep_data:
        lam = entry["l2_lambda"]
        key = f"l2_lambda_{lam:.0e}"
        entry["test_bleu"] = round(
            bleu_results.get(key, {}).get("test", 0.0), 2
        )
        if args.sweep_metric == "bleu":
            entry["val_bleu"] = round(
                bleu_results.get(key, {}).get("val", 0.0), 2
            )
        sweep_with_bleu.append(entry)

    # === Print final summary table ===
    metric_cols = [
        ("nll", "NLL"),
        ("accuracy", "Acc"),
        ("ece", "ECE"),
        ("bleu", "BLEU"),
    ]
    header = f"{'Method':<20}" + "".join(f" {h:>8}" for _, h in metric_cols)
    width = len(header) + 2
    print("\n" + "=" * width)
    print(header)
    print("-" * width)
    for method, metrics in results.items():
        row = f"{method:<20}"
        for key, _ in metric_cols:
            val = metrics.get(key, 0.0)
            row += f" {val:>8.4f}" if key != "bleu" else f" {val:>8.2f}"
        print(row)
    print("=" * width)

    # Save final results
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "calibration_results.json", "w") as f:
        json.dump(results, f, indent=2)

    with open(output_dir / "sweep_results.json", "w") as f:
        json.dump(sweep_with_bleu, f, indent=2)

    # Save best lambda state
    torch.save(
        lambda_states[best_lambda],
        output_dir / "calibrated_head.pt",
    )

    logger.info(f"Final results saved to {output_dir}")


if __name__ == "__main__":
    main()
