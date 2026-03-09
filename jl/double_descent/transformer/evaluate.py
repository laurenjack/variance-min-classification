"""Evaluate variance across training splits for Transformer models.

Loads all variance-mode models (model_d*_split*.pt), runs them on the test set,
and computes:
  - Mean test loss across splits
  - Jensen Gap: E[log(q_bar[y] / q_j[y])] - the variance term in bias-variance decomposition

Output: evaluation.jsonl written alongside the model files.

With --temperature-scaling: fits a scalar temperature T per d_model on one randomly
chosen model (L-BFGS on test set NLL, batch-wise to avoid storing full logits),
then recomputes the full decomposition with softmax(logits/T). Results go to
temperature-scaled/evaluation.jsonl.

With --ece: computes per-token Expected Calibration Error (M=20 bins) using only
split 0 models on the test set. One model per d_model, parallelized across GPUs.
Output: ece.jsonl alongside the model files.

Usage:
    python -m jl.double_descent.transformer.evaluate \
        --model-path ./output/transformer_variance/03-01-1010 \
        --data-path ./data/iwslt14.tokenized.de-en

    python -m jl.double_descent.transformer.evaluate \
        --model-path ./output/transformer_variance/03-01-1010 \
        --data-path ./data/iwslt14.tokenized.de-en --temperature-scaling

    python -m jl.double_descent.transformer.evaluate \
        --model-path ./output/transformer_variance/03-01-1010 \
        --data-path ./data/iwslt14.tokenized.de-en --ece
"""

import argparse
import json
import logging
import os
import random
import re
import tempfile
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Dict, List

import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from jl.double_descent.transformer.transformer_config import TDDConfig
from jl.double_descent.transformer.transformer_data import (
    TranslationDataset,
    Vocab,
    build_vocab,
    collate_fn,
    load_split,
)
from jl.double_descent.transformer.transformer_model import TransformerModel

logger = logging.getLogger(__name__)

EVAL_BATCH_SIZE = 32
ECE_NUM_BINS = 20


def compute_ece(confidences, correct, num_bins=ECE_NUM_BINS):
    """Compute Expected Calibration Error with equal-width bins.

    Args:
        confidences: Tensor of max softmax probabilities per prediction.
        correct: Boolean tensor of whether each prediction was correct.
        num_bins: Number of equal-width bins in [0, 1].

    Returns:
        ECE as a float.
    """
    bin_boundaries = torch.linspace(0, 1, num_bins + 1)
    n = len(confidences)
    ece = 0.0
    for i in range(num_bins):
        lo, hi = bin_boundaries[i].item(), bin_boundaries[i + 1].item()
        mask = (confidences > lo) & (confidences <= hi)
        if i == 0:
            mask = mask | (confidences == lo)
        n_bin = mask.sum().item()
        if n_bin > 0:
            avg_confidence = confidences[mask].mean().item()
            avg_accuracy = correct[mask].float().mean().item()
            ece += (n_bin / n) * abs(avg_accuracy - avg_confidence)
    return ece


def _prepare_batch(tgt, pad_idx):
    """Split target tensor into decoder input/output and compute padding mask."""
    tgt_input = tgt[:, :-1]
    target = tgt[:, 1:].contiguous()
    mask = target != pad_idx
    num_tokens = mask.sum().item()
    return tgt_input, target, mask, num_tokens


def _load_test_data(data_path):
    """Build vocab and test DataLoader from preprocessed IWSLT data.

    Returns:
        Tuple of (test_dataset, test_loader, vocab).
    """
    vocab = build_vocab(data_path)
    test_src, test_tgt = load_split(data_path, "test")
    test_dataset = TranslationDataset(test_src, test_tgt, vocab)
    loader = DataLoader(
        test_dataset, batch_size=EVAL_BATCH_SIZE, shuffle=False,
        collate_fn=partial(collate_fn, pad_idx=vocab.pad_idx),
    )
    return test_dataset, loader, vocab


def compute_decomposition(all_q_j_y, total_loss_sum, num_models, total_tokens):
    """Compute mean test loss and Jensen Gap from per-model q_j[y] vectors.

    No Bessel's correction (divides by n, not n-1).

    Returns:
        Tuple of (mean_test_loss, mean_jensen_gap).
    """
    q_j_y = torch.stack(all_q_j_y)
    q_bar_y = q_j_y.mean(dim=0)

    log_q_bar_y = torch.log(q_bar_y + 1e-10)
    total_jensen = 0.0
    for j in range(num_models):
        log_q_j_y = torch.log(q_j_y[j] + 1e-10)
        total_jensen += (log_q_bar_y - log_q_j_y).sum().item()

    mean_jensen_gap = total_jensen / (num_models * total_tokens)
    mean_test_loss = total_loss_sum / (num_models * total_tokens)
    return mean_test_loss, mean_jensen_gap


def discover_models(model_dir: str) -> Dict[int, List[Path]]:
    """Discover variance model files grouped by d_model.

    Returns:
        Dict mapping d_model -> sorted list of model file Paths.
    """
    path = Path(model_dir)
    model_files = sorted(path.glob("model_d*_split*.pt"))

    grouped: Dict[int, List[Path]] = defaultdict(list)
    for f in model_files:
        match = re.match(r"model_d(\d+)_split(\d+)\.pt", f.name)
        if match:
            d_model = int(match.group(1))
            grouped[d_model].append(f)

    return dict(sorted(grouped.items()))


def load_model(
    model_path: Path,
    d_model: int,
    vocab_size: int,
    pad_idx: int,
    config: TDDConfig,
    device: torch.device,
) -> TransformerModel:
    """Instantiate model architecture and load saved weights."""
    model = TransformerModel(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        d_ff_multiplier=config.d_ff_multiplier,
        pad_idx=pad_idx,
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    return model


def evaluate_d_model(
    model_paths: List[Path],
    d_model: int,
    test_dataset: TranslationDataset,
    vocab: Vocab,
    config: TDDConfig,
    device: torch.device,
) -> Dict:
    """Compute mean test loss and Jensen Gap for one d_model.

    For each test batch:
      1. Forward pass all models, collect softmax distributions
      2. Compute q_bar = mean distribution across models
      3. Compute Jensen Gap: log(q_bar[y] / q_j[y]) for each model
    """
    num_models = len(model_paths)
    logger.info(f"d_model={d_model}: loading {num_models} models")

    models = [
        load_model(p, d_model, len(vocab), vocab.pad_idx, config, device)
        for p in model_paths
    ]

    loader = DataLoader(
        test_dataset,
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        collate_fn=partial(collate_fn, pad_idx=vocab.pad_idx),
    )

    criterion = nn.CrossEntropyLoss(
        ignore_index=vocab.pad_idx,
        label_smoothing=config.label_smoothing or 0.0,
    )

    total_loss_per_model = [0.0] * num_models
    total_jensen_gap = 0.0
    total_tokens = 0

    with torch.no_grad():
        for src, tgt in loader:
            src = src.to(device)
            tgt = tgt.to(device)
            tgt_input, target, mask, num_tokens = _prepare_batch(tgt, vocab.pad_idx)

            all_probs = []
            for j, model in enumerate(models):
                logits = model(src, tgt_input)  # [B, T, V]

                loss = criterion(
                    logits.view(-1, logits.size(-1)),
                    target.view(-1),
                )
                total_loss_per_model[j] += loss.item() * num_tokens

                probs = F.softmax(logits, dim=-1)  # [B, T, V]
                all_probs.append(probs)

            # Compute Jensen Gap: log(q_bar[y] / q_j[y])
            all_probs_t = torch.stack(all_probs, dim=0)  # [M, B, T, V]
            q_bar = all_probs_t.mean(dim=0)  # [B, T, V]
            log_q_bar = torch.log(q_bar + 1e-10)

            target_idx = target.unsqueeze(-1)  # [B, T, 1]
            log_q_bar_y = log_q_bar.gather(dim=-1, index=target_idx).squeeze(-1)  # [B, T]

            batch_jensen = 0.0
            for j in range(num_models):
                log_q_j = torch.log(all_probs_t[j] + 1e-10)
                log_q_j_y = log_q_j.gather(dim=-1, index=target_idx).squeeze(-1)  # [B, T]
                jensen_per_token = log_q_bar_y - log_q_j_y  # [B, T]
                batch_jensen += (jensen_per_token * mask).sum().item()

            total_jensen_gap += batch_jensen / num_models
            total_tokens += num_tokens

    del models
    torch.cuda.empty_cache()

    mean_test_loss = (
        sum(total_loss_per_model) / (num_models * total_tokens)
        if total_tokens > 0
        else 0.0
    )
    mean_jensen_gap = total_jensen_gap / total_tokens if total_tokens > 0 else 0.0

    logger.info(
        f"d_model={d_model}: mean_test_loss={mean_test_loss:.4f}, "
        f"mean_jensen_gap={mean_jensen_gap:.6f}"
    )

    return {
        "d_model": d_model,
        "mean_test_loss": round(mean_test_loss, 6),
        "mean_jensen_gap": round(mean_jensen_gap, 6),
        "num_models": num_models,
        "total_tokens": total_tokens,
    }


def fit_temperature(model, test_loader, pad_idx, device):
    """Fit scalar temperature T via L-BFGS to minimize NLL on test set.

    Processes batches within the L-BFGS closure to avoid storing full logits
    (which can be huge for large vocab). Model forward pass runs with no_grad;
    only the temperature scalar gets gradients.
    """
    temperature = nn.Parameter(torch.ones(1, device=device))
    optimizer = torch.optim.LBFGS([temperature], lr=0.01, max_iter=50)

    def closure():
        optimizer.zero_grad()
        total_loss = 0.0
        total_tokens = 0
        for src, tgt in test_loader:
            src, tgt = src.to(device), tgt.to(device)
            tgt_input, target, _, num_tokens = _prepare_batch(tgt, pad_idx)

            with torch.no_grad():
                logits = model(src, tgt_input)

            batch_loss = F.cross_entropy(
                (logits / temperature).view(-1, logits.size(-1)),
                target.view(-1),
                ignore_index=pad_idx,
                reduction='sum',
            )
            batch_loss.backward()
            total_loss += batch_loss.item()
            total_tokens += num_tokens

        if total_tokens > 0:
            temperature.grad.div_(total_tokens)
        return torch.tensor(total_loss / total_tokens if total_tokens > 0 else 0.0)

    optimizer.step(closure)
    return temperature.item()


def _ts_eval_worker(gpu_id, model_path_str, d_model, temperature,
                    data_path, label_smoothing, output_file):
    """Worker process: evaluate one transformer model with temperature scaling.

    Loads model on assigned GPU, runs forward pass with temperature T,
    saves q_j[y] (per non-pad token) and loss to output_file.
    """
    device = torch.device(f"cuda:{gpu_id}")
    config = TDDConfig()

    _, loader, vocab = _load_test_data(data_path)
    model = load_model(
        Path(model_path_str), d_model, len(vocab),
        vocab.pad_idx, config, device,
    )

    criterion = nn.CrossEntropyLoss(
        ignore_index=vocab.pad_idx,
        label_smoothing=label_smoothing or 0.0,
    )

    all_q_j_y = []
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for src, tgt in loader:
            src, tgt = src.to(device), tgt.to(device)
            tgt_input, target, mask, num_tokens = _prepare_batch(tgt, vocab.pad_idx)

            logits = model(src, tgt_input)

            loss = criterion(
                (logits / temperature).view(-1, logits.size(-1)),
                target.view(-1),
            )
            total_loss += loss.item() * num_tokens

            probs = F.softmax(logits / temperature, dim=-1)
            q_j_y = probs.gather(dim=-1, index=target.unsqueeze(-1)).squeeze(-1)
            all_q_j_y.append(q_j_y[mask].cpu())
            total_tokens += num_tokens

    torch.save({
        'q_j_y': torch.cat(all_q_j_y),
        'total_loss': torch.tensor(total_loss),
        'total_tokens': torch.tensor(total_tokens),
    }, output_file)


def _ece_worker(gpu_id, model_path_str, d_model, data_path, output_file):
    """Worker process: compute per-token ECE inputs for one transformer model.

    Loads model on assigned GPU, runs forward pass on test set,
    saves per-token confidences and correctness (non-pad only) to output_file.
    """
    device = torch.device(f"cuda:{gpu_id}")
    config = TDDConfig()

    _, loader, vocab = _load_test_data(data_path)
    model = load_model(
        Path(model_path_str), d_model, len(vocab),
        vocab.pad_idx, config, device,
    )

    all_confidences = []
    all_correct = []

    with torch.no_grad():
        for src, tgt in loader:
            src, tgt = src.to(device), tgt.to(device)
            tgt_input, target, mask, _ = _prepare_batch(tgt, vocab.pad_idx)

            logits = model(src, tgt_input)
            probs = F.softmax(logits, dim=-1)
            max_probs, predictions = probs.max(dim=-1)

            all_confidences.append(max_probs[mask].cpu())
            all_correct.append((predictions == target)[mask].cpu())

    torch.save({
        'confidences': torch.cat(all_confidences),
        'correct': torch.cat(all_correct),
    }, output_file)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Evaluate variance across training splits for Transformer models"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Directory containing model_d*_split*.pt files",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Directory containing preprocessed IWSLT data",
    )
    parser.add_argument(
        "--temperature-scaling",
        action="store_true",
        help="Fit per-d_model temperature and recompute bias-variance decomposition",
    )
    parser.add_argument(
        "--ece",
        action="store_true",
        help="Compute per-token Expected Calibration Error using split 0 models",
    )
    args = parser.parse_args()

    if args.temperature_scaling and args.ece:
        parser.error("--temperature-scaling and --ece are mutually exclusive")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    grouped = discover_models(args.model_path)
    if not grouped:
        raise FileNotFoundError(
            f"No model_d*_split*.pt files found in {args.model_path}"
        )
    logger.info(
        f"Found models for d_model values: {list(grouped.keys())} "
        f"({sum(len(v) for v in grouped.values())} total)"
    )

    config = TDDConfig()

    test_dataset, test_loader, vocab = _load_test_data(args.data_path)
    logger.info(f"Test set: {len(test_dataset)} examples, vocab: {len(vocab)} tokens")

    if args.ece:
        mp.set_start_method('spawn', force=True)

        ece_file = Path(args.model_path) / "ece.jsonl"
        if ece_file.exists():
            ece_file.unlink()

        d_model_values = sorted(grouped.keys())
        num_gpus = max(1, torch.cuda.device_count())

        for batch_start in range(0, len(d_model_values), num_gpus):
            batch_d = d_model_values[batch_start:batch_start + num_gpus]
            logger.info(f"ECE batch: d_model={batch_d}")

            with tempfile.TemporaryDirectory() as tmp_dir:
                processes = []
                for gpu_id, d_model in enumerate(batch_d):
                    split0 = [p for p in grouped[d_model]
                              if '_split0.pt' in p.name][0]
                    out_f = os.path.join(tmp_dir, f"d{d_model}.pt")
                    p = mp.Process(
                        target=_ece_worker,
                        args=(gpu_id, str(split0), d_model,
                              args.data_path, out_f),
                    )
                    p.start()
                    processes.append(p)

                for p in processes:
                    p.join()

                for d_model in batch_d:
                    data = torch.load(
                        os.path.join(tmp_dir, f"d{d_model}.pt"),
                        weights_only=True,
                    )
                    ece = compute_ece(data['confidences'], data['correct'])
                    result = {
                        "d_model": d_model,
                        "ece": round(ece, 6),
                        "num_tokens": len(data['confidences']),
                    }
                    logger.info(f"d_model={d_model}: ECE={ece:.6f}")
                    with open(ece_file, "a") as fh:
                        fh.write(json.dumps(result) + "\n")

        logger.info(f"ECE results: {ece_file}")

    elif args.temperature_scaling:
        mp.set_start_method('spawn', force=True)

        ts_output = Path(args.model_path) / "temperature-scaled"
        ts_output.mkdir(parents=True, exist_ok=True)
        ts_eval_file = ts_output / "evaluation.jsonl"
        if ts_eval_file.exists():
            ts_eval_file.unlink()

        d_model_values = sorted(grouped.keys())
        # 8 splits per d_model = 8 GPUs, one d_model at a time
        num_gpus = torch.cuda.device_count()

        for d_model in d_model_values:
            model_paths = grouped[d_model]
            num_models = len(model_paths)

            # Fit temperature on one random model
            rng = random.Random(d_model)
            calib_idx = rng.randrange(num_models)

            logger.info(
                f"d_model={d_model}: fitting temperature on "
                f"{model_paths[calib_idx].name}"
            )

            calib_model = load_model(
                model_paths[calib_idx], d_model, len(vocab),
                vocab.pad_idx, config, device,
            )
            temp_val = fit_temperature(
                calib_model, test_loader, vocab.pad_idx, device,
            )
            del calib_model
            torch.cuda.empty_cache()
            logger.info(f"d_model={d_model}: fitted T={temp_val:.4f}")

            # Spawn parallel workers across GPUs
            with tempfile.TemporaryDirectory() as tmp_dir:
                processes = []
                for split_idx, model_path in enumerate(model_paths):
                    gpu_id = split_idx % num_gpus
                    out_f = os.path.join(tmp_dir, f"d{d_model}_s{split_idx}.pt")
                    p = mp.Process(
                        target=_ts_eval_worker,
                        args=(gpu_id, str(model_path), d_model, temp_val,
                              args.data_path, config.label_smoothing, out_f),
                    )
                    p.start()
                    processes.append(p)

                for p in processes:
                    p.join()

                # Collect worker results and compute decomposition
                all_q_j_y = []
                total_loss_sum = 0.0
                total_tokens = 0

                for split_idx in range(num_models):
                    data = torch.load(
                        os.path.join(tmp_dir, f"d{d_model}_s{split_idx}.pt"),
                        weights_only=True,
                    )
                    all_q_j_y.append(data['q_j_y'])
                    total_loss_sum += data['total_loss'].item()
                    total_tokens = data['total_tokens'].item()

                mean_test_loss, mean_jensen_gap = compute_decomposition(
                    all_q_j_y, total_loss_sum, num_models, total_tokens,
                )

                result = {
                    "d_model": d_model,
                    "mean_test_loss": round(mean_test_loss, 6),
                    "mean_jensen_gap": round(mean_jensen_gap, 6),
                    "temperature": round(temp_val, 6),
                    "num_models": num_models,
                    "total_tokens": int(total_tokens),
                }

                logger.info(
                    f"d_model={d_model}: T={temp_val:.4f}, "
                    f"mean_test_loss={mean_test_loss:.4f}, "
                    f"mean_jensen_gap={mean_jensen_gap:.6f}"
                )

                with open(ts_eval_file, "a") as fh:
                    fh.write(json.dumps(result) + "\n")

        logger.info(f"Temperature-scaled results: {ts_eval_file}")

    else:
        output_path = Path(args.model_path) / "evaluation.jsonl"
        if output_path.exists():
            output_path.unlink()

        for d_model, model_paths in grouped.items():
            result = evaluate_d_model(
                model_paths, d_model, test_dataset, vocab, config, device
            )
            with open(output_path, "a") as fh:
                fh.write(json.dumps(result) + "\n")

        logger.info(f"Results written to {output_path}")


if __name__ == "__main__":
    main()
