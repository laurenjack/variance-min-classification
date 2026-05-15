"""Evaluate variance across training splits for Transformer models.

Loads all variance-mode models (model_d*_split*.pt), runs them on the
held-out in-distribution test chunk produced by
`load_iwslt14_variance_split` / `load_m2m100_iwslt14_variance_split`, and
computes:
  - Mean test loss across splits (per-token CE)
  - Jensen Gap E[log(q_bar[y] / q_j[y])] — the variance term in the
    label-only bias-variance decomposition.

With `--reference-logits` pointing at an M2M100 oracle distribution
(extracted on the same held-out chunk), the full distributional decomposition
CE(p, q_bar) = H(p) + KL(p || q_bar) + Jensen Gap is computed.

Output: evaluation.jsonl alongside the model files.

Usage:
    python -m jl.double_descent.transformer.variance_evaluation \\
        --model-path ./output/transformer_variance/03-01-1010 \\
        --data-path ./data/iwslt14.tokenized.de-en \\
        --num-splits 4 --samples-per-split 32000

    # Evaluate early-stop checkpoints instead
    python -m jl.double_descent.transformer.variance_evaluation \\
        --model-path ./output/transformer_variance/03-01-1010 \\
        --data-path ./data/iwslt14.tokenized.de-en \\
        --num-splits 4 --samples-per-split 32000 --early-stop
"""

import argparse
import json
import logging
import math
import os
import re
import tempfile
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from jl.double_descent.temperature_scaling import fit_temperature
from jl.double_descent.transformer.transformer_config import TDDConfig
from jl.double_descent.transformer.transformer_data import (
    M2M100TranslationDataset,
    M2M100Vocab,
    TranslationDataset,
    Vocab,
    build_vocab,
    collate_fn,
    load_iwslt14_variance_split,
    load_m2m100_iwslt14_variance_split,
    load_m2m100_split_ids,
)
from jl.double_descent.transformer.transformer_model import TransformerModel

logger = logging.getLogger(__name__)

EVAL_BATCH_SIZE = 32


def _prepare_batch(tgt, pad_idx):
    """Split target tensor into decoder input/output and compute padding mask."""
    tgt_input = tgt[:, :-1]
    target = tgt[:, 1:].contiguous()
    mask = target != pad_idx
    num_tokens = mask.sum().item()
    return tgt_input, target, mask, num_tokens


def _load_holdout_test_data(
    data_path, num_splits, samples_per_split, subsample_seed, holdout_samples,
):
    """Build vocab and DataLoader on the held-out variance chunk (BPE)."""
    train_dataset, _, test_dataset, vocab = load_iwslt14_variance_split(
        data_dir=data_path,
        split_id=0,  # any split_id is fine; held-out chunk is independent of it
        num_splits=num_splits,
        samples_per_split=samples_per_split,
        subsample_seed=subsample_seed,
        holdout_samples=holdout_samples,
    )
    del train_dataset
    loader = DataLoader(
        test_dataset, batch_size=EVAL_BATCH_SIZE, shuffle=False,
        collate_fn=partial(collate_fn, pad_idx=vocab.pad_idx),
    )
    return test_dataset, loader, vocab


def _load_valid_data(data_path):
    """Build vocab + DataLoader for the IWSLT valid split (used for TS fitting)."""
    from jl.double_descent.transformer.transformer_data import build_vocab, load_split
    vocab = build_vocab(data_path)
    valid_src, valid_tgt = load_split(data_path, "valid")
    valid_dataset = TranslationDataset(valid_src, valid_tgt, vocab)
    loader = DataLoader(
        valid_dataset, batch_size=EVAL_BATCH_SIZE, shuffle=False,
        collate_fn=partial(collate_fn, pad_idx=vocab.pad_idx),
    )
    return valid_dataset, loader, vocab


def _collect_logits(model, loader, pad_idx, device):
    """Forward the entire loader, returning flat (logits, labels) tensors for
    non-pad target positions only. Used to feed `fit_temperature`."""
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for src, tgt in loader:
            src, tgt = src.to(device), tgt.to(device)
            tgt_input, target, mask, _ = _prepare_batch(tgt, pad_idx)
            logits = model(src, tgt_input)  # [B, T, V]
            all_logits.append(logits[mask].cpu().float())
            all_labels.append(target[mask].cpu())
    return torch.cat(all_logits, dim=0), torch.cat(all_labels, dim=0)


def _load_holdout_m2m100_test_data(
    data_path, num_splits, samples_per_split, subsample_seed, holdout_samples,
):
    """Build M2M100 vocab and DataLoader on the held-out variance chunk."""
    train_dataset, _, test_dataset, vocab = load_m2m100_iwslt14_variance_split(
        data_dir=data_path,
        split_id=0,
        num_splits=num_splits,
        samples_per_split=samples_per_split,
        subsample_seed=subsample_seed,
        holdout_samples=holdout_samples,
    )
    del train_dataset
    loader = DataLoader(
        test_dataset, batch_size=EVAL_BATCH_SIZE, shuffle=False,
        collate_fn=partial(collate_fn, pad_idx=vocab.pad_idx),
    )
    return test_dataset, loader, vocab


def compute_decomposition(all_log_q_j_y, total_loss_sum, num_models, total_tokens):
    """Mean test loss + Jensen Gap from per-model log q_j[y] vectors (log-space)."""
    log_q_j_y = torch.stack(all_log_q_j_y)
    log_q_bar_y = torch.logsumexp(log_q_j_y, dim=0) - math.log(num_models)

    total_jensen = 0.0
    for j in range(num_models):
        total_jensen += (log_q_bar_y - log_q_j_y[j]).sum().item()

    mean_jensen_gap = total_jensen / (num_models * total_tokens)
    mean_test_loss = total_loss_sum / (num_models * total_tokens)
    return mean_test_loss, mean_jensen_gap


def discover_models(model_dir: str) -> Dict[int, List[Path]]:
    """Discover variance model files grouped by d_model."""
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


def _eval_worker(
    gpu_id, model_path_str, d_model, data_path, label_smoothing,
    num_splits, samples_per_split, subsample_seed, holdout_samples, output_file,
    apply_temperature_scaling: bool = False,
):
    """Worker process: evaluate one transformer model on one GPU on the
    held-out variance chunk. Saves log_q_j_y, total loss, total tokens.

    If apply_temperature_scaling is True, fits a scalar T on the IWSLT valid
    split first (Guo et al. protocol) and divides test logits by T before
    computing log-softmax. Per-model T + convergence diagnostics are written
    into the same output_file under key 'ts_diag'.
    """
    device = torch.device(f"cuda:{gpu_id}")
    config = TDDConfig()

    _, loader, vocab = _load_holdout_test_data(
        data_path, num_splits, samples_per_split, subsample_seed, holdout_samples,
    )
    model = load_model(
        Path(model_path_str), d_model, len(vocab),
        vocab.pad_idx, config, device,
    )

    ts_diag = None
    temperature = 1.0
    if apply_temperature_scaling:
        # Collect val logits, fit T, log convergence.
        _, val_loader, _ = _load_valid_data(data_path)
        val_logits, val_labels = _collect_logits(
            model, val_loader, vocab.pad_idx, device,
        )
        # Move to device for L-BFGS speed; chunk to avoid OOM on big vocab.
        val_logits = val_logits.to(device)
        val_labels = val_labels.to(device)
        temperature, ts_diag = fit_temperature(
            val_logits, val_labels, chunk_size=8192, return_diagnostics=True,
        )
        del val_logits, val_labels
        # Hard abort if the fit didn't converge — user opted for fail-fast.
        if not ts_diag["converged"]:
            raise RuntimeError(
                f"L-BFGS did not converge fitting T on val for "
                f"{model_path_str}: T={ts_diag['T']:.4f}, "
                f"|dCE/dT|={abs(ts_diag['final_grad']):.2e} "
                f"(tolerance 1e-4). Aborting."
            )

    criterion = nn.CrossEntropyLoss(
        ignore_index=vocab.pad_idx,
        label_smoothing=label_smoothing or 0.0,
    )

    all_log_q_j_y = []
    total_loss = 0.0
    total_tokens = 0
    min_log_q = float('inf')

    with torch.no_grad():
        for src, tgt in loader:
            src, tgt = src.to(device), tgt.to(device)
            tgt_input, target, mask, num_tokens = _prepare_batch(tgt, vocab.pad_idx)

            logits = model(src, tgt_input)
            if apply_temperature_scaling:
                logits = logits / temperature

            loss = criterion(
                logits.view(-1, logits.size(-1)),
                target.view(-1),
            )
            total_loss += loss.item() * num_tokens

            log_probs = F.log_softmax(logits, dim=-1)
            log_q_j_y = log_probs.gather(dim=-1, index=target.unsqueeze(-1)).squeeze(-1)
            masked_vals = log_q_j_y[mask]
            all_log_q_j_y.append(masked_vals.cpu())
            if masked_vals.numel() > 0:
                min_log_q = min(min_log_q, masked_vals.min().item())
            total_tokens += num_tokens

    torch.save({
        'log_q_j_y': torch.cat(all_log_q_j_y),
        'total_loss': torch.tensor(total_loss),
        'total_tokens': torch.tensor(total_tokens),
        'min_log_q': torch.tensor(min_log_q),
        'ts_diag': ts_diag,
    }, output_file)


def evaluate_d_model_parallel(
    model_paths: List[Path],
    d_model: int,
    data_path: str,
    config: TDDConfig,
    num_gpus: int,
    num_splits: int,
    samples_per_split: int,
    subsample_seed: int,
    holdout_samples: Optional[int] = None,
    apply_temperature_scaling: bool = False,
) -> Dict:
    """Mean test loss + Jensen Gap for one d_model, parallelised across GPUs.

    Splits are processed in batches of `num_gpus`; one process per GPU per
    batch. Works with any num_gpus <= num_models.
    """
    num_models = len(model_paths)
    logger.info(
        f"d_model={d_model}: evaluating {num_models} models on {num_gpus} GPUs "
        f"({(num_models + num_gpus - 1) // num_gpus} sequential batches)"
        f"{' (TS)' if apply_temperature_scaling else ''}"
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        # Process splits in batches of size num_gpus so we never have more
        # than num_gpus processes alive simultaneously.
        for batch_start in range(0, num_models, num_gpus):
            batch_end = min(batch_start + num_gpus, num_models)
            processes = []
            for offset, split_idx in enumerate(range(batch_start, batch_end)):
                gpu_id = offset
                out_f = os.path.join(tmp_dir, f"d{d_model}_s{split_idx}.pt")
                p = mp.Process(
                    target=_eval_worker,
                    args=(gpu_id, str(model_paths[split_idx]), d_model,
                          data_path, config.label_smoothing,
                          num_splits, samples_per_split, subsample_seed,
                          holdout_samples, out_f, apply_temperature_scaling),
                )
                p.start()
                processes.append(p)
            for p in processes:
                p.join()
            # Surface worker failures (e.g. TS non-convergence) instead of
            # silently dropping the model from the average.
            failed = [
                (i, p.exitcode)
                for i, p in enumerate(processes)
                if p.exitcode != 0
            ]
            if failed:
                raise RuntimeError(
                    f"d_model={d_model}: worker(s) failed (split offset, "
                    f"exitcode): {failed}. Check the parent stderr for the "
                    f"worker traceback (TS non-convergence is the most "
                    f"common cause)."
                )
            logger.info(
                f"  d_model={d_model}: batch splits "
                f"{batch_start}..{batch_end - 1} complete"
            )

        all_log_q_j_y = []
        total_loss_sum = 0.0
        total_tokens = 0
        min_log_qs = []
        ts_diags = []

        for split_idx in range(num_models):
            data = torch.load(
                os.path.join(tmp_dir, f"d{d_model}_s{split_idx}.pt"),
                weights_only=False,
            )
            all_log_q_j_y.append(data['log_q_j_y'])
            total_loss_sum += data['total_loss'].item()
            total_tokens = data['total_tokens'].item()
            min_log_qs.append(data['min_log_q'].item())
            if data.get('ts_diag') is not None:
                ts_diags.append({"split_id": split_idx, **data['ts_diag']})

        mean_test_loss, mean_jensen_gap = compute_decomposition(
            all_log_q_j_y, total_loss_sum, num_models, total_tokens,
        )

    min_log_qs.sort()
    logger.info(
        f"d_model={d_model}: min log q_j(y|x) per model: "
        f"{[f'{v:.2f}' for v in min_log_qs]}"
    )

    logger.info(
        f"d_model={d_model}: mean_test_loss={mean_test_loss:.4f}, "
        f"mean_jensen_gap={mean_jensen_gap:.6f}"
    )

    result = {
        "d_model": d_model,
        "mean_test_loss": round(mean_test_loss, 6),
        "mean_jensen_gap": round(mean_jensen_gap, 6),
        "num_models": num_models,
        "total_tokens": total_tokens,
    }
    if ts_diags:
        result["ts_temperatures"] = [round(d["T"], 6) for d in ts_diags]
        result["ts_all_converged"] = all(d["converged"] for d in ts_diags)
        result["ts_max_abs_grad"] = max(abs(d["final_grad"]) for d in ts_diags)
    return result


def _eval_single_model(
    gpu_id, model_path_str, d_model, data_path,
    num_splits, samples_per_split, subsample_seed, holdout_samples,
    apply_temperature_scaling: bool = False,
):
    """Evaluate one M2M100-vocab model on the held-out chunk; return
    (full_log_q_fp16 [N, V], ts_diag_or_None).

    If apply_temperature_scaling, fits T on the IWSLT-m2m100 valid split
    (Guo et al. protocol) and divides test logits by T before log-softmax.
    Aborts (RuntimeError) if the L-BFGS fit doesn't converge.
    """
    device = torch.device(f"cuda:{gpu_id}")
    config = TDDConfig()

    _, loader, vocab = _load_holdout_m2m100_test_data(
        data_path, num_splits, samples_per_split, subsample_seed, holdout_samples,
    )
    model = load_model(
        Path(model_path_str), d_model, len(vocab),
        vocab.pad_idx, config, device,
    )

    ts_diag = None
    temperature = 1.0
    if apply_temperature_scaling:
        # Collect val logits on the M2M100-vocab valid split.
        valid_dataset = M2M100TranslationDataset(
            *load_m2m100_split_ids(data_path, "valid"), vocab,
        )
        val_loader = DataLoader(
            valid_dataset, batch_size=EVAL_BATCH_SIZE, shuffle=False,
            collate_fn=partial(collate_fn, pad_idx=vocab.pad_idx),
        )
        val_logits, val_labels = _collect_logits(
            model, val_loader, vocab.pad_idx, device,
        )
        val_logits = val_logits.to(device)
        val_labels = val_labels.to(device)
        temperature, ts_diag = fit_temperature(
            val_logits, val_labels, chunk_size=8192, return_diagnostics=True,
        )
        del val_logits, val_labels
        if not ts_diag["converged"]:
            raise RuntimeError(
                f"L-BFGS did not converge fitting T for {model_path_str}: "
                f"T={ts_diag['T']:.4f}, |dCE/dT|={abs(ts_diag['final_grad']):.2e}. "
                f"Aborting."
            )

    all_log_q = []
    with torch.no_grad():
        for src, tgt in loader:
            src, tgt = src.to(device), tgt.to(device)
            tgt_input, target, mask, num_tokens = _prepare_batch(tgt, vocab.pad_idx)

            logits = model(src, tgt_input)
            if apply_temperature_scaling:
                logits = logits / temperature
            log_q = F.log_softmax(logits, dim=-1)
            # Cast to fp16 per-batch to halve the CPU accumulator memory.
            # The final torch.cat is then bounded by fp16 storage cost.
            all_log_q.append(log_q[mask].cpu().half())

    return torch.cat(all_log_q, dim=0), ts_diag


def compute_distributional_decomposition(
    all_log_q, ref_log_probs, ref_entropy, num_models, chunk_size: int = 8192,
):
    """Exact bias-variance decomposition with reference distribution p:
        CE(p, q_bar) = H(p) + KL(p || q_bar) + Jensen Gap

    Streams over chunks of positions so peak memory stays bounded even
    when the [N, V] log-prob tensors are tens of GB.

    Args:
        all_log_q: list of M fp16 [N, V] tensors (typically on CPU).
        ref_log_probs: fp16 [N, V] reference log-probabilities.
        ref_entropy: fp32 [N] per-position entropy.
        num_models: M.
        chunk_size: number of positions processed at a time. With
            V=18144 and M=4, a chunk of 8192 needs ~2.5 GB in fp32.
    """
    N = ref_log_probs.shape[0]

    mean_entropy = float(ref_entropy.float().mean().item())

    total_bias = 0.0
    total_variance = 0.0
    n_seen = 0

    log_M = math.log(num_models)

    for s in range(0, N, chunk_size):
        e = min(s + chunk_size, N)
        log_p_chunk = ref_log_probs[s:e].float()  # [c, V]
        p_chunk = log_p_chunk.exp()

        # Stack all M models' log_q on this chunk (upcast to fp32 per-chunk).
        log_q_chunks = torch.stack(
            [all_log_q[j][s:e].float() for j in range(num_models)],
            dim=0,
        )  # [M, c, V]

        log_q_bar = torch.logsumexp(log_q_chunks, dim=0) - log_M  # [c, V]

        # KL(p || q_bar) summed over positions in this chunk.
        bias_sum = (p_chunk * (log_p_chunk - log_q_bar)).sum().item()
        total_bias += bias_sum

        # Jensen Gap: mean_j sum_x p(x) (log q_bar(x) - log q_j(x))
        # summed over positions in this chunk.
        var_sum = 0.0
        for j in range(num_models):
            var_sum += (p_chunk * (log_q_bar - log_q_chunks[j])).sum().item()
        total_variance += var_sum

        n_seen += (e - s)

    mean_bias = total_bias / n_seen
    mean_variance = total_variance / (num_models * n_seen)
    mean_test_loss = mean_entropy + mean_bias + mean_variance

    return {
        "mean_test_loss": mean_test_loss,
        "entropy": mean_entropy,
        "bias": mean_bias,
        "variance": mean_variance,
    }


def evaluate_d_model_distributional(
    model_paths, d_model, data_path, config, num_gpus, ref_data,
    num_splits, samples_per_split, subsample_seed,
    holdout_samples: Optional[int] = None,
    apply_temperature_scaling: bool = False,
):
    """Distributional decomposition for one d_model using M2M100 reference."""
    num_models = len(model_paths)
    logger.info(
        f"d_model={d_model}: distributional eval with {num_models} models "
        f"on {num_gpus} GPUs{' (TS)' if apply_temperature_scaling else ''}"
    )

    # Keep ref log_probs in fp16 to avoid a 2x memory copy; the decomposition
    # upcasts per-chunk. ref_entropy is small (one float per position).
    ref_log_probs = ref_data['log_probs']  # fp16
    ref_entropy = ref_data['entropy'].float()

    all_log_q = []
    ts_diags = []
    for split_idx, model_path in enumerate(model_paths):
        gpu_id = split_idx % num_gpus
        logger.info(f"  Evaluating split {split_idx} on GPU {gpu_id}...")
        log_q, ts_diag = _eval_single_model(
            gpu_id, str(model_path), d_model, data_path,
            num_splits, samples_per_split, subsample_seed, holdout_samples,
            apply_temperature_scaling,
        )
        # _eval_single_model already returns fp16; keep it that way.
        all_log_q.append(log_q)
        if ts_diag is not None:
            ts_diags.append({"split_id": split_idx, **ts_diag})

    decomp = compute_distributional_decomposition(
        all_log_q, ref_log_probs, ref_entropy, num_models,
    )

    logger.info(
        f"d_model={d_model}: test_loss={decomp['mean_test_loss']:.4f}, "
        f"entropy={decomp['entropy']:.4f}, bias={decomp['bias']:.4f}, "
        f"variance={decomp['variance']:.6f}"
    )

    result = {
        "d_model": d_model,
        "mean_test_loss": round(decomp["mean_test_loss"], 6),
        "entropy": round(decomp["entropy"], 6),
        "bias": round(decomp["bias"], 6),
        "variance": round(decomp["variance"], 6),
        "num_models": num_models,
    }
    if ts_diags:
        result["ts_temperatures"] = [round(d["T"], 6) for d in ts_diags]
        result["ts_all_converged"] = all(d["converged"] for d in ts_diags)
        result["ts_max_abs_grad"] = max(abs(d["final_grad"]) for d in ts_diags)
    return result


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
        help="Run directory containing model_d*_split*.pt (or its early_stop/ "
             "subdir when --early-stop is passed)",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Directory containing preprocessed IWSLT data",
    )
    parser.add_argument(
        "--num-splits",
        type=int,
        default=4,
        help="num_splits used at training time (default 4). Determines which "
             "chunk is the held-out test set (chunk index == num_splits).",
    )
    parser.add_argument(
        "--samples-per-split",
        type=int,
        default=32000,
        help="samples_per_split used at training time (default 32000).",
    )
    parser.add_argument(
        "--subsample-seed",
        type=int,
        default=TDDConfig().subsample_seed,
        help="subsample_seed used at training time (default matches TDDConfig).",
    )
    parser.add_argument(
        "--holdout-samples",
        type=int,
        default=None,
        help="Size of the held-out test chunk. Default: all leftover indices "
             "after the num_splits training splits (use this for legacy runs "
             "that didn't reserve a fixed-size held-out chunk at training time).",
    )
    parser.add_argument(
        "--reference-logits",
        type=str,
        default=None,
        help="Path to reference_logits.pt for distributional bias-variance "
             "decomposition (M2M100 oracle on the held-out chunk).",
    )
    parser.add_argument(
        "--early-stop",
        action="store_true",
        help="Evaluate the early_stop/model_d*_split*.pt checkpoints instead. "
             "Output goes to early_stop/evaluation.jsonl.",
    )
    parser.add_argument(
        "--temperature-scaling",
        action="store_true",
        help="Fit a scalar temperature T per (d_model, split) on IWSLT valid "
             "(Guo et al. protocol) and apply it to test logits before the "
             "variance decomposition. Convergence diagnostics are logged per "
             "model and the per-d_model evaluation row gets ts_temperatures + "
             "ts_all_converged + ts_max_abs_grad. Output goes to "
             "evaluation_ts.jsonl alongside the non-TS one.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    if args.early_stop:
        model_dir = Path(args.model_path) / "early_stop"
        if not model_dir.is_dir():
            raise FileNotFoundError(f"Expected {model_dir} to exist")
    else:
        model_dir = Path(args.model_path)

    grouped = discover_models(str(model_dir))
    if not grouped:
        raise FileNotFoundError(
            f"No model_d*_split*.pt files found in {model_dir}"
        )
    logger.info(
        f"Found models for d_model values: {list(grouped.keys())} "
        f"({sum(len(v) for v in grouped.values())} total) "
        f"({'ES' if args.early_stop else 'FINAL'} checkpoints)"
    )

    config = TDDConfig()

    if args.reference_logits:
        # Distributional mode with M2M100 oracle
        logger.info(f"Loading reference logits from {args.reference_logits}")
        ref_data = torch.load(args.reference_logits, weights_only=True)
        logger.info(
            f"Reference: {ref_data['log_probs'].shape[0]} positions, "
            f"vocab_size={ref_data['log_probs'].shape[1]}, "
            f"mean entropy={ref_data['entropy'].float().mean():.4f}"
        )

        test_dataset, _, vocab = _load_holdout_m2m100_test_data(
            args.data_path, args.num_splits, args.samples_per_split,
            args.subsample_seed, args.holdout_samples,
        )
        logger.info(f"Held-out test set: {len(test_dataset)} examples, vocab: {len(vocab)} tokens")

        mp.set_start_method('spawn', force=True)
        num_gpus = torch.cuda.device_count()

        ref_output = model_dir / "reference"
        ref_output.mkdir(parents=True, exist_ok=True)
        eval_name = "evaluation_ts.jsonl" if args.temperature_scaling else "evaluation.jsonl"
        eval_file = ref_output / eval_name
        if eval_file.exists():
            eval_file.unlink()

        for d_model, model_paths in grouped.items():
            result = evaluate_d_model_distributional(
                model_paths, d_model, args.data_path, config, num_gpus, ref_data,
                args.num_splits, args.samples_per_split, args.subsample_seed,
                args.holdout_samples, args.temperature_scaling,
            )
            with open(eval_file, "a") as fh:
                fh.write(json.dumps(result) + "\n")

        logger.info(f"Distributional results written to {eval_file}")
        return

    # Label-only mode
    test_dataset, _, vocab = _load_holdout_test_data(
        args.data_path, args.num_splits, args.samples_per_split,
        args.subsample_seed, args.holdout_samples,
    )
    logger.info(f"Held-out test set: {len(test_dataset)} examples, vocab: {len(vocab)} tokens")

    mp.set_start_method('spawn', force=True)
    num_gpus = torch.cuda.device_count()

    out_name = "evaluation_ts.jsonl" if args.temperature_scaling else "evaluation.jsonl"
    output_path = model_dir / out_name
    if output_path.exists():
        output_path.unlink()

    for d_model, model_paths in grouped.items():
        result = evaluate_d_model_parallel(
            model_paths, d_model, args.data_path, config, num_gpus,
            args.num_splits, args.samples_per_split, args.subsample_seed,
            args.holdout_samples, args.temperature_scaling,
        )
        with open(output_path, "a") as fh:
            fh.write(json.dumps(result) + "\n")

    logger.info(f"Results written to {output_path}")


if __name__ == "__main__":
    main()
