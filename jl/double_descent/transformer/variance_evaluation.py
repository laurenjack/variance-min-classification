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

Usage:
    python -m jl.double_descent.transformer.variance_evaluation \
        --model-path ./output/transformer_variance/03-01-1010 \
        --data-path ./data/iwslt14.tokenized.de-en

    python -m jl.double_descent.transformer.variance_evaluation \
        --model-path ./output/transformer_variance/03-01-1010 \
        --data-path ./data/iwslt14.tokenized.de-en --temperature-scaling
"""

import argparse
import json
import logging
import math
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
    M2M100TranslationDataset,
    M2M100Vocab,
    TranslationDataset,
    Vocab,
    build_vocab,
    collate_fn,
    load_m2m100_split_ids,
    load_split,
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


def compute_decomposition(all_log_q_j_y, total_loss_sum, num_models, total_tokens):
    """Compute mean test loss and Jensen Gap from per-model log q_j[y] vectors.

    Uses log-space arithmetic for numerical stability.

    Returns:
        Tuple of (mean_test_loss, mean_jensen_gap).
    """
    log_q_j_y = torch.stack(all_log_q_j_y)  # [M, N]

    # log(q_bar(y)) = logsumexp(log_q_j(y), dim=0) - log(M)
    log_q_bar_y = torch.logsumexp(log_q_j_y, dim=0) - math.log(num_models)

    total_jensen = 0.0
    for j in range(num_models):
        total_jensen += (log_q_bar_y - log_q_j_y[j]).sum().item()

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


def _eval_worker(gpu_id, model_path_str, d_model, data_path, label_smoothing, output_file):
    """Worker process: evaluate one transformer model on one GPU.

    Saves log_q_j_y (per non-pad token), total loss, and total tokens to output_file.
    Uses log_softmax for numerical stability.
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

    all_log_q_j_y = []
    total_loss = 0.0
    total_tokens = 0
    min_log_q = float('inf')

    with torch.no_grad():
        for src, tgt in loader:
            src, tgt = src.to(device), tgt.to(device)
            tgt_input, target, mask, num_tokens = _prepare_batch(tgt, vocab.pad_idx)

            logits = model(src, tgt_input)

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
    }, output_file)


def evaluate_d_model_parallel(
    model_paths: List[Path],
    d_model: int,
    data_path: str,
    config: TDDConfig,
    num_gpus: int,
) -> Dict:
    """Compute mean test loss and Jensen Gap for one d_model using parallel GPU workers.

    Spawns one process per model (one per GPU), collects log_q_j_y results,
    and computes the decomposition in log space.
    """
    num_models = len(model_paths)
    logger.info(f"d_model={d_model}: evaluating {num_models} models on {num_gpus} GPUs")

    with tempfile.TemporaryDirectory() as tmp_dir:
        processes = []
        for split_idx, model_path in enumerate(model_paths):
            gpu_id = split_idx % num_gpus
            out_f = os.path.join(tmp_dir, f"d{d_model}_s{split_idx}.pt")
            p = mp.Process(
                target=_eval_worker,
                args=(gpu_id, str(model_path), d_model,
                      data_path, config.label_smoothing, out_f),
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        # Collect results
        all_log_q_j_y = []
        total_loss_sum = 0.0
        total_tokens = 0
        min_log_qs = []

        for split_idx in range(num_models):
            data = torch.load(
                os.path.join(tmp_dir, f"d{d_model}_s{split_idx}.pt"),
                weights_only=True,
            )
            all_log_q_j_y.append(data['log_q_j_y'])
            total_loss_sum += data['total_loss'].item()
            total_tokens = data['total_tokens'].item()
            min_log_qs.append(data['min_log_q'].item())

        mean_test_loss, mean_jensen_gap = compute_decomposition(
            all_log_q_j_y, total_loss_sum, num_models, total_tokens,
        )

    # DEBUG: print smallest log q values across all models
    min_log_qs.sort()
    logger.info(
        f"d_model={d_model}: min log q_j(y|x) per model: "
        f"{[f'{v:.2f}' for v in min_log_qs]}"
    )

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

    all_log_q_j_y = []
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

            log_probs = F.log_softmax(logits / temperature, dim=-1)
            log_q_j_y = log_probs.gather(dim=-1, index=target.unsqueeze(-1)).squeeze(-1)
            all_log_q_j_y.append(log_q_j_y[mask].cpu())
            total_tokens += num_tokens

    torch.save({
        'log_q_j_y': torch.cat(all_log_q_j_y),
        'total_loss': torch.tensor(total_loss),
        'total_tokens': torch.tensor(total_tokens),
    }, output_file)


def _load_m2m100_test_data(data_path):
    """Build M2M100 vocab and test DataLoader from M2M100-preprocessed IWSLT data."""
    vocab = M2M100Vocab(str(Path(data_path) / "vocab_mapping.json"))
    test_src, test_tgt = load_m2m100_split_ids(data_path, "test")
    test_dataset = M2M100TranslationDataset(test_src, test_tgt, vocab)
    loader = DataLoader(
        test_dataset, batch_size=EVAL_BATCH_SIZE, shuffle=False,
        collate_fn=partial(collate_fn, pad_idx=vocab.pad_idx),
    )
    return test_dataset, loader, vocab


def _distributional_eval_worker(gpu_id, model_path_str, d_model, data_path,
                                ref_token_ids, ref_log_probs, output_file):
    """Worker: evaluate one model, gather log q_j at the top-K reference positions.

    Saves log_q_j_at_ref [N_positions, K] to output_file.
    """
    device = torch.device(f"cuda:{gpu_id}")
    config = TDDConfig()

    _, loader, vocab = _load_m2m100_test_data(data_path)
    model = load_model(
        Path(model_path_str), d_model, len(vocab),
        vocab.pad_idx, config, device,
    )

    all_log_q_at_ref = []
    total_loss = 0.0
    total_tokens = 0
    position_offset = 0

    with torch.no_grad():
        for src, tgt in loader:
            src, tgt = src.to(device), tgt.to(device)
            tgt_input, target, mask, num_tokens = _prepare_batch(tgt, vocab.pad_idx)

            logits = model(src, tgt_input)
            log_q = F.log_softmax(logits, dim=-1)  # [batch, seq_len, vocab_size]

            # Compute loss at ground truth for test_loss metric
            log_q_y = log_q.gather(dim=-1, index=target.unsqueeze(-1)).squeeze(-1)
            total_loss += -log_q_y[mask].sum().item()

            # For each non-padding position, gather log q at the reference top-K tokens
            for b in range(src.size(0)):
                b_mask = mask[b]  # [seq_len]
                n_pos = b_mask.sum().item()
                if n_pos == 0:
                    continue

                # Get reference token IDs for these positions
                ref_tids = ref_token_ids[position_offset:position_offset + n_pos].to(device)
                # ref_tids: [n_pos, K] compact token IDs

                # Gather log q at those positions
                b_log_q = log_q[b][b_mask]  # [n_pos, vocab_size]
                b_log_q_at_ref = b_log_q.gather(dim=-1, index=ref_tids.long())  # [n_pos, K]

                all_log_q_at_ref.append(b_log_q_at_ref.cpu())
                position_offset += n_pos
                total_tokens += n_pos

    torch.save({
        'log_q_at_ref': torch.cat(all_log_q_at_ref, dim=0),  # [total_positions, K]
        'total_loss': torch.tensor(total_loss),
        'total_tokens': torch.tensor(total_tokens),
    }, output_file)


def compute_distributional_decomposition(all_log_q_at_ref, ref_log_probs, ref_entropy,
                                         total_loss_sum, num_models, total_tokens):
    """Compute entropy, bias, and variance from distributional data.

    Args:
        all_log_q_at_ref: list of [N, K] tensors, one per model
        ref_log_probs: [N, K] reference log-probabilities
        ref_entropy: [N] per-position entropy from full distribution
        total_loss_sum: sum of per-model total losses
        num_models: M
        total_tokens: N

    Returns:
        dict with mean_test_loss, entropy, bias, variance
    """
    log_q_j = torch.stack(all_log_q_at_ref)  # [M, N, K]

    # Ensemble: q_bar
    log_q_bar = torch.logsumexp(log_q_j, dim=0) - math.log(num_models)  # [N, K]

    # Reference distribution
    log_p = ref_log_probs  # [N, K]
    p = log_p.exp()        # [N, K]

    # Entropy: H(p) from full distribution (pre-computed, not top-K approximation)
    mean_entropy = ref_entropy.mean().item()

    # Bias: KL(p || q_bar) = Σ_x p(x) [log p(x) - log q_bar(x)]
    bias_per_position = (p * (log_p - log_q_bar)).sum(dim=-1)  # [N]
    mean_bias = bias_per_position.mean().item()

    # Variance (Jensen Gap): E_j[Σ_x p(x) log(q_bar(x) / q_j(x))]
    total_variance = 0.0
    for j in range(num_models):
        variance_j = (p * (log_q_bar - log_q_j[j])).sum(dim=-1)  # [N]
        total_variance += variance_j.mean().item()
    mean_variance = total_variance / num_models

    mean_test_loss = total_loss_sum / (num_models * total_tokens)

    return {
        "mean_test_loss": mean_test_loss,
        "entropy": mean_entropy,
        "bias": mean_bias,
        "variance": mean_variance,
    }


def evaluate_d_model_distributional(
    model_paths, d_model, data_path, config, num_gpus, ref_data,
):
    """Compute distributional decomposition for one d_model using reference logits."""
    num_models = len(model_paths)
    logger.info(f"d_model={d_model}: distributional eval with {num_models} models on {num_gpus} GPUs")

    ref_token_ids = ref_data['token_ids']    # [total_positions, K]
    ref_log_probs = ref_data['log_probs'].float()  # [total_positions, K]
    ref_entropy = ref_data['entropy'].float()      # [total_positions]

    with tempfile.TemporaryDirectory() as tmp_dir:
        processes = []
        for split_idx, model_path in enumerate(model_paths):
            gpu_id = split_idx % num_gpus
            out_f = os.path.join(tmp_dir, f"d{d_model}_s{split_idx}.pt")
            p = mp.Process(
                target=_distributional_eval_worker,
                args=(gpu_id, str(model_path), d_model, data_path,
                      ref_token_ids, ref_log_probs, out_f),
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        # Collect results
        all_log_q_at_ref = []
        total_loss_sum = 0.0
        total_tokens = 0

        for split_idx in range(num_models):
            data = torch.load(
                os.path.join(tmp_dir, f"d{d_model}_s{split_idx}.pt"),
                weights_only=True,
            )
            all_log_q_at_ref.append(data['log_q_at_ref'])
            total_loss_sum += data['total_loss'].item()
            total_tokens = data['total_tokens'].item()

    decomp = compute_distributional_decomposition(
        all_log_q_at_ref, ref_log_probs, ref_entropy,
        total_loss_sum, num_models, total_tokens,
    )

    logger.info(
        f"d_model={d_model}: test_loss={decomp['mean_test_loss']:.4f}, "
        f"entropy={decomp['entropy']:.4f}, bias={decomp['bias']:.4f}, "
        f"variance={decomp['variance']:.6f}"
    )

    return {
        "d_model": d_model,
        "mean_test_loss": round(decomp["mean_test_loss"], 6),
        "entropy": round(decomp["entropy"], 6),
        "bias": round(decomp["bias"], 6),
        "variance": round(decomp["variance"], 6),
        "num_models": num_models,
        "total_tokens": total_tokens,
    }


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
        "--reference-logits",
        type=str,
        default=None,
        help="Path to reference_logits.pt for distributional bias-variance decomposition",
    )
    args = parser.parse_args()

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

    if args.reference_logits:
        # Distributional mode: use M2M100 reference logits
        logger.info(f"Loading reference logits from {args.reference_logits}")
        ref_data = torch.load(args.reference_logits, weights_only=True)
        logger.info(
            f"Reference: {ref_data['token_ids'].shape[0]} positions, "
            f"top-{ref_data['top_k']}, mean entropy={ref_data['entropy'].float().mean():.4f}"
        )

        test_dataset, test_loader, vocab = _load_m2m100_test_data(args.data_path)
        logger.info(f"Test set: {len(test_dataset)} examples, vocab: {len(vocab)} tokens")

        mp.set_start_method('spawn', force=True)
        num_gpus = torch.cuda.device_count()

        ref_output = Path(args.model_path) / "reference"
        ref_output.mkdir(parents=True, exist_ok=True)
        eval_file = ref_output / "evaluation.jsonl"
        if eval_file.exists():
            eval_file.unlink()

        for d_model, model_paths in grouped.items():
            result = evaluate_d_model_distributional(
                model_paths, d_model, args.data_path, config, num_gpus, ref_data,
            )
            with open(eval_file, "a") as fh:
                fh.write(json.dumps(result) + "\n")

        logger.info(f"Distributional results written to {eval_file}")
        return

    test_dataset, test_loader, vocab = _load_test_data(args.data_path)
    logger.info(f"Test set: {len(test_dataset)} examples, vocab: {len(vocab)} tokens")

    if args.temperature_scaling:
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
                all_log_q_j_y = []
                total_loss_sum = 0.0
                total_tokens = 0

                for split_idx in range(num_models):
                    data = torch.load(
                        os.path.join(tmp_dir, f"d{d_model}_s{split_idx}.pt"),
                        weights_only=True,
                    )
                    all_log_q_j_y.append(data['log_q_j_y'])
                    total_loss_sum += data['total_loss'].item()
                    total_tokens = data['total_tokens'].item()

                mean_test_loss, mean_jensen_gap = compute_decomposition(
                    all_log_q_j_y, total_loss_sum, num_models, total_tokens,
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
        mp.set_start_method('spawn', force=True)
        num_gpus = torch.cuda.device_count()

        output_path = Path(args.model_path) / "evaluation.jsonl"
        if output_path.exists():
            output_path.unlink()

        for d_model, model_paths in grouped.items():
            result = evaluate_d_model_parallel(
                model_paths, d_model, args.data_path, config, num_gpus
            )
            with open(output_path, "a") as fh:
                fh.write(json.dumps(result) + "\n")

        logger.info(f"Results written to {output_path}")


if __name__ == "__main__":
    main()
