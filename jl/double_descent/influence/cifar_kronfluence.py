#!/usr/bin/env python3
"""Kronfluence (EK-FAC influence functions) sweep for the CIFAR resnet18 runs.

For each model_k*.pt in --model-dir, fits EK-FAC factors on the noisy
training data, computes the pairwise [N_test, N_train] influence-score
matrix against the clean test set, and reports mislabeled-vs-clean share
metrics (B = signed/abs, C = abs/abs) -- per test point, then averaged.

Usage:
    pip install kronfluence
    python -m jl.double_descent.influence.cifar_kronfluence \\
        --model-dir   data/resnet18/figure_2 \\
        --output-dir  data/resnet18/figure_2/kronfluence \\
        --data-path   ./data
"""

import argparse
import json
import logging
import re
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from jl.double_descent.influence.decompose import build_mislabel_mask
from jl.double_descent.resnet18.resnet18_data import (
    NoisyCIFAR10,
    compute_val_split_indices,
)
from jl.double_descent.resnet18.resnet18k import make_resnet18k

logger = logging.getLogger(__name__)


# ----------------------------- Kronfluence Task --------------------------- #


def _make_task():
    """Build the Kronfluence Task object lazily so kronfluence isn't a hard import dep."""
    from kronfluence.task import Task

    class CIFARClassificationTask(Task):
        """Standard label-CE task for CIFAR.

        compute_train_loss: per-batch SUM CE loss (Kronfluence convention).
        compute_measurement: same form; the test-time scalar we measure
            influence on.
        """

        def compute_train_loss(self, batch, model, sample: bool = False):
            images, labels = batch
            logits = model(images)
            if sample:
                # For Fisher estimation, draw "soft" labels from the model.
                with torch.no_grad():
                    probs = F.softmax(logits.detach(), dim=-1)
                    sampled = torch.multinomial(probs, num_samples=1).squeeze(-1)
                return F.cross_entropy(logits, sampled, reduction="sum")
            return F.cross_entropy(logits, labels, reduction="sum")

        def compute_measurement(self, batch, model):
            images, labels = batch
            logits = model(images)
            return F.cross_entropy(logits, labels, reduction="sum")

        def get_influence_tracked_modules(self) -> List[str]:
            # Returning None tells Kronfluence to auto-track all nn.Linear /
            # nn.Conv2d layers.  Explicit list available if we ever need to
            # restrict to e.g. the final layer.
            return None

        def get_attention_mask(self, batch):
            return None

    return CIFARClassificationTask()


# ----------------------------- per-k pipeline ----------------------------- #


def _share_metrics_from_scores(scores: torch.Tensor, mis_mask_bool: torch.Tensor):
    """Compute share metrics B (signed / abs) and C (abs / abs).

    scores: [N_test, N_train] float tensor.
    mis_mask_bool: [N_train] bool tensor, True = mislabeled.
    Returns dict of summary stats.
    """
    if scores.dim() != 2:
        raise ValueError(f"Expected 2D scores, got {scores.shape}")
    scores = scores.float()
    mis = mis_mask_bool.to(scores.device)

    abs_scores = scores.abs()
    total_abs_per_t = abs_scores.sum(dim=1)                # [N_test]
    safe_total = total_abs_per_t.clamp_min(1e-30)

    # B: signed mislabeled / abs total, per test t, then mean
    mis_signed_per_t = scores[:, mis].sum(dim=1)            # [N_test]
    share_B_per_t = mis_signed_per_t / safe_total
    share_B = float(share_B_per_t.mean().item())

    # C: abs mislabeled / abs total, per test t, then mean
    mis_abs_per_t = abs_scores[:, mis].sum(dim=1)
    share_C_per_t = mis_abs_per_t / safe_total
    share_C = float(share_C_per_t.mean().item())

    # Diagnostics
    mean_signed_total_per_t = scores.sum(dim=1)
    return {
        "share_B_signed_over_abs": share_B,
        "share_C_abs_over_abs": share_C,
        "mean_signed_mislabeled_per_test": float(mis_signed_per_t.mean().item()),
        "mean_signed_total_per_test": float(mean_signed_total_per_t.mean().item()),
        "mean_abs_total_per_test": float(total_abs_per_t.mean().item()),
        "fraction_mislabeled_in_train": float(mis.float().mean().item()),
    }


def run_for_k(
    k: int,
    model_path: Path,
    train_dataset,
    test_dataset,
    mislabel_mask: np.ndarray,
    output_dir: Path,
    device: torch.device,
    factor_batch_size: int = 128,
    query_batch_size: int = 64,
    bf16: bool = False,
) -> dict:
    """Run the full Kronfluence pipeline for one (k, model checkpoint).

    When bf16=True, FactorArguments / ScoreArguments use amp_dtype=bfloat16.
    Eigendecomposition stays FP64 (kronfluence default) for numerical
    stability.
    """
    from kronfluence.analyzer import Analyzer, prepare_model
    from kronfluence.arguments import FactorArguments, ScoreArguments

    logger.info(f"--- k={k} ({model_path}) ---")
    model = make_resnet18k(k=k, num_classes=10).to(device)
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )
    model.eval()

    task = _make_task()
    model_prepared = prepare_model(model=model, task=task)

    analysis_name = f"figure_2_k{k}"
    analyzer = Analyzer(
        analysis_name=analysis_name,
        model=model_prepared,
        task=task,
        output_dir=str(output_dir / "kronfluence_internal"),
    )

    amp_dtype = torch.bfloat16 if bf16 else None

    # Factor computation phase
    factors_name = f"ekfac_k{k}"
    factor_args = FactorArguments(strategy="ekfac", amp_dtype=amp_dtype)
    logger.info(
        f"  fitting EK-FAC factors (strategy=ekfac, batch={factor_batch_size}, "
        f"amp_dtype={amp_dtype})..."
    )
    analyzer.fit_all_factors(
        factors_name=factors_name,
        dataset=train_dataset,
        per_device_batch_size=factor_batch_size,
        factor_args=factor_args,
        overwrite_output_dir=True,
    )

    # Score computation phase
    scores_name = f"scores_k{k}"
    score_args = ScoreArguments(amp_dtype=amp_dtype)
    logger.info(f"  computing pairwise scores (query batch={query_batch_size})...")
    analyzer.compute_pairwise_scores(
        scores_name=scores_name,
        factors_name=factors_name,
        query_dataset=test_dataset,
        train_dataset=train_dataset,
        per_device_query_batch_size=query_batch_size,
        per_device_train_batch_size=factor_batch_size,
        score_args=score_args,
        overwrite_output_dir=True,
    )

    # Load scores: kronfluence stores them as a dict keyed by module / "all_modules"
    scores_dict = analyzer.load_pairwise_scores(scores_name=scores_name)
    # Concatenate across modules (sum-influence-across-layers convention).  If
    # already a single tensor under "all_modules", use directly.
    if isinstance(scores_dict, dict):
        if "all_modules" in scores_dict:
            scores = scores_dict["all_modules"].float()
        else:
            scores = sum(t.float() for t in scores_dict.values())
    else:
        scores = scores_dict.float()
    logger.info(f"  scores shape={tuple(scores.shape)}")

    # Save the full matrix locally
    matrix_path = output_dir / f"influence_k{k}.pt"
    torch.save(
        {
            "scores": scores.cpu(),
            "k": k,
            "model_path": str(model_path),
        },
        matrix_path,
    )

    # Compute share metrics
    mis_mask_bool = torch.from_numpy(mislabel_mask.astype(bool))
    metrics = _share_metrics_from_scores(scores, mis_mask_bool)
    record = {
        "k": k,
        "n_train": int(scores.shape[1]),
        "n_test": int(scores.shape[0]),
        "model_path": str(model_path),
        **metrics,
    }
    logger.info(
        f"  share_B = {metrics['share_B_signed_over_abs']:.4f}  "
        f"share_C = {metrics['share_C_abs_over_abs']:.4f}  "
        f"(mislabel rate baseline = "
        f"{metrics['fraction_mislabeled_in_train']:.4f})"
    )
    return record


# --------------------------------- main ---------------------------------- #


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True,
                        help="Directory containing model_k*.pt")
    parser.add_argument("--output-dir", required=True,
                        help="Where to write influence matrices and share records")
    parser.add_argument("--data-path", default="./data",
                        help="Root for CIFAR-10 data")
    parser.add_argument("--device", default=None)
    parser.add_argument("--k", type=int, default=None,
                        help="Only run this single k value (default: all in model-dir)")
    parser.add_argument("--ks", type=int, nargs="+", default=None,
                        help="Only run these k values (overrides --k)")
    parser.add_argument("--factor-batch-size", type=int, default=128)
    parser.add_argument("--query-batch-size", type=int, default=64)
    parser.add_argument("--bf16", action="store_true",
                        help="Use bfloat16 AMP for factor + score phases "
                             "(eigendecomposition stays FP64)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logger.info(f"device = {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Data (matches existing cifar_influence_share.py setup)
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
    )
    test_transform = transforms.Compose([transforms.ToTensor(), normalize])

    noisy_train_full = NoisyCIFAR10(
        root=args.data_path, train=True, noise_prob=0.15,
        transform=test_transform, seed=42,
    )
    train_indices, _ = compute_val_split_indices()
    train_subset = Subset(noisy_train_full, train_indices.tolist())

    test_dataset = NoisyCIFAR10(
        root=args.data_path, train=False, noise_prob=0.0,
        transform=test_transform, seed=42,
    )

    mislabel_mask = build_mislabel_mask(
        noisy_train_full.labels,
        noisy_train_full.cifar.targets,
        train_indices,
    )
    n_mis = int(mislabel_mask.sum())
    logger.info(
        f"Train: {len(train_subset)} samples, {n_mis} mislabeled "
        f"({100*n_mis/len(train_subset):.1f}%)"
    )

    # Discover models
    model_paths = {}
    for p in sorted(Path(args.model_dir).glob("model_k*.pt")):
        m = re.match(r"model_k(\d+)\.pt", p.name)
        if not m:
            continue
        k = int(m.group(1))
        if args.ks is not None and k not in args.ks:
            continue
        if args.k is not None and args.ks is None and k != args.k:
            continue
        model_paths[k] = p
    if not model_paths:
        raise FileNotFoundError(f"No matching model_k*.pt in {args.model_dir}")
    logger.info(f"Found {len(model_paths)} models: k={sorted(model_paths.keys())}")

    records_path = output_dir / "share_records.jsonl"

    for k in sorted(model_paths):
        result_path = output_dir / f"influence_k{k}.pt"
        if result_path.exists():
            logger.info(f"k={k}: {result_path} already exists, skipping")
            continue
        try:
            rec = run_for_k(
                k, model_paths[k],
                train_subset, test_dataset,
                mislabel_mask, output_dir, device,
                factor_batch_size=args.factor_batch_size,
                query_batch_size=args.query_batch_size,
                bf16=args.bf16,
            )
            with open(records_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
        except Exception as e:
            logger.exception(f"k={k}: failed -- {e}")

    logger.info(f"Done. Records: {records_path}")


if __name__ == "__main__":
    main()
