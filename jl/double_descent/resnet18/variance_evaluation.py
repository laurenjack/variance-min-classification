"""Evaluate variance across training splits for ResNet18 models.

Loads all variance-mode models (model_k*_split*.pt), runs them on the test
set, and computes:
  - Mean test loss across splits
  - Jensen Gap: E[log(q_bar[y] / q_j[y])] - the variance term in
    bias-variance decomposition

Output: evaluation.jsonl written alongside the model files.

Usage:
    python -m jl.double_descent.resnet18.variance_evaluation \
        --model-path ./output/resnet18_variance/03-01-1010 \
        --data-path ./data

    # With per-model temperature scaling (Guo et al. protocol). The CIFAR
    # test set is split deterministically 5K/5K: first half fits T per
    # (k, split) model, second half is used for the decomposition. Output
    # goes to evaluation_ts.jsonl. Aborts if any L-BFGS fit fails to converge.
    python -m jl.double_descent.resnet18.variance_evaluation \
        --model-path ./output/resnet18_variance/03-01-1010 \
        --data-path ./data --temperature-scaling

    # Evaluate the early-stop checkpoints instead (writes to
    # early_stop/evaluation.jsonl):
    python -m jl.double_descent.resnet18.variance_evaluation \
        --model-path ./output/resnet18_variance/03-01-1010 \
        --data-path ./data --early-stop
"""

import argparse
import json
import logging
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from jl.double_descent.resnet18.resnet18_data import NoisyCIFAR10
from jl.double_descent.resnet18.resnet18k import make_resnet18k
from jl.double_descent.temperature_scaling import fit_temperature
import torchvision.transforms as transforms

logger = logging.getLogger(__name__)

EVAL_BATCH_SIZE = 256

# Deterministic seed for the 10K -> 5K/5K val/test split used at TS time.
TS_SPLIT_SEED = 91827


def discover_models(model_dir: str) -> Dict[int, List[Path]]:
    """Discover variance model files grouped by k."""
    path = Path(model_dir)
    model_files = sorted(path.glob("model_k*_split*.pt"))

    grouped: Dict[int, List[Path]] = defaultdict(list)
    for f in model_files:
        match = re.match(r"model_k(\d+)_split(\d+)\.pt", f.name)
        if match:
            k = int(match.group(1))
            grouped[k].append(f)

    return dict(sorted(grouped.items()))


def load_model(
    model_path: Path,
    k: int,
    device: torch.device,
) -> torch.nn.Module:
    """Instantiate model architecture and load saved weights."""
    model = make_resnet18k(k=k, num_classes=10).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    return model


def _collect_logits(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Forward the loader and return (logits, labels) tensors on `device`."""
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            all_logits.append(model(images))
            all_labels.append(labels)
    return torch.cat(all_logits, dim=0), torch.cat(all_labels, dim=0)


def evaluate_k(
    model_paths: List[Path],
    k: int,
    val_loader: Optional[DataLoader],
    test_loader: DataLoader,
    device: torch.device,
    apply_temperature_scaling: bool = False,
) -> Dict:
    """Compute mean test loss and Jensen Gap for one k value.

    When apply_temperature_scaling, fits a per-model T on val_loader and
    divides each model's test logits by T before computing log-softmax.
    Aborts (RuntimeError) if any L-BFGS fit fails to converge.
    """
    num_models = len(model_paths)
    logger.info(
        f"k={k}: loading {num_models} models"
        f"{' (TS)' if apply_temperature_scaling else ''}"
    )

    models = [load_model(p, k, device) for p in model_paths]

    # Per-model fitted temperatures (defaults to 1.0 i.e. identity).
    temperatures = [1.0] * num_models
    ts_diags: List[Dict] = []
    if apply_temperature_scaling:
        assert val_loader is not None
        for j, model in enumerate(models):
            val_logits, val_labels = _collect_logits(model, val_loader, device)
            T, diag = fit_temperature(
                val_logits, val_labels, return_diagnostics=True,
            )
            if not diag["converged"]:
                raise RuntimeError(
                    f"L-BFGS did not converge for k={k}, split={j}: "
                    f"T={diag['T']:.4f}, |dCE/dT|={abs(diag['final_grad']):.2e}. "
                    f"Aborting."
                )
            temperatures[j] = T
            ts_diags.append({"split_id": j, **diag})
            del val_logits, val_labels

    total_loss_per_model = [0.0] * num_models
    total_jensen_gap = 0.0
    total_samples = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            batch_size = images.size(0)

            all_log_probs = []
            for j, model in enumerate(models):
                logits = model(images)  # [B, 10]
                if apply_temperature_scaling:
                    logits = logits / temperatures[j]

                loss = F.cross_entropy(logits, labels, reduction='sum')
                total_loss_per_model[j] += loss.item()

                log_probs = F.log_softmax(logits, dim=-1)  # [B, 10]
                all_log_probs.append(log_probs)

            # Jensen Gap in log space: log(q_bar[y]) - log(q_j[y]).
            all_log_probs_t = torch.stack(all_log_probs, dim=0)  # [M, B, 10]
            log_q_bar = torch.logsumexp(all_log_probs_t, dim=0) - math.log(num_models)  # [B, 10]

            log_q_bar_y = log_q_bar.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)  # [B]

            batch_jensen = 0.0
            for j in range(num_models):
                log_q_j_y = all_log_probs_t[j].gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
                jensen_per_sample = log_q_bar_y - log_q_j_y
                batch_jensen += jensen_per_sample.sum().item()

            total_jensen_gap += batch_jensen / num_models
            total_samples += batch_size

    del models
    torch.cuda.empty_cache()

    mean_test_loss = (
        sum(total_loss_per_model) / (num_models * total_samples)
        if total_samples > 0
        else 0.0
    )
    mean_jensen_gap = total_jensen_gap / total_samples if total_samples > 0 else 0.0

    logger.info(
        f"k={k}: mean_test_loss={mean_test_loss:.4f}, "
        f"mean_jensen_gap={mean_jensen_gap:.6f}"
        + (f", mean T={sum(temperatures)/len(temperatures):.4f}" if apply_temperature_scaling else "")
    )

    result = {
        "k": k,
        "mean_test_loss": round(mean_test_loss, 6),
        "mean_jensen_gap": round(mean_jensen_gap, 6),
        "num_models": num_models,
        "total_samples": total_samples,
    }
    if ts_diags:
        result["ts_temperatures"] = [round(d["T"], 6) for d in ts_diags]
        result["ts_all_converged"] = all(d["converged"] for d in ts_diags)
        result["ts_max_abs_grad"] = max(abs(d["final_grad"]) for d in ts_diags)
    return result


def _split_test_indices(
    n_total: int = 10000,
    val_size: int = 5000,
    seed: int = TS_SPLIT_SEED,
) -> Tuple[np.ndarray, np.ndarray]:
    """Deterministic 50/50 split of CIFAR-10 test indices: val_indices for
    T fitting, test_indices for the variance decomposition."""
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n_total)
    return perm[:val_size], perm[val_size:]


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Evaluate variance across training splits for ResNet18 models"
    )
    parser.add_argument(
        "--model-path", type=str, required=True,
        help="Run directory containing model_k*_split*.pt (or its "
             "early_stop/ subdir when --early-stop is passed)",
    )
    parser.add_argument(
        "--data-path", type=str, default="./data",
        help="Directory containing CIFAR-10 data",
    )
    parser.add_argument(
        "--early-stop", action="store_true",
        help="Evaluate the early_stop/model_k*_split*.pt checkpoints "
             "instead of the FINAL ones. Output goes to early_stop/evaluation.jsonl.",
    )
    parser.add_argument(
        "--temperature-scaling", action="store_true",
        help="Fit a scalar temperature T per (k, split) on a deterministic "
             "5K subset of the CIFAR-10 test set (Guo et al. protocol) and "
             "divide test logits by T before the variance decomposition. "
             "The remaining 5K is used for the decomp. Aborts if any L-BFGS "
             "fit fails to converge (|dCE/dT| >= 1e-4). Output goes to "
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
            f"No model_k*_split*.pt files found in {model_dir}"
        )
    logger.info(
        f"Found models for k values: {list(grouped.keys())} "
        f"({sum(len(v) for v in grouped.values())} total) "
        f"({'ES' if args.early_stop else 'FINAL'} checkpoints)"
    )

    # Load test set with no augmentation.
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
    )
    test_transform = transforms.Compose([transforms.ToTensor(), normalize])
    test_dataset = NoisyCIFAR10(
        root=args.data_path,
        train=False,
        noise_prob=0.0,
        transform=test_transform,
    )

    val_loader: Optional[DataLoader] = None
    if args.temperature_scaling:
        # 5K/5K deterministic split of CIFAR test.
        val_idx, decomp_idx = _split_test_indices(
            n_total=len(test_dataset), val_size=5000, seed=TS_SPLIT_SEED,
        )
        val_subset = Subset(test_dataset, val_idx.tolist())
        decomp_subset = Subset(test_dataset, decomp_idx.tolist())
        val_loader = DataLoader(
            val_subset, batch_size=EVAL_BATCH_SIZE, shuffle=False,
            num_workers=4, pin_memory=True,
        )
        test_loader = DataLoader(
            decomp_subset, batch_size=EVAL_BATCH_SIZE, shuffle=False,
            num_workers=4, pin_memory=True,
        )
        logger.info(
            f"TS mode: val (T-fit) = {len(val_subset)} samples, "
            f"test (decomp) = {len(decomp_subset)} samples, seed={TS_SPLIT_SEED}"
        )
    else:
        test_loader = DataLoader(
            test_dataset, batch_size=EVAL_BATCH_SIZE, shuffle=False,
            num_workers=4, pin_memory=True,
        )
        logger.info(f"Test set: {len(test_dataset)} samples")

    out_name = "evaluation_ts.jsonl" if args.temperature_scaling else "evaluation.jsonl"
    output_path = model_dir / out_name
    if output_path.exists():
        output_path.unlink()

    for k, model_paths in grouped.items():
        result = evaluate_k(
            model_paths, k, val_loader, test_loader, device,
            apply_temperature_scaling=args.temperature_scaling,
        )
        with open(output_path, "a") as fh:
            fh.write(json.dumps(result) + "\n")

    logger.info(f"Results written to {output_path}")


if __name__ == "__main__":
    main()
