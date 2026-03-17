"""Evaluate variance across training splits for ResNet18 models.

Loads all variance-mode models (model_k*_split*.pt), runs them on the test set,
and computes:
  - Mean test loss across splits
  - Jensen Gap: E[log(q_bar[y] / q_j[y])] - the variance term in bias-variance decomposition

Uses Bessel's correction (n-1) for unbiased variance estimation with 4 models.

Output: evaluation.jsonl written alongside the model files.

With --temperature-scaling: fits a scalar temperature T per k value on one randomly
chosen model's logits (L-BFGS on test set NLL), then recomputes the full decomposition
with softmax(logits/T). Results go to temperature-scaled/evaluation.jsonl.

Usage:
    python -m jl.double_descent.resnet18.variance_evaluation \
        --model-path ./output/resnet18_variance/03-01-1010 \
        --data-path ./data

    python -m jl.double_descent.resnet18.variance_evaluation \
        --model-path ./output/resnet18_variance/03-01-1010 \
        --data-path ./data --temperature-scaling
"""

import argparse
import json
import logging
import os
import random
import re
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from jl.double_descent.resnet18.resnet18_data import NoisyCIFAR10
from jl.double_descent.resnet18.resnet18k import make_resnet18k
import torchvision.transforms as transforms

logger = logging.getLogger(__name__)

EVAL_BATCH_SIZE = 256


def discover_models(model_dir: str) -> Dict[int, List[Path]]:
    """Discover variance model files grouped by k.

    Returns:
        Dict mapping k -> sorted list of model file Paths.
    """
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


def evaluate_k(
    model_paths: List[Path],
    k: int,
    test_loader: DataLoader,
    device: torch.device,
) -> Dict:
    """Compute mean test loss and Jensen Gap for one k value.

    For each test batch:
      1. Forward pass all models, collect softmax distributions
      2. Compute q_bar = mean distribution across models
      3. Compute Jensen Gap: log(q_bar[y] / q_j[y]) for each model

    Uses Bessel's correction (n-1) for unbiased variance estimation.
    """
    num_models = len(model_paths)
    logger.info(f"k={k}: loading {num_models} models")

    models = [load_model(p, k, device) for p in model_paths]

    total_loss_per_model = [0.0] * num_models
    total_jensen_gap = 0.0
    total_samples = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            batch_size = images.size(0)

            all_probs = []
            for j, model in enumerate(models):
                logits = model(images)  # [B, 10]

                loss = F.cross_entropy(logits, labels, reduction='sum')
                total_loss_per_model[j] += loss.item()

                probs = F.softmax(logits, dim=-1)  # [B, 10]
                all_probs.append(probs)

            # Compute Jensen Gap: log(q_bar[y] / q_j[y])
            all_probs_t = torch.stack(all_probs, dim=0)  # [M, B, 10]
            q_bar = all_probs_t.mean(dim=0)  # [B, 10]
            log_q_bar = torch.log(q_bar + 1e-10)

            # Get log q_bar[y] for each sample
            log_q_bar_y = log_q_bar.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)  # [B]

            batch_jensen = 0.0
            for j in range(num_models):
                log_q_j = torch.log(all_probs_t[j] + 1e-10)
                log_q_j_y = log_q_j.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)  # [B]
                jensen_per_sample = log_q_bar_y - log_q_j_y  # [B]
                batch_jensen += jensen_per_sample.sum().item()

            # Bessel's correction: divide by (n-1) instead of n for unbiased variance
            total_jensen_gap += batch_jensen / (num_models - 1)
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
    )

    return {
        "k": k,
        "mean_test_loss": round(mean_test_loss, 6),
        "mean_jensen_gap": round(mean_jensen_gap, 6),
        "num_models": num_models,
        "total_samples": total_samples,
    }


def fit_temperature(model, test_loader, device):
    """Fit scalar temperature T via L-BFGS to minimize NLL on test set.

    Collects all logits first (10K x 10 for CIFAR-10, trivially small),
    then optimizes T on CPU.
    """
    all_logits = []
    all_labels = []
    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            logits = model(images.to(device))
            all_logits.append(logits.cpu())
            all_labels.append(labels)

    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)

    temperature = nn.Parameter(torch.ones(1))
    optimizer = torch.optim.LBFGS([temperature], lr=0.01, max_iter=50)

    def closure():
        optimizer.zero_grad()
        loss = F.cross_entropy(all_logits / temperature, all_labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return temperature.item()


def _ts_eval_worker(gpu_id, model_path_str, k, temperature, data_path, output_file):
    """Worker process: evaluate one model with temperature scaling.

    Loads model on assigned GPU, runs forward pass with temperature T,
    saves q_j[y] (probability of correct class) and loss to output_file.
    """
    device = torch.device(f"cuda:{gpu_id}")

    model = make_resnet18k(k=k, num_classes=10).to(device)
    model.load_state_dict(
        torch.load(model_path_str, map_location=device, weights_only=True)
    )
    model.eval()

    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
    )
    test_dataset = NoisyCIFAR10(
        root=data_path,
        train=False,
        noise_prob=0.0,
        transform=transforms.Compose([transforms.ToTensor(), normalize]),
    )
    test_loader = DataLoader(
        test_dataset, batch_size=EVAL_BATCH_SIZE, shuffle=False,
        num_workers=2, pin_memory=True,
    )

    all_q_j_y = []
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)

            loss = F.cross_entropy(logits / temperature, labels, reduction='sum')
            total_loss += loss.item()

            probs = F.softmax(logits / temperature, dim=-1)
            q_j_y = probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
            all_q_j_y.append(q_j_y.cpu())
            total_samples += labels.size(0)

    torch.save({
        'q_j_y': torch.cat(all_q_j_y),
        'total_loss': torch.tensor(total_loss),
        'total_samples': torch.tensor(total_samples),
    }, output_file)


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
        "--model-path",
        type=str,
        required=True,
        help="Directory containing model_k*_split*.pt files",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="./data",
        help="Directory containing CIFAR-10 data",
    )
    parser.add_argument(
        "--temperature-scaling",
        action="store_true",
        help="Fit per-k temperature and recompute bias-variance decomposition",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    grouped = discover_models(args.model_path)
    if not grouped:
        raise FileNotFoundError(
            f"No model_k*_split*.pt files found in {args.model_path}"
        )
    logger.info(
        f"Found models for k values: {list(grouped.keys())} "
        f"({sum(len(v) for v in grouped.values())} total)"
    )

    # Load test set
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
    )
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    test_dataset = NoisyCIFAR10(
        root=args.data_path,
        train=False,
        noise_prob=0.0,
        transform=test_transform,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    logger.info(f"Test set: {len(test_dataset)} samples")

    if args.temperature_scaling:
        mp.set_start_method('spawn', force=True)

        ts_output = Path(args.model_path) / "temperature-scaled"
        ts_output.mkdir(parents=True, exist_ok=True)
        ts_eval_file = ts_output / "evaluation.jsonl"
        if ts_eval_file.exists():
            ts_eval_file.unlink()

        k_values = sorted(grouped.keys())
        splits_per_k = len(grouped[k_values[0]])
        k_per_batch = max(1, torch.cuda.device_count() // splits_per_k)

        for batch_start in range(0, len(k_values), k_per_batch):
            batch_k = k_values[batch_start:batch_start + k_per_batch]

            # Fit temperature for each k (sequential, fast)
            temperatures = {}
            for k in batch_k:
                rng = random.Random(k)
                calib_idx = rng.randrange(len(grouped[k]))
                model = load_model(grouped[k][calib_idx], k, device)
                temperatures[k] = fit_temperature(model, test_loader, device)
                del model
                torch.cuda.empty_cache()
                logger.info(f"k={k}: fitted T={temperatures[k]:.4f}")

            # Spawn parallel workers across GPUs
            with tempfile.TemporaryDirectory() as tmp_dir:
                processes = []
                gpu_id = 0
                for k in batch_k:
                    for split_idx, model_path in enumerate(grouped[k]):
                        out_f = os.path.join(tmp_dir, f"k{k}_s{split_idx}.pt")
                        p = mp.Process(
                            target=_ts_eval_worker,
                            args=(gpu_id, str(model_path), k,
                                  temperatures[k], args.data_path, out_f),
                        )
                        p.start()
                        processes.append(p)
                        gpu_id += 1

                for p in processes:
                    p.join()

                # Compute decomposition for each k
                for k in batch_k:
                    num_models = len(grouped[k])
                    all_q_j_y = []
                    total_loss_sum = 0.0
                    total_samples = 0

                    for split_idx in range(num_models):
                        data = torch.load(
                            os.path.join(tmp_dir, f"k{k}_s{split_idx}.pt"),
                            weights_only=True,
                        )
                        all_q_j_y.append(data['q_j_y'])
                        total_loss_sum += data['total_loss'].item()
                        total_samples = data['total_samples'].item()

                    q_j_y = torch.stack(all_q_j_y)  # [M, N]
                    q_bar_y = q_j_y.mean(dim=0)  # [N]

                    log_q_bar_y = torch.log(q_bar_y + 1e-10)
                    total_jensen = 0.0
                    for j in range(num_models):
                        log_q_j_y = torch.log(q_j_y[j] + 1e-10)
                        total_jensen += (log_q_bar_y - log_q_j_y).sum().item()

                    # Bessel's correction (n-1)
                    mean_jensen_gap = total_jensen / ((num_models - 1) * total_samples)
                    mean_test_loss = total_loss_sum / (num_models * total_samples)

                    result = {
                        "k": k,
                        "mean_test_loss": round(mean_test_loss, 6),
                        "mean_jensen_gap": round(mean_jensen_gap, 6),
                        "temperature": round(temperatures[k], 6),
                        "num_models": num_models,
                        "total_samples": total_samples,
                    }

                    logger.info(
                        f"k={k}: T={temperatures[k]:.4f}, "
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

        for k, model_paths in grouped.items():
            result = evaluate_k(model_paths, k, test_loader, device)
            with open(output_path, "a") as fh:
                fh.write(json.dumps(result) + "\n")

        logger.info(f"Results written to {output_path}")


if __name__ == "__main__":
    main()
