#!/usr/bin/env python3
"""DEPRECATED — superseded by influence_main.py (which now saves
share_per_test_k*.npy directly) + plot_influence_share.py. Will be removed
on the next refactor pass.

Influence-share analysis: for each k, what fraction of each test point's
total contribution magnitude comes from the mislabeled training points?

For each (training i, test t) pair, define:
    c_{i, t} = ||r_i||_2 · |phi_i · phi_t|

(L2 norm of training point i's contribution vector to test point t's 10
class logits, the per-(i,t) factor inside our RMS influence formula.)

Then for each test point:
    share_M(t) = sum over i in M of c_{i,t}  /  sum over all i of c_{i,t}

where M = mislabeled training-point indices. Final metric per k:
    metric(k) = mean over test t of share_M(t)

Baseline = 0.151 (the dataset mislabel rate). The support-vector theory
predicts this should rise above the baseline near the interpolation peak.

Reuses the saved L2-fitted linear layers (finetune_k{k}.pt) from a prior
influence_main run, so no re-fitting needed.

Usage:
    python -m jl.double_descent.influence.cifar_influence_share \\
        --model-dir ./data/resnet18/04-11-1602 \\
        --influence-dir ./data/resnet18/04-11-1602/influence \\
        --data-path ./data \\
        --output-dir ./data/resnet18/04-11-1602/influence
"""

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from jl.double_descent.influence.decompose import build_mislabel_mask
from jl.double_descent.influence.l2_finetune import extract_features
from jl.double_descent.resnet18.evaluation import discover_models
from jl.double_descent.resnet18.resnet18_data import (
    NoisyCIFAR10,
    compute_val_split_indices,
)
from jl.double_descent.resnet18.resnet18k import make_resnet18k

logger = logging.getLogger(__name__)


def compute_share_for_k(
    k: int,
    model_path: Path,
    finetune_path: Path,
    train_loader: DataLoader,
    test_loader: DataLoader,
    mislabel_mask: np.ndarray,
    device: torch.device,
) -> dict:
    """Returns {k, mean_share, median_share, std_share, fraction_above_baseline}."""
    logger.info(f"--- k={k} ---")

    model = make_resnet18k(k=k, num_classes=10).to(device)
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )
    model.eval()

    # Load the L2-fine-tuned final layer (overwrites model.linear)
    ft_state = torch.load(finetune_path, map_location=device, weights_only=True)
    model.linear.load_state_dict(ft_state)
    model.eval()

    # Extract features (no augmentation; train_loader uses test-time transforms)
    phi_train, train_labels = extract_features(model, train_loader, device)
    phi_test, _ = extract_features(model, test_loader, device)

    # Residual norms ||r_i||_2 from the L2-fitted model
    with torch.no_grad():
        logits = model.linear(phi_train)
        probs = torch.softmax(logits, dim=-1)
        one_hot = torch.zeros_like(probs)
        one_hot.scatter_(1, train_labels.unsqueeze(1), 1.0)
        residuals = probs - one_hot
        grad_norms = residuals.norm(dim=1)  # [N_train]

    n_train = phi_train.size(0)
    n_test = phi_test.size(0)
    logger.info(f"  N_train={n_train}, N_test={n_test}, mislabel rate={mislabel_mask.mean():.4f}")

    mis_idx = torch.from_numpy(mislabel_mask.astype(bool)).to(device)

    # Compute contribution sums per test point
    # c_{i,t} = grad_norm_i * |phi_i · phi_t|
    # total_t = sum_i c_{i,t}
    # mis_t = sum_{i in M} c_{i,t}
    # Memory budget: phi_train @ phi_test^T can be large for big k. Chunk over test.
    test_chunk = 1024
    total_contrib = torch.zeros(n_test, device=device, dtype=torch.float64)
    mis_contrib = torch.zeros(n_test, device=device, dtype=torch.float64)

    with torch.no_grad():
        # weighted_train_features = grad_norms.unsqueeze(1) * phi_train  -- but
        # we need |phi_train · phi_test|, then multiply by grad_norm. The
        # weighted form doesn't help because abs is non-linear.
        for ts in range(0, n_test, test_chunk):
            te = min(ts + test_chunk, n_test)
            # [N_train, chunk] absolute dot products
            dots = phi_train @ phi_test[ts:te].t()
            abs_dots = dots.abs()
            # multiply by grad_norm row-wise: [N_train, 1] * [N_train, chunk]
            contrib = grad_norms.unsqueeze(1) * abs_dots  # [N_train, chunk]
            total_contrib[ts:te] = contrib.sum(dim=0).double()
            mis_contrib[ts:te] = contrib[mis_idx].sum(dim=0).double()

    share = mis_contrib / total_contrib  # [N_test]
    share_np = share.cpu().numpy()

    baseline = float(mislabel_mask.mean())
    record = {
        "k": k,
        "n_train": n_train,
        "n_test": n_test,
        "mislabel_rate": baseline,
        "mean_share": float(share_np.mean()),
        "median_share": float(np.median(share_np)),
        "std_share": float(share_np.std()),
        "fraction_above_baseline": float((share_np > baseline).mean()),
    }
    logger.info(
        f"  mean_share={record['mean_share']:.4f} "
        f"(baseline={baseline:.4f}), "
        f"std={record['std_share']:.4f}, "
        f"frac_above={record['fraction_above_baseline']:.3f}"
    )

    del phi_train, phi_test, residuals, model, grad_norms, dots, abs_dots, contrib
    torch.cuda.empty_cache()
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--influence-dir", required=True,
                        help="Directory with finetune_k{K}.pt files from a prior influence_main run")
    parser.add_argument("--data-path", default="./data")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default=None)
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

    model_dir = Path(args.model_dir)
    influence_dir = Path(args.influence_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Discover models
    models = discover_models(str(model_dir))
    logger.info(f"Found {len(models)} models: k={list(models.keys())}")

    # Build train/test loaders matching the influence pipeline
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
    train_loader = DataLoader(
        train_subset, batch_size=256, shuffle=False, num_workers=2,
    )

    test_dataset = NoisyCIFAR10(
        root=args.data_path, train=False, noise_prob=0.0,
        transform=test_transform, seed=42,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=256, shuffle=False, num_workers=2,
    )

    mislabel_mask = build_mislabel_mask(
        noisy_train_full.labels,
        noisy_train_full.cifar.targets,
        train_indices,
    )

    # Compute share per k
    records = []
    for k in sorted(models.keys()):
        finetune_path = influence_dir / f"finetune_k{k}.pt"
        if not finetune_path.exists():
            logger.warning(f"  skipping k={k}: {finetune_path} missing")
            continue
        rec = compute_share_for_k(
            k, models[k], finetune_path, train_loader, test_loader,
            mislabel_mask, device,
        )
        records.append(rec)

    # Save
    out_path = output_dir / "influence_share.jsonl"
    with open(out_path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    logger.info(f"Wrote {out_path}")

    # Plot
    if records:
        ks = [r["k"] for r in records]
        means = [r["mean_share"] for r in records]
        baseline = records[0]["mislabel_rate"]

        plt.rcParams.update({"font.family": "serif", "font.size": 11})
        fig, ax = plt.subplots(1, 1, figsize=(8, 5), dpi=150)
        ax.plot(ks, means, "o-", color="#d62728", markersize=5,
                markerfacecolor="white", markeredgewidth=1.2, linewidth=1.5,
                label="Mean mislabeled-share over test points")
        ax.axhline(baseline, color="0.5", linestyle="--", linewidth=0.8,
                   alpha=0.8, label=f"Mislabel rate ({baseline*100:.1f}%)")
        ax.set_xlabel("Width parameter k")
        ax.set_ylabel(r"$\mathrm{share}_M(t) = \frac{\sum_{i \in M} c_{i,t}}{\sum_i c_{i,t}}$, mean over $t$")
        ax.set_title(
            "Influence share by mislabeled training points vs k\n"
            r"$c_{i,t} = \|r_i\|_2 \cdot |\varphi_i \cdot \varphi_t|$"
        )
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        plot_path = output_dir / "influence_share_vs_k.png"
        fig.savefig(plot_path, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Wrote {plot_path}")


if __name__ == "__main__":
    main()
