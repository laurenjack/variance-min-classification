"""Shared calibration sweep logic.

Given pre-extracted features and logits, runs all calibration baselines
and an L2 lambda sweep, then reports results.

The L2 lambda sweep parallelizes across available GPUs using
spawn-based multiprocessing (one lambda per GPU, batched).
Works transparently on a single GPU.
"""

import json
import logging
import torch.multiprocessing as mp
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from jl.double_descent.calibration.baselines import (
    apply_dirichlet_l2,
    apply_histogram_binning,
    apply_temperature,
    apply_vector_scaling,
    fit_dirichlet_l2,
    fit_histogram_binning,
    fit_temperature,
    fit_vector_scaling,
)
from jl.double_descent.calibration.evaluate import evaluate_logits, evaluate_probs
from jl.double_descent.l2_calibrate_lib import l2_calibrate_final_layer

logger = logging.getLogger(__name__)

DEFAULT_LAMBDAS = [1e-4, 1e-3, 1e-2, 5e-2, 1e-1, 2e-1, 3e-1, 5e-1, 7e-1, 1.0, 2.0, 3.0, 5.0, 10.0]


def _sweep_worker(
    gpu_id: int,
    lam: float,
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    val_features: torch.Tensor,
    val_labels: torch.Tensor,
    original_head_state: dict,
    num_classes: int,
    feature_dim: int,
    max_steps: int,
    result_dict: dict,
) -> None:
    """Worker that runs L2 calibration for a single lambda on one GPU."""
    device = torch.device(f"cuda:{gpu_id}")

    linear = nn.Linear(feature_dim, num_classes).to(device)
    linear.load_state_dict(original_head_state)

    l2_calibrate_final_layer(
        features=train_features,
        targets=train_labels,
        linear_layer=linear,
        l2_lambda=lam,
        max_steps=max_steps,
        device=device,
    )

    linear.eval()
    with torch.no_grad():
        val_cal_logits = linear(val_features.to(device)).cpu()
    val_cal_metrics = evaluate_logits(val_cal_logits, val_labels)

    result_dict[lam] = (val_cal_metrics, linear.state_dict())


def _run_lambda_sweep(
    lambdas: List[float],
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    val_features: torch.Tensor,
    val_labels: torch.Tensor,
    original_head_state: dict,
    num_classes: int,
    feature_dim: int,
    max_steps: int,
) -> List[Tuple[float, Dict, dict]]:
    """Run L2 calibration across lambdas, parallelizing across GPUs.

    Returns:
        List of (lambda, val_metrics, state_dict) tuples.
    """
    num_gpus = max(torch.cuda.device_count(), 1)
    use_multiprocessing = torch.cuda.is_available() and num_gpus > 1

    if use_multiprocessing:
        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass

        manager = mp.Manager()
        result_dict = manager.dict()

        for batch_start in range(0, len(lambdas), num_gpus):
            batch = lambdas[batch_start: batch_start + num_gpus]
            batch_num = batch_start // num_gpus + 1
            total_batches = (len(lambdas) + num_gpus - 1) // num_gpus
            logger.info(f"Lambda batch {batch_num}/{total_batches}: λ = {[f'{l:.0e}' for l in batch]}")

            processes = []
            for gpu_id, lam in enumerate(batch):
                p = mp.Process(
                    target=_sweep_worker,
                    args=(
                        gpu_id,
                        lam,
                        train_features,
                        train_labels,
                        val_features,
                        val_labels,
                        original_head_state,
                        num_classes,
                        feature_dim,
                        max_steps,
                        result_dict,
                    ),
                )
                p.start()
                processes.append(p)

            for p in processes:
                p.join()
                if p.exitcode != 0:
                    logger.error(f"Worker exited with code {p.exitcode}")

        sweep_results = []
        for lam in lambdas:
            val_metrics, state = result_dict[lam]
            sweep_results.append((lam, val_metrics, state))
            logger.info(f"  λ={lam:.0e}: val_ece={val_metrics['ece']:.4f}, val_nll={val_metrics['nll']:.4f}")

    else:
        # Single GPU / CPU: run sequentially
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        sweep_results = []
        for lam in lambdas:
            linear = nn.Linear(feature_dim, num_classes).to(device)
            linear.load_state_dict(original_head_state)

            l2_calibrate_final_layer(
                features=train_features,
                targets=train_labels,
                linear_layer=linear,
                l2_lambda=lam,
                max_steps=max_steps,
                device=device,
            )

            linear.eval()
            with torch.no_grad():
                val_cal_logits = linear(val_features.to(device)).cpu()
            val_cal_metrics = evaluate_logits(val_cal_logits, val_labels)

            sweep_results.append((lam, val_cal_metrics, linear.state_dict()))
            logger.info(f"  λ={lam:.0e}: val_ece={val_cal_metrics['ece']:.4f}, val_nll={val_cal_metrics['nll']:.4f}")

    return sweep_results


def run_calibration_sweep(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    val_features: torch.Tensor,
    val_labels: torch.Tensor,
    test_features: torch.Tensor,
    test_labels: torch.Tensor,
    original_head_state: dict,
    num_classes: int,
    feature_dim: int,
    lambdas: Optional[List[float]] = None,
    max_steps: int = 30,
    sweep_metric: str = "ece",
    device: Optional[torch.device] = None,
    output_dir: Optional[Path] = None,
) -> Dict:
    """Run all calibration methods and L2 lambda sweep.

    Baselines (fit on val logits): temperature scaling, vector scaling,
    histogram binning, Dirichlet L2.

    L2 calibration (fit on train features): sweep over lambda values
    in parallel across available GPUs, select best by val metric,
    report on test.

    Returns:
        Dict with all results keyed by method name.
    """
    if lambdas is None:
        lambdas = DEFAULT_LAMBDAS
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Compute logits from original head for baselines
    head = nn.Linear(feature_dim, num_classes)
    head.load_state_dict(original_head_state)
    head.eval()
    with torch.no_grad():
        val_logits = head(val_features).detach()
        test_logits = head(test_features).detach()

    # === Uncalibrated ===
    logger.info("=== Uncalibrated evaluation ===")
    uncal_metrics = evaluate_logits(test_logits, test_labels)
    logger.info(f"Uncalibrated: {uncal_metrics}")

    # === Temperature scaling (fit on val) ===
    logger.info("=== Temperature scaling ===")
    T = fit_temperature(val_logits, val_labels)
    ts_metrics = evaluate_logits(apply_temperature(test_logits, T), test_labels)
    logger.info(f"Temperature-scaled (T={T:.4f}): {ts_metrics}")

    # === Vector scaling (fit on val) ===
    logger.info("=== Vector scaling ===")
    vs_weights, vs_biases = fit_vector_scaling(val_logits, val_labels)
    vs_metrics = evaluate_logits(apply_vector_scaling(test_logits, vs_weights, vs_biases), test_labels)
    logger.info(f"Vector-scaled: {vs_metrics}")

    # === Histogram binning (fit on val) ===
    logger.info("=== Histogram binning ===")
    bin_bounds, bin_accs = fit_histogram_binning(val_logits, val_labels)
    hb_metrics = evaluate_probs(apply_histogram_binning(test_logits, bin_bounds, bin_accs), test_labels)
    logger.info(f"Histogram-binned: {hb_metrics}")

    # === Dirichlet calibration L2 (fit on val) ===
    logger.info("=== Dirichlet calibration L2 ===")
    dir_W, dir_b = fit_dirichlet_l2(val_logits, val_labels)
    dir_metrics = evaluate_logits(apply_dirichlet_l2(test_logits, dir_W, dir_b), test_labels)
    logger.info(f"Dirichlet L2: {dir_metrics}")

    # === L2 calibration sweep (fit on train, select by val) ===
    num_gpus = max(torch.cuda.device_count(), 1) if torch.cuda.is_available() else 1
    logger.info(
        f"=== L2 calibration sweep (selecting by val {sweep_metric}, "
        f"{len(lambdas)} lambdas, {num_gpus} GPU(s)) ==="
    )

    sweep_results = _run_lambda_sweep(
        lambdas=lambdas,
        train_features=train_features,
        train_labels=train_labels,
        val_features=val_features,
        val_labels=val_labels,
        original_head_state=original_head_state,
        num_classes=num_classes,
        feature_dim=feature_dim,
        max_steps=max_steps,
    )

    # Select best by val metric
    best_idx = min(range(len(sweep_results)), key=lambda i: sweep_results[i][1][sweep_metric])
    best_lambda, best_val_metrics, best_state = sweep_results[best_idx]
    logger.info(f"Best λ={best_lambda:.0e} (val {sweep_metric}={best_val_metrics[sweep_metric]:.4f})")

    # Print sweep table
    print(f"\n(Selecting by val {sweep_metric})")
    print("=" * 50)
    print(f"{'Lambda':<12} {'Val ECE':>10} {'Val NLL':>10} {'Val Acc':>10}")
    print("-" * 50)
    for lam, val_m, _ in sweep_results:
        marker = " <-- best" if lam == best_lambda else ""
        print(f"{lam:<12.0e} {val_m['ece']:>10.4f} {val_m['nll']:>10.4f} {val_m['accuracy']:>10.4f}{marker}")
    print("=" * 50)

    # Evaluate best on test
    linear = nn.Linear(feature_dim, num_classes).to(device)
    linear.load_state_dict(best_state)
    linear.eval()
    with torch.no_grad():
        l2_test_logits = linear(test_features.to(device)).cpu()
    l2_metrics = evaluate_logits(l2_test_logits, test_labels)
    logger.info(f"L2-calibrated (λ={best_lambda:.0e}): {l2_metrics}")

    # === Assemble results ===
    results = {
        "uncalibrated": uncal_metrics,
        "temperature_scaled": {**ts_metrics, "temperature": round(T, 6)},
        "vector_scaled": vs_metrics,
        "histogram_binning": hb_metrics,
        "dirichlet_l2": dir_metrics,
        "l2_calibrated": {**l2_metrics, "l2_lambda": best_lambda, "max_steps": max_steps},
    }

    # Print summary table
    print("\n" + "=" * 70)
    print(f"{'Method':<20} {'NLL':>8} {'Acc':>8} {'ECE':>8} {'Brier':>8} {'AUROC':>8} {'AUPR':>8}")
    print("-" * 70)
    for method, metrics in results.items():
        print(
            f"{method:<20} {metrics['nll']:>8.4f} {metrics['accuracy']:>8.4f} "
            f"{metrics['ece']:>8.4f} {metrics['brier']:>8.4f} "
            f"{metrics['auroc']:>8.4f} {metrics['aupr']:>8.4f}"
        )
    print("=" * 70)

    # Print deltas
    delta_methods = ["temperature_scaled", "vector_scaled", "histogram_binning", "dirichlet_l2", "l2_calibrated"]
    print(f"\n{'Method':<20} {'ΔNLL':>8} {'ΔAcc':>8} {'ΔECE':>8} {'ΔBrier':>8} {'ΔAUROC':>8} {'ΔAUPR':>8}")
    print("-" * 70)
    base = uncal_metrics
    for method in delta_methods:
        m = results[method]
        print(
            f"{method:<20} {m['nll'] - base['nll']:>+8.4f} {m['accuracy'] - base['accuracy']:>+8.4f} "
            f"{m['ece'] - base['ece']:>+8.4f} {m['brier'] - base['brier']:>+8.4f} "
            f"{m['auroc'] - base['auroc']:>+8.4f} {m['aupr'] - base['aupr']:>+8.4f}"
        )
    print("=" * 70)

    # Save if output_dir provided
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_dir / "calibration_results.json", "w") as f:
            json.dump(results, f, indent=2)

        with open(output_dir / "sweep_results.json", "w") as f:
            json.dump(
                [{"l2_lambda": lam, **vm} for lam, vm, _ in sweep_results],
                f, indent=2,
            )

        torch.save(best_state, output_dir / "calibrated_head.pt")
        torch.save({"logits": test_logits, "labels": test_labels}, output_dir / "test_logits.pt")

        logger.info(f"Saved results to {output_dir}")

    return results
