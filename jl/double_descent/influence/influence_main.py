#!/usr/bin/env python3
"""Training point influence decomposition for ResNet18 double descent.

For each saved ResNet18k model, this script:
1. Extracts penultimate features (frozen backbone)
2. Fine-tunes the final linear layer with L2 regularization (L-BFGS)
3. Validates the Yeh & Kim (2018) analytic decomposition
4. Computes per-training-point influence scores
5. Saves results and generates Figure X (original vs fine-tuned loss)

Usage:
    python -m jl.double_descent.influence.influence_main \
        --model-dir ./data/resnet18/04-11-1602 \
        --data-path ./data \
        --lambda-l2 1e-4 \
        --output-dir ./data/resnet18/04-11-1602/influence
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from jl.double_descent.resnet18.evaluation import (
    _metrics_pass,
    discover_models,
)
from jl.double_descent.resnet18.resnet18_data import (
    NoisyCIFAR10,
    compute_val_split_indices,
)
from jl.double_descent.resnet18.resnet18k import make_resnet18k

from .decompose import (
    build_mislabel_mask,
    compute_influence_scores,
    compute_residuals,
    compute_summary_stats,
    save_influence_jsonl,
)
from .l2_finetune import extract_features, l2_finetune
from .validate import plot_figure_x, validate_decomposition

logger = logging.getLogger(__name__)


def process_k(
    k: int,
    model_path: Path,
    train_loader: DataLoader,
    test_loader: DataLoader,
    mislabel_mask: np.ndarray,
    train_indices: np.ndarray,
    noisy_labels: list,
    original_labels: list,
    lambda_l2: float,
    output_dir: Path,
    device: torch.device,
    use_float64: bool = False,
    file_suffix: str = "",
    distill: bool = False,
    temperature: float = None,
    lr: float = 1.0,
    tolerance_grad: float = 1e-7,
    max_iter: int = 200,
) -> dict:
    """Run the full influence pipeline for one k value.

    Returns a summary dict with influence stats and fine-tuned metrics.
    """
    logger.info(f"--- k={k} ---")

    # Load model
    model = make_resnet18k(k=k, num_classes=10).to(device)
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )
    model.eval()

    # Optional: pre-scale the linear layer by 1/T so that the distill target
    # and the L-BFGS init both live in the TS-scaled space.  The features
    # phi remain unscaled (TS rescales logits, not features), and the
    # diagnostic comparisons (cos, norm_ratio) are then done in scaled
    # space (cosine is scale-invariant; norm ratio uses W_orig/T as anchor).
    if temperature is not None:
        if not distill:
            raise ValueError("temperature is only meaningful in distill mode")
        with torch.no_grad():
            model.linear.weight.div_(temperature)
            model.linear.bias.div_(temperature)
        logger.info(f"k={k}: scaled linear layer by 1/T={1/temperature:.4f}")

    # Extract features
    logger.info(f"Extracting features (dim={8*k})...")
    phi_train, train_labels = extract_features(model, train_loader, device)
    phi_test, _ = extract_features(model, test_loader, device)
    logger.info(
        f"Features: train {phi_train.shape}, test {phi_test.shape}"
    )

    # L2 fine-tune
    loss_mode = "distill" if distill else "label-CE"
    if temperature is not None:
        loss_mode += f"_ts(T={temperature:.4f})"
    logger.info(
        f"Fine-tuning final layer with L-BFGS "
        f"(dtype={'float64' if use_float64 else 'float32'}, loss={loss_mode})..."
    )
    # Snapshot original W,b so we can compute rotation/scale diagnostics
    # AND use as distillation soft-target if distill=True
    W_orig = model.linear.weight.detach().clone().float()
    b_orig = model.linear.bias.detach().clone().float()
    grad_norm = l2_finetune(
        model, phi_train, train_labels, lambda_l2=lambda_l2,
        use_float64=use_float64,
        distill_W_orig=W_orig if distill else None,
        distill_b_orig=b_orig if distill else None,
        lr=lr,
        tolerance_grad=tolerance_grad,
        max_iter=max_iter,
    )
    W_ft = model.linear.weight.detach().float()
    b_ft = model.linear.bias.detach().float()

    # Validate decomposition identity
    val_record = validate_decomposition(
        phi_train, train_labels, model.linear, lambda_l2,
        distill_W_orig=W_orig if distill else None,
        distill_b_orig=b_orig if distill else None,
    )
    val_record["grad_norm"] = grad_norm
    val_record["lambda_l2"] = lambda_l2
    val_record["use_float64"] = use_float64
    val_record["loss_mode"] = loss_mode
    if temperature is not None:
        val_record["temperature"] = temperature
    # Rotation / scale diagnostics
    cos_per_class = torch.nn.functional.cosine_similarity(
        W_orig, W_ft, dim=1,
    ).cpu().tolist()
    frob_cos = float(
        (W_orig * W_ft).sum() / (W_orig.norm() * W_ft.norm() + 1e-30)
    )
    norm_ratio_per_class = (
        W_ft.norm(dim=1) / W_orig.norm(dim=1).clamp_min(1e-30)
    ).cpu().tolist()
    val_record["cos_per_class"] = cos_per_class
    val_record["mean_per_class_cos"] = float(sum(cos_per_class) / len(cos_per_class))
    val_record["frob_cos"] = frob_cos
    val_record["norm_ratio_per_class"] = norm_ratio_per_class
    val_record["mean_norm_ratio"] = float(sum(norm_ratio_per_class) / len(norm_ratio_per_class))

    # Pearson correlation between FT and original model's predictions over
    # the training set (flattened over N*C). For TS-distill these are
    # already in TS-scaled space; Pearson is scale-invariant.
    with torch.no_grad():
        Wo = W_orig.to(phi_train.device)
        bo = b_orig.to(phi_train.device)
        Wf = W_ft.to(phi_train.device)
        bf = b_ft.to(phi_train.device)
        logits_orig_flat = (phi_train @ Wo.t() + bo).flatten()
        logits_ft_flat = (phi_train @ Wf.t() + bf).flatten()
        probs_orig_flat = torch.nn.functional.softmax(
            phi_train @ Wo.t() + bo, dim=1
        ).flatten()
        probs_ft_flat = torch.nn.functional.softmax(
            phi_train @ Wf.t() + bf, dim=1
        ).flatten()

        def _pearson(x, y):
            xm = x - x.mean()
            ym = y - y.mean()
            return (xm * ym).sum() / (xm.norm() * ym.norm() + 1e-30)

        corr_logits = float(_pearson(logits_orig_flat, logits_ft_flat).item())
        corr_probs = float(_pearson(probs_orig_flat, probs_ft_flat).item())
    val_record["corr_logits"] = corr_logits
    val_record["corr_probs"] = corr_probs

    # Save validation
    val_path = output_dir / f"validation_k{k}{file_suffix}.json"
    with open(val_path, "w") as f:
        json.dump(val_record, f, indent=2)

    # Compute fine-tuned test metrics for Figure X
    ft_test_loss, ft_test_error, _ = _metrics_pass(
        model, test_loader, device
    )
    ft_metrics = {"test_loss": ft_test_loss}
    logger.info(f"Fine-tuned test_loss={ft_test_loss:.4f}")

    # Compute influence scores
    logger.info("Computing influence scores...")
    residuals = compute_residuals(phi_train, train_labels, model.linear)
    influence = compute_influence_scores(phi_train, phi_test, residuals)

    # Save per-point influence
    save_influence_jsonl(
        influence, mislabel_mask, train_indices,
        noisy_labels, original_labels, output_dir, k,
        file_suffix=file_suffix,
    )

    # Save fine-tuned linear layer
    ft_path = output_dir / f"finetune_k{k}{file_suffix}.pt"
    torch.save(model.linear.state_dict(), ft_path)

    # Canonical mislabeled-influence share, per test point t:
    #   v(t)          = sum_i r_i * (phi_i . phi_t)         [in class space]
    #   v_mislabel(t) = sum_{i in M} r_i * (phi_i . phi_t)
    #   c_mislabel(t) = <v_mislabel(t), v(t)> / ||v(t)||^2   (standard projection scalar)
    # Since v = v_mislabel + v_correct, we have c_mislabel + c_correct = 1
    # by construction, so c_mislabel IS the share (no further normalization).
    # The L2 scale factor cancels in the ratio so we omit it.
    logger.info("Computing mislabeled influence share per test point...")
    mis_mask_t = torch.from_numpy(mislabel_mask.astype(bool)).to(phi_train.device)
    with torch.no_grad():
        n_test = phi_test.size(0)
        test_chunk = 1024
        share_per_test = torch.zeros(
            n_test, device=phi_train.device, dtype=torch.float64
        )
        r_M = residuals[mis_mask_t]                                    # [N_mis, C]
        phi_M = phi_train[mis_mask_t]                                  # [N_mis, d]
        for ts in range(0, n_test, test_chunk):
            te = min(ts + test_chunk, n_test)
            phi_t = phi_test[ts:te]                                    # [chunk, d]
            # v(t)          [C, chunk]
            v = residuals.t() @ (phi_train @ phi_t.t())
            # v_mislabel(t) [C, chunk]
            v_mis = r_M.t() @ (phi_M @ phi_t.t())
            # per-test projection scalar:  <v_mis, v> / ||v||^2
            v_dot_v = (v * v).sum(dim=0)                               # [chunk]
            v_mis_dot_v = (v_mis * v).sum(dim=0)                       # [chunk]
            share_per_test[ts:te] = (
                v_mis_dot_v / v_dot_v.clamp_min(1e-30)
            ).double()
    share_per_test_np = share_per_test.cpu().numpy()
    share_path = output_dir / f"share_per_test_k{k}{file_suffix}.npy"
    np.save(share_path, share_per_test_np)
    mean_share = float(share_per_test_np.mean())
    logger.info(
        f"mislabeled influence share (mean over test) = {mean_share:.4f} "
        f"(baseline mislabel rate = {float(mislabel_mask.mean()):.4f})"
    )

    # Summary stats (legacy: keeps mis/clean ratio for back-compat)
    stats = compute_summary_stats(influence, mislabel_mask)
    stats["mean_mislabel_share"] = mean_share
    stats["mislabel_rate"] = float(mislabel_mask.mean())
    logger.info(
        f"Influence ratio (mislabeled/clean): {stats['influence_ratio']:.3f}"
    )

    # Clean up GPU memory
    del phi_train, phi_test, residuals, influence, model
    torch.cuda.empty_cache()

    return {"metrics": ft_metrics, **stats}


def main():
    parser = argparse.ArgumentParser(
        description="Training point influence decomposition"
    )
    parser.add_argument(
        "--model-dir", required=True,
        help="Directory containing model_k*.pt and evaluation.jsonl",
    )
    parser.add_argument(
        "--data-path", default="./data",
        help="Root directory for CIFAR-10 data",
    )
    parser.add_argument(
        "--lambda-l2", type=float, default=1e-4,
        help="L2 regularization strength for fine-tuning",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Output directory (default: <model-dir>/influence)",
    )
    parser.add_argument(
        "--ts-eval-path", default=None,
        help="Path to temperature_scaled_evaluation.jsonl for Figure X",
    )
    parser.add_argument(
        "--device", default=None,
        help="Device (default: cuda if available, else cpu)",
    )
    parser.add_argument(
        "--use-float64", action="store_true",
        help="Cast features/W to float64 inside L-BFGS. Needed at small lambda "
             "where the FP32 noise floor exceeds the reconstruction tolerance.",
    )
    parser.add_argument(
        "--file-suffix", default=None,
        help="Suffix appended to finetune_k*.pt / validation_k*.json / "
             "influence_k*.jsonl filenames to disambiguate lambda sweeps. "
             "Default: auto = '_lam{lambda:g}_fp{32|64}[_distill]'.",
    )
    parser.add_argument(
        "--distill", action="store_true",
        help="Use Yeh-Kim 3.2 distillation loss: KL(softmax(W_orig*f) || "
             "softmax(W*f)) + lambda * ||W||^2. Anchors W_FT close to the "
             "original trained linear; influence decomposition residual is "
             "computed accordingly.",
    )
    parser.add_argument(
        "--ts-distill", action="store_true",
        help="Apply per-k temperature (from <model-dir>/evaluation.jsonl) "
             "to the distill target AND the L-BFGS init: scales W_orig and "
             "b_orig by 1/T before fine-tuning, so KL starts at 0 and the "
             "distill target is the temperature-scaled prediction. Requires "
             "--distill.",
    )
    parser.add_argument(
        "--k", type=int, default=None,
        help="If set, only process this single k value (smoke test mode).",
    )
    parser.add_argument(
        "--ks", type=int, nargs="+", default=None,
        help="If set, only process these k values (overrides --k).",
    )
    parser.add_argument(
        "--lr", type=float, default=1.0,
        help="L-BFGS step-size multiplier (`lr` of torch.optim.LBFGS). "
             "Default 1.0; try 0.1 for stability at very small lambda.",
    )
    parser.add_argument(
        "--tolerance-grad", type=float, default=1e-7,
        help="L-BFGS gradient-norm convergence tolerance. Default 1e-7. "
             "Needs to be << lambda*||W|| for the analytic decomposition "
             "(which has a 1/lambda amplification) to remain valid.",
    )
    parser.add_argument(
        "--max-iter", type=int, default=200,
        help="L-BFGS max_iter. Default 200.",
    )
    parser.add_argument(
        "--ts-reg", action="store_true",
        help="Scale the L2 regularizer by 1/T^2 per k (T read from "
             "<model-dir>/evaluation.jsonl). Equivalent to penalizing "
             "||W/T||^2: weakens regularization for overconfident "
             "(high-T) models. Loss otherwise unchanged. Not combined "
             "with --ts-distill.",
    )
    args = parser.parse_args()

    if args.ts_distill and not args.distill:
        parser.error("--ts-distill requires --distill")
    if args.ts_reg and args.ts_distill:
        parser.error("--ts-reg and --ts-distill are mutually exclusive")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir) if args.output_dir else model_dir / "influence"
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logger.info(f"Using device: {device}")

    # Discover models
    models = discover_models(str(model_dir))
    if args.ks is not None:
        models = {k: p for k, p in models.items() if k in args.ks}
    elif args.k is not None:
        models = {k: p for k, p in models.items() if k == args.k}
    if not models:
        raise FileNotFoundError(f"No matching model_k*.pt in {model_dir}")
    logger.info(f"Selected {len(models)} models: k={sorted(models.keys())}")

    # Load data with test-time transforms (no augmentation)
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
    )
    test_transform = transforms.Compose([transforms.ToTensor(), normalize])

    # Training set: full 50K with noise, then subset to 45K train indices
    noisy_train_full = NoisyCIFAR10(
        root=args.data_path, train=True, noise_prob=0.15,
        transform=test_transform, seed=42,
    )
    train_indices, _ = compute_val_split_indices()
    train_subset = Subset(noisy_train_full, train_indices.tolist())

    train_loader = DataLoader(
        train_subset, batch_size=256, shuffle=False, num_workers=2,
    )

    # Test set: 10K CIFAR-10 test (no noise)
    test_dataset = NoisyCIFAR10(
        root=args.data_path, train=False, noise_prob=0.0,
        transform=test_transform, seed=42,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=256, shuffle=False, num_workers=2,
    )

    # Build mislabel mask (same for all k)
    mislabel_mask = build_mislabel_mask(
        noisy_train_full.labels,
        noisy_train_full.cifar.targets,
        train_indices,
    )
    n_mislabeled = mislabel_mask.sum()
    logger.info(
        f"Mislabel mask: {n_mislabeled}/{len(mislabel_mask)} "
        f"({100*n_mislabeled/len(mislabel_mask):.1f}%) corrupted"
    )

    # Process each k
    finetuned_metrics = {}
    summary_records = []

    if args.file_suffix is not None:
        file_suffix = args.file_suffix
    else:
        # Auto: _lam1e-05_fp64[_distill][_ts|_tsreg][_lr0.1][_tg1e-10] etc.
        if args.lambda_l2 != 1e-4 or args.use_float64 or args.distill or args.ts_reg:
            file_suffix = f"_lam{args.lambda_l2:g}_fp{64 if args.use_float64 else 32}"
            if args.distill:
                file_suffix += "_distill"
            if args.ts_distill:
                file_suffix += "_ts"
            if args.ts_reg:
                file_suffix += "_tsreg"
        else:
            file_suffix = ""
        if args.lr != 1.0:
            file_suffix += f"_lr{args.lr:g}"
        if args.tolerance_grad != 1e-7:
            file_suffix += f"_tg{args.tolerance_grad:g}"
    if file_suffix:
        logger.info(f"Output file suffix: '{file_suffix}'")

    # Load per-k temperatures if --ts-distill or --ts-reg is on
    k_to_temperature = {}
    if args.ts_distill or args.ts_reg:
        eval_path = model_dir / "evaluation.jsonl"
        if not eval_path.exists():
            raise FileNotFoundError(
                f"--ts-distill/--ts-reg requires {eval_path} with per-k "
                f"'temperature' fields (produced by the training pipeline)."
            )
        with open(eval_path) as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                if "temperature" not in rec:
                    raise KeyError(
                        f"--ts-*: evaluation.jsonl entry for k={rec.get('k')} "
                        f"missing 'temperature' field"
                    )
                k_to_temperature[rec["k"]] = float(rec["temperature"])
        logger.info(
            f"Loaded temperatures for k={sorted(k_to_temperature.keys())} "
            f"from {eval_path}"
        )

    for k, model_path in sorted(models.items()):
        # Per-k effective lambda (only different when --ts-reg is on)
        if args.ts_reg:
            T_k = k_to_temperature[k]
            effective_lambda = args.lambda_l2 / (T_k * T_k)
            logger.info(
                f"k={k}: --ts-reg active, T={T_k:.4f}, "
                f"effective lambda = {args.lambda_l2:g} / T^2 = {effective_lambda:g}"
            )
        else:
            effective_lambda = args.lambda_l2

        result = process_k(
            k=k,
            model_path=model_path,
            train_loader=train_loader,
            test_loader=test_loader,
            mislabel_mask=mislabel_mask,
            train_indices=train_indices,
            noisy_labels=noisy_train_full.labels,
            original_labels=noisy_train_full.cifar.targets,
            lambda_l2=effective_lambda,
            output_dir=output_dir,
            device=device,
            use_float64=args.use_float64,
            file_suffix=file_suffix,
            distill=args.distill,
            temperature=k_to_temperature.get(k) if args.ts_distill else None,
            lr=args.lr,
            tolerance_grad=args.tolerance_grad,
            max_iter=args.max_iter,
        )
        finetuned_metrics[k] = result.pop("metrics")
        summary_records.append({"k": k, **result})

    # Write summary
    summary_path = output_dir / "summary.jsonl"
    with open(summary_path, "w") as f:
        for rec in summary_records:
            f.write(json.dumps(rec) + "\n")
    logger.info(f"Saved summary to {summary_path}")

    # Plot Figure X (temperature-scaled vs L2 fine-tuned test loss)
    ts_eval_path = (
        Path(args.ts_eval_path) if args.ts_eval_path
        else model_dir / "evaluation.jsonl"
    )
    if ts_eval_path.exists():
        try:
            plot_figure_x(
                ts_eval_path, finetuned_metrics,
                output_dir / "figure_x.png",
            )
        except KeyError as e:
            logger.warning(
                f"plot_figure_x skipped: {ts_eval_path} missing TS fields ({e}). "
                "Run with --val-split if you want temperature-scaled comparison."
            )
    else:
        logger.warning(
            f"No temperature_scaled_evaluation.jsonl found at {ts_eval_path}, "
            "skipping Figure X"
        )

    logger.info("Done.")


if __name__ == "__main__":
    main()
