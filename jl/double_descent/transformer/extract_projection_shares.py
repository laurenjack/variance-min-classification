#!/usr/bin/env python3
"""Compute signed projection-based bucket shares from saved shadow tensors.

For each bucket b with shadow tensor Δ_b (the AdamW-decomposed cumulative
gradient-driven update attributable to that entropy bucket), and a chosen
reference vector V, compute the signed scalar projection coefficient onto V

    s_b = ⟨Δ_b, V⟩ / ‖V‖²

i.e. the scalar you'd multiply V by to get Δ_b's component along V.

Two reference choices via --reference:

  * gradient_sum   V = Σ_b Δ_b (the cumulative gradient-driven trajectory).
                   Then Σ_b s_b = 1 identically (linear identity), and
                   each s_b reads as "fraction of the trajectory built by
                   bucket b". No model needed.

  * final_weight   V = W_T (the trained model parameters). Now Σ_b s_b is
                   ⟨V_grad, W_T⟩ / ‖W_T‖²: the gradient trajectory's
                   scalar projection onto W_T per ‖W_T‖². Tells you what
                   fraction of W_T's magnitude was actually constructed
                   by gradients vs. left over from W_0 + weight-decay
                   self-shrinkage. Requires --model-dir so we can load
                   model_d{d}_{Nk}.pt.

Inputs : bucket_shadows_d{d}_{Nk}.pt + (for final_weight) model_d{d}_{Nk}.pt
Outputs: sibling bucket_projection_shares_d{d}_{Nk}.json carrying the raw
         inner products, the normalized shares, the per-bucket cosine
         alignments, ‖V‖², and the reference choice for documentation.

Usage:
    # Default: project onto W_T
    python -m jl.double_descent.transformer.extract_projection_shares \\
        ./data/transformer_m2m100_shadow_adamw/05-09-0627 \\
        --model-dir ./data/transformer_m2m100_shadow_adamw/05-09-0627

    # Old behavior (project onto Σ_b Δ_b)
    python -m jl.double_descent.transformer.extract_projection_shares \\
        ./data/transformer_m2m100_shadow_adamw/05-09-0627 \\
        --reference gradient_sum
"""

import argparse
import json
import logging
import math
import re
from pathlib import Path
from typing import Optional

import torch

logger = logging.getLogger(__name__)

SHADOW_RE = re.compile(r"bucket_shadows_d(\d+)_(\d+)k\.pt$")


def compute_projection_shares(
    shadow_path: Path,
    reference: str,
    model_path: Optional[Path],
    device: torch.device,
) -> dict:
    ckpt = torch.load(shadow_path, map_location="cpu", weights_only=False)
    shadows = ckpt["shadows"]                      # [n_bins][n_params]
    param_names = ckpt["param_names"]
    n_bins, n_params = len(shadows), len(shadows[0])

    if reference == "final_weight":
        if model_path is None:
            raise ValueError("--model-dir is required when --reference final_weight")
        state = torch.load(model_path, map_location="cpu", weights_only=True)
        try:
            v_per_param = [state[name] for name in param_names]
        except KeyError as e:
            raise KeyError(
                f"Missing parameter in model state_dict: {e}. "
                f"Shadow file param_names: {param_names[:3]}..."
            )
    elif reference == "gradient_sum":
        v_per_param = None  # built per-param below
    else:
        raise ValueError(f"Unknown --reference: {reference}")

    proj = torch.zeros(n_bins, dtype=torch.float64)
    delta_norm_sq = torch.zeros(n_bins, dtype=torch.float64)
    v_norm_sq = torch.zeros((), dtype=torch.float64)

    for p_idx in range(n_params):
        # n_bins shadow tensors for this param, on device, fp64
        deltas = [shadows[b][p_idx].to(device, dtype=torch.float64, non_blocking=True)
                  for b in range(n_bins)]

        if reference == "gradient_sum":
            V_p = torch.zeros_like(deltas[0])
            for d in deltas:
                V_p.add_(d)
        else:  # final_weight
            V_p = v_per_param[p_idx].to(device, dtype=torch.float64, non_blocking=True)

        v_norm_sq += V_p.pow(2).sum().cpu()
        for b in range(n_bins):
            proj[b] += (deltas[b] * V_p).sum().cpu()
            delta_norm_sq[b] += deltas[b].pow(2).sum().cpu()
        del deltas, V_p

    proj_sum = float(proj.sum())
    v_norm_sq_f = float(v_norm_sq)
    if v_norm_sq_f <= 0:
        logger.warning(f"  ‖V‖² ≈ 0 ({v_norm_sq_f:.3e}); shares undefined")
        shares = [float("nan")] * n_bins
        sum_share = float("nan")
    else:
        # Scalar projection coefficients onto V's direction (per ‖V‖²).
        # Sums to ⟨Σ_b Δ_b, V⟩ / ‖V‖²:
        #   - For gradient_sum (V = Σ_b Δ_b), this is identically 1.
        #   - For final_weight (V = W_T), it's the gradient trajectory's
        #     scalar projection onto W_T per ‖W_T‖²; informs how much of
        #     W_T's magnitude was actually built by gradients.
        shares = (proj / v_norm_sq_f).tolist()
        sum_share = proj_sum / v_norm_sq_f

    delta_norms = [math.sqrt(float(x)) for x in delta_norm_sq.tolist()]
    v_norm = math.sqrt(float(v_norm_sq))
    cos_alignments = [
        (float(proj[b]) / (delta_norms[b] * v_norm)) if (delta_norms[b] > 0 and v_norm > 0)
        else float("nan")
        for b in range(n_bins)
    ]

    return {
        "reference_vector": reference,
        "projection_shares": shares,                    # c_b / ‖V‖²
        "projection_shares_sum": sum_share,             # ⟨Σ Δ_b, V⟩ / ‖V‖²
        "projection_inner_products": proj.tolist(),     # raw c_b = ⟨Δ_b, V⟩
        "projection_inner_products_sum": proj_sum,      # ⟨Σ Δ_b, V⟩
        "cos_alignments": cos_alignments,
        "delta_norms": delta_norms,
        "v_norm_sq": float(v_norm_sq),
        "n_bins": n_bins,
        "bucket_entropy_edges": ckpt["bucket_entropy_edges"],
        "bucket_entropy_centers": ckpt["bucket_entropy_centers"],
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source_dir", help="Directory with bucket_shadows_d*.pt files")
    p.add_argument("--reference", default="final_weight",
                   choices=["final_weight", "gradient_sum"],
                   help="Vector to project onto (default: final_weight = W_T)")
    p.add_argument("--model-dir", default=None,
                   help="Directory with model_d*_*k.pt (required when reference=final_weight). "
                        "Defaults to source_dir.")
    p.add_argument("--output-dir", default=None,
                   help="Where to write sibling JSONs (default: source_dir)")
    p.add_argument("--device", default=None,
                   help="cuda / cpu / cuda:N (default: auto)")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(name)s] %(message)s",
                        datefmt="%H:%M:%S")

    src = Path(args.source_dir)
    out = Path(args.output_dir) if args.output_dir else src
    out.mkdir(parents=True, exist_ok=True)
    model_dir = Path(args.model_dir) if args.model_dir else src

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logger.info(f"Device: {device}; reference={args.reference}")

    shadow_files = sorted(src.glob("bucket_shadows_d*_*k.pt"))
    if not shadow_files:
        raise FileNotFoundError(f"No bucket_shadows_d*_*k.pt under {src}")
    logger.info(f"Found {len(shadow_files)} shadow files")

    for sf in shadow_files:
        m = SHADOW_RE.search(sf.name)
        if not m:
            logger.warning(f"  skipping unparsable filename: {sf.name}")
            continue
        d_model = int(m.group(1))
        samples_k = int(m.group(2))

        logger.info(f"--- d={d_model}, {samples_k}k ---")
        model_path = (model_dir / f"model_d{d_model}_{samples_k}k.pt"
                      if args.reference == "final_weight" else None)
        if model_path is not None and not model_path.exists():
            logger.warning(f"  skipping: missing {model_path}")
            continue

        result = compute_projection_shares(sf, args.reference, model_path, device)
        result["d_model"] = d_model
        result["train_samples"] = samples_k * 1000
        result["source_shadow_file"] = sf.name
        if model_path is not None:
            result["source_model_file"] = model_path.name

        logger.info(
            f"  Σ_b ⟨Δ_b, V⟩ = {result['projection_inner_products_sum']:.4e}; "
            f"‖V‖² = {result['v_norm_sq']:.4e}; "
            f"Σ shares = {result['projection_shares_sum']:.6f}  "
            f"(=1 for gradient_sum; gradient/W_T fraction for final_weight)"
        )
        logger.info(
            f"  shares = {[round(x, 4) for x in result['projection_shares']]}"
        )
        logger.info(
            f"  cos    = {[round(x, 4) for x in result['cos_alignments']]}"
        )

        out_path = out / f"bucket_projection_shares_d{d_model}_{samples_k}k.json"
        out_path.write_text(json.dumps(result, indent=2))
        logger.info(f"  wrote {out_path}")


if __name__ == "__main__":
    main()
