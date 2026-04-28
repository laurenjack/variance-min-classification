#!/usr/bin/env python3
"""Sweep Adam fine-tune of the output projection across d_model values.

Runs N parallel transformer_main jobs (one per GPU, by CUDA_VISIBLE_DEVICES),
aggregates the per-d_model {original_test_loss, ft_test_loss} numbers, and
plots both vs d_model so we can see whether the L2 fine-tune preserves /
destroys the double-descent shape we'd see in the original test loss curve.

Each job uses Adam (no L-BFGS polish — verified to be a no-op when Adam
reaches its noise floor at lambda=1e-3).

Usage:
    python -m jl.double_descent.influence.transformer_ft_sweep \\
        --model-dir ./data/transformer_m2m100_variance/04-02-1520 \\
        --d-models 16 32 48 64 80 96 \\
        --split-id 0 \\
        --data-path ./data/iwslt14.m2m100.de-en \\
        --output-dir ./data/transformer_m2m100_variance/04-02-1520/ft_sweep \\
        --gpus 0 1 2 3 4 5
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)


def launch_one(d_model: int, gpu_id: int, args, log_path: Path):
    """Spawn transformer_main on a single GPU."""
    out_dir = Path(args.output_dir) / f"d{d_model}_split{args.split_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "jl.double_descent.influence.transformer_main",
        "--model-path", str(Path(args.model_dir) / f"model_d{d_model}_split{args.split_id}.pt"),
        "--d-model", str(d_model),
        "--split-id", str(args.split_id),
        "--data-path", str(args.data_path),
        "--output-dir", str(out_dir),
        "--lambda-l2", str(args.lambda_l2),
        "--optimizer", "adam",
        "--num-adam-steps", str(args.num_adam_steps),
        "--adam-lr", str(args.adam_lr),
        "--adam-warmup-steps", str(args.adam_warmup_steps),
        "--adam-beta2", str(args.adam_beta2),
        "--batch-size", str(args.batch_size),
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    log_file = open(log_path, "w")
    proc = subprocess.Popen(cmd, env=env, stdout=log_file, stderr=subprocess.STDOUT)
    return proc, log_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True,
                        help="Directory containing model_d{N}_split{K}.pt")
    parser.add_argument("--d-models", type=int, nargs="+",
                        default=[16, 32, 48, 64, 80, 96])
    parser.add_argument("--split-id", type=int, default=0)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--gpus", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5])
    parser.add_argument("--lambda-l2", type=float, default=1e-3)
    parser.add_argument("--num-adam-steps", type=int, default=3000)
    parser.add_argument("--adam-lr", type=float, default=1e-2)
    parser.add_argument("--adam-warmup-steps", type=int, default=150)
    parser.add_argument("--adam-beta2", type=float, default=0.9999)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    if len(args.d_models) > len(args.gpus):
        logger.warning(
            f"More d_models ({len(args.d_models)}) than GPUs ({len(args.gpus)}); "
            "running in batches."
        )

    # Launch in batches of len(args.gpus)
    summaries = {}
    n_gpus = len(args.gpus)
    for batch_start in range(0, len(args.d_models), n_gpus):
        batch = args.d_models[batch_start : batch_start + n_gpus]
        logger.info(f"=== Launching batch: d_models={batch} on GPUs={args.gpus[:len(batch)]} ===")
        procs = []
        files = []
        for i, d_model in enumerate(batch):
            gpu_id = args.gpus[i]
            log_path = log_dir / f"d{d_model}.log"
            proc, fh = launch_one(d_model, gpu_id, args, log_path)
            procs.append((d_model, gpu_id, proc, log_path))
            files.append(fh)
            logger.info(f"  d={d_model} on GPU {gpu_id} (PID {proc.pid}, log={log_path})")

        # Wait for all in this batch
        for d_model, gpu_id, proc, log_path in procs:
            ret = proc.wait()
            logger.info(f"  d={d_model} on GPU {gpu_id} done (exit {ret})")
            if ret != 0:
                logger.error(f"  FAILED: d={d_model}, see {log_path}")
                continue
            val_path = (output_dir / f"d{d_model}_split{args.split_id}" / "validation.json")
            if val_path.exists():
                with open(val_path) as fh:
                    summaries[d_model] = json.load(fh)
        for fh in files:
            fh.close()

    # Aggregate
    rows = []
    for d_model in sorted(summaries.keys()):
        s = summaries[d_model]
        rows.append({
            "d_model": d_model,
            "split_id": args.split_id,
            "original_test_loss": s["original_test_loss"],
            "ft_test_loss": s["ft_test_loss"],
            "ft_test_loss_delta": s["ft_test_loss_delta"],
            "lambda_l2": args.lambda_l2,
            "adam_grad_norm": s["lbfgs"].get("grad_norm") if s["lbfgs"] else None,
            "kl_div_per_token": s["decomposition"]["kl_div_per_token"],
        })
    out_jsonl = output_dir / "ft_sweep.jsonl"
    with open(out_jsonl, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    logger.info(f"Wrote {out_jsonl}")

    # Plot
    try:
        import matplotlib.pyplot as plt
        plt.rcParams.update({"font.family": "serif", "font.size": 11})
        ds = [r["d_model"] for r in rows]
        orig = [r["original_test_loss"] for r in rows]
        ft = [r["ft_test_loss"] for r in rows]
        fig, ax = plt.subplots(1, 1, figsize=(8, 5), dpi=150)
        ax.plot(ds, orig, "o-", color="#1f77b4", markersize=5, markerfacecolor="white",
                markeredgewidth=1.2, linewidth=1.5, label="Original (tied output_proj)")
        ax.plot(ds, ft, "o-", color="#d62728", markersize=5, markerfacecolor="white",
                markeredgewidth=1.2, linewidth=1.5,
                label=f"Adam fine-tuned (λ={args.lambda_l2})")
        ax.set_xlabel("d_model")
        ax.set_ylabel("Test loss (cross-entropy, nats / token)")
        ax.set_title("Test loss vs d_model: original vs L2-fine-tuned output projection")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        plot_path = output_dir / "ft_sweep_test_loss.png"
        fig.savefig(plot_path, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Wrote {plot_path}")
    except Exception as e:
        logger.error(f"Plot failed: {e}")


if __name__ == "__main__":
    main()
