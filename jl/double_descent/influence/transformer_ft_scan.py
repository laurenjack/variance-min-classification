#!/usr/bin/env python3
"""Cartesian-product scan over (lambda, lr, beta2) for a single d_model.

Drives N parallel transformer_main jobs (one per GPU), batching as needed.
Each run writes to <output-dir>/lam{X}_lr{Y}_b2{Z}/.

Usage:
    python -m jl.double_descent.influence.transformer_ft_scan \\
        --model-path output/.../model_d40_36k.pt \\
        --d-model 40 \\
        --data-path ./data/iwslt14.m2m100.de-en \\
        --output-dir output/.../distill_scan/d40 \\
        --lambdas 1e-6 3e-6 1e-5 3e-5 \\
        --lrs 1e-2 3e-2 \\
        --beta2s 0.99 0.999 0.9999 \\
        --gpus 0 1 2 3 4 5 6 7 \\
        --distill
"""

import argparse
import itertools
import logging
import os
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def _fmt(x: float) -> str:
    """Compact float-to-string for directory tags (1e-06, 3e-06, 0.99 etc.)."""
    return ("%g" % x).replace("+", "")


def launch_one(d_model: int, model_path: Path, lambda_l2: float, lr: float,
               beta2: float, gpu_id: int, args, out_dir: Path, log_path: Path):
    cmd = [
        sys.executable, "-m", "jl.double_descent.influence.transformer_main",
        "--model-path", str(model_path),
        "--d-model", str(d_model),
        "--split-id", str(args.split_id),
        "--data-path", str(args.data_path),
        "--output-dir", str(out_dir),
        "--lambda-l2", str(lambda_l2),
        "--optimizer", "adam",
        "--num-adam-steps", str(args.num_adam_steps),
        "--adam-lr", str(lr),
        "--adam-warmup-steps", str(args.adam_warmup_steps),
        "--adam-beta1", str(args.adam_beta1),
        "--adam-beta2", str(beta2),
        "--batch-size", str(args.batch_size),
        "--device", f"cuda:{gpu_id}",
    ]
    if args.distill:
        cmd.append("--distill")
    env = os.environ.copy()
    log_file = open(log_path, "w")
    proc = subprocess.Popen(cmd, env=env, stdout=log_file, stderr=subprocess.STDOUT)
    return proc, log_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--d-model", type=int, required=True)
    parser.add_argument("--split-id", type=int, default=0)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--gpus", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5, 6, 7])
    parser.add_argument("--lambdas", type=float, nargs="+", required=True)
    parser.add_argument("--lrs", type=float, nargs="+", required=True)
    parser.add_argument("--beta2s", type=float, nargs="+", required=True)
    parser.add_argument("--num-adam-steps", type=int, default=3000)
    parser.add_argument("--adam-warmup-steps", type=int, default=150)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--distill", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(name)s] %(message)s",
                        datefmt="%H:%M:%S")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    runs = list(itertools.product(args.lambdas, args.lrs, args.beta2s))
    n_gpus = len(args.gpus)
    n_batches = (len(runs) + n_gpus - 1) // n_gpus
    logger.info(f"Total runs: {len(runs)} on {n_gpus} GPUs ({n_batches} batches)")

    for batch_start in range(0, len(runs), n_gpus):
        batch = runs[batch_start : batch_start + n_gpus]
        bidx = batch_start // n_gpus + 1
        logger.info(f"=== Batch {bidx}/{n_batches}: {batch} ===")
        procs = []
        files = []
        for i, (lam, lr, b2) in enumerate(batch):
            gpu_id = args.gpus[i]
            tag = f"lam{_fmt(lam)}_lr{_fmt(lr)}_b2{_fmt(b2)}"
            out_dir = output_dir / tag
            out_dir.mkdir(parents=True, exist_ok=True)
            log_path = log_dir / f"{tag}.log"
            proc, fh = launch_one(
                args.d_model, Path(args.model_path), lam, lr, b2,
                gpu_id, args, out_dir, log_path,
            )
            procs.append((tag, gpu_id, proc, log_path))
            files.append(fh)
            logger.info(f"  {tag} on GPU {gpu_id} (PID {proc.pid})")

        for tag, gpu_id, proc, log_path in procs:
            ret = proc.wait()
            logger.info(f"  {tag} on GPU {gpu_id} done (exit {ret})")
            if ret != 0:
                logger.error(f"  FAILED: {tag}, see {log_path}")
        for fh in files:
            fh.close()

    logger.info("All batches complete.")


if __name__ == "__main__":
    main()
