#!/usr/bin/env python3
"""Main entry point for Transformer Double Descent training.

Reproduces Figure 3 from Nakkiran et al. (2019) "Deep Double Descent":
6-layer encoder-decoder Transformer with varying embedding dimension d_model
trained on IWSLT'14 German-to-English translation.

Trains 24 models: 24 d_model values x 1 sample size (36K).
d_model: 8, 16, 24, ..., 192. 3 batches total, 8 models per batch.

Usage:
    python -m jl.double_descent.transformer.transformer_main \\
        --output-path ./output \\
        --data-path ./data/iwslt14.tokenized.de-en
"""

import argparse
import logging
import os
import time

import torch
import torch.multiprocessing as mp

from jl.double_descent.transformer.transformer_config import TDDConfig
from jl.double_descent.transformer.trainer import train_single_model

# Configure logging with timestamps
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Default experiment parameters (overridable via CLI)
TRAIN_SAMPLES = [36000]  # 36K samples
D_MODEL_VALUES = list(range(8, 392, 8))  # [8, 16, 24, ..., 384] - 48 values (default mode)

# For --variance runs we cap at d=192 (Nakkiran et al. Figure 3 range) so the
# sweep matches the existing 03-24-0859 BPE variance run shape and can be
# compared against it.
VARIANCE_D_MODEL_VALUES = list(range(8, 200, 8))  # [8, 16, ..., 192] - 24 values



def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train Transformer models for Deep Double Descent reproduction"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Path to save training metrics"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="./data/iwslt14.tokenized.de-en",
        help="Path to preprocessed IWSLT'14 data"
    )
    parser.add_argument(
        "--m2m100",
        action="store_true",
        help="Use M2M100-tokenized data (compact ~18K vocab) instead of BPE 10K. "
             "Expects --data-path to contain vocab_mapping.json + *.de.ids / *.en.ids."
    )
    parser.add_argument(
        "--d-models",
        type=int,
        nargs="+",
        default=None,
        help="Subset of d_model values to train (default: full sweep 8..384). "
             "Number of available GPUs (CUDA_VISIBLE_DEVICES-aware) sets the per-batch parallelism."
    )
    parser.add_argument(
        "--variance",
        action="store_true",
        help="Variance mode: train --num-splits independent models per "
             "d_model on disjoint chunks of IWSLT train, with one extra "
             "chunk held out as the in-distribution test set. See "
             "TRANSFORMER_PLAN.md for the held-out-test motivation.",
    )
    parser.add_argument(
        "--num-splits",
        type=int,
        default=4,
        help="Number of disjoint training splits per d_model for --variance "
             "mode (default 4).",
    )
    parser.add_argument(
        "--samples-per-split",
        type=int,
        default=32000,
        help="Samples per split for --variance mode. (num_splits + 1) * "
             "samples_per_split must be <= |IWSLT train| (~160K). Default "
             "32000 → 5 chunks × 32K = 160K, fits exactly.",
    )
    return parser.parse_args()


def run_double_descent(args, config, d_models, num_gpus):
    """Run the double-descent experiment.

    Default mode: one job per d_model.
    Variance mode: one job per (d_model, split_id) for split_id in
        0..num_splits-1.

    All jobs are round-robin'd onto the available GPUs and spawned
    concurrently as one mp.Process each. With more jobs than GPUs the
    excess processes share each GPU — NVIDIA MPS
    (`nvidia-cuda-mps-control -d`) makes kernels from different processes
    actually run in parallel; without MPS they will time-slice.
    """
    total_start = time.time()

    train_samples = TRAIN_SAMPLES[0]
    samples_k = train_samples // 1000

    if config.variance:
        jobs = [
            (d_model, split_id)
            for d_model in d_models
            for split_id in range(config.num_splits)
        ]
    else:
        jobs = [(d_model, None) for d_model in d_models]

    # Round-robin assign jobs to GPUs so each GPU gets a mix of d_models.
    gpu_to_jobs = [[] for _ in range(num_gpus)]
    for i, job in enumerate(jobs):
        gpu_to_jobs[i % num_gpus].append(job)
    max_per_gpu = max(len(j) for j in gpu_to_jobs)

    logger.info("Transformer Double Descent")
    logger.info(f"Mode: {'variance' if config.variance else 'default'}")
    logger.info(f"d_model values: {d_models}")
    if config.variance:
        logger.info(
            f"Variance: {config.num_splits} splits/d_model x "
            f"{config.samples_per_split} samples per split + 1 held-out "
            f"test chunk = {(config.num_splits + 1) * config.samples_per_split} sentences"
        )
    else:
        logger.info(f"Train samples: {train_samples}")
    logger.info(f"Total jobs: {len(jobs)} | GPUs: {num_gpus} | max jobs/GPU: {max_per_gpu}")
    for gpu_id, jjs in enumerate(gpu_to_jobs):
        logger.info(f"  GPU {gpu_id} -> {jjs}")
    logger.info(f"BF16 autocast: {config.use_bf16}")
    mps_pipe = os.environ.get("CUDA_MPS_PIPE_DIRECTORY")
    if mps_pipe:
        logger.info(f"CUDA_MPS_PIPE_DIRECTORY: {mps_pipe}")
    else:
        logger.info(
            "CUDA_MPS_PIPE_DIRECTORY: unset — MPS not detected; models on "
            "the same GPU will time-slice instead of running concurrently."
        )
    logger.info(f"Max steps: {config.max_steps}, Max tokens/batch: {config.max_tokens}")

    logger.info(f"\n{'='*60}")
    logger.info("Starting runs (all processes spawned concurrently)")
    logger.info(f"{'='*60}")

    mp.set_start_method('spawn', force=True)

    processes = []
    for gpu_id, gpu_jobs in enumerate(gpu_to_jobs):
        for d_model, split_id in gpu_jobs:
            p = mp.Process(
                target=train_single_model,
                args=(gpu_id, d_model, train_samples, config,
                      args.output_path, args.data_path, args.m2m100,
                      split_id),
            )
            p.start()
            label = f"d{d_model}" + (f"_split{split_id}" if split_id is not None else "")
            processes.append((gpu_id, label, p))
            logger.info(f"Started process for {label} on GPU {gpu_id} (pid={p.pid})")

    for gpu_id, label, p in processes:
        p.join()
        logger.info(f"Process {label} (GPU {gpu_id}) completed")

    total_time = time.time() - total_start
    logger.info("\n" + "=" * 60)
    logger.info("Full experiment complete!")
    logger.info(f"Total time: {total_time:.2f}s ({total_time/3600:.2f}h)")
    if config.variance:
        logger.info(f"Metrics files: {args.output_path}/metrics_d*_split*.jsonl")
    else:
        logger.info(f"Metrics files: {args.output_path}/metrics_d*_{samples_k}k.jsonl")


def main():
    args = parse_args()
    config = TDDConfig()
    if args.variance:
        config.variance = True
        config.num_splits = args.num_splits
        config.samples_per_split = args.samples_per_split

    # Use whatever GPUs are visible (CUDA_VISIBLE_DEVICES filters this).
    num_gpus = torch.cuda.device_count()
    if num_gpus < 1:
        raise RuntimeError("No CUDA GPUs visible to this process.")

    if args.d_models:
        d_models = args.d_models
    elif config.variance:
        # Variance sweep caps at d=192 to match the existing 03-24-0859
        # BPE variance run, and so the held-out chunk's bias term can be
        # compared at every d_model in that range.
        d_models = VARIANCE_D_MODEL_VALUES
    else:
        d_models = D_MODEL_VALUES

    # Create output directory
    os.makedirs(args.output_path, exist_ok=True)

    # Check for preprocessed data
    if args.m2m100:
        required_files = [
            "vocab_mapping.json",
            "train.de.ids", "train.en.ids",
            "valid.de.ids", "valid.en.ids",
            "test.de.ids", "test.en.ids",
        ]
        prep_hint = "  python -m jl.double_descent.transformer.prepare_m2m100_data\n"
    else:
        required_files = ["train.de", "train.en", "valid.de", "valid.en", "test.de", "test.en", "code"]
        prep_hint = "  ./infra/prepare_iwslt14.sh\n"
    missing_files = [f for f in required_files if not os.path.isfile(os.path.join(args.data_path, f))]
    if missing_files:
        raise FileNotFoundError(
            f"Preprocessed IWSLT'14 data not found at {args.data_path}.\n"
            f"Missing files: {missing_files}\n\n"
            "Please run preprocessing first:\n" + prep_hint
        )
    run_double_descent(args, config, d_models, num_gpus)


if __name__ == "__main__":
    main()
