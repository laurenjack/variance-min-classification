"""Fine-tune ResNet18 final layer with L-BFGS + L2 regularization.

Loads trained ResNet18 models, extracts features from the frozen backbone,
and fine-tunes only the final linear layer to reach a stationary point.
This enables training-point decomposition per Yeh & Kim et al. (2018).

Usage:
    python -m jl.double_descent.resnet18.fine_tune \
        --model-path ./output/resnet18/03-01-1010 \
        --data-path ./data \
        --l2-lambda 1e-5 --max-steps 100
"""

import argparse
import json
import logging
from pathlib import Path

import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from jl.double_descent.fine_tune_lib import fine_tune_final_layer
from jl.double_descent.resnet18.evaluation import discover_models
from jl.double_descent.resnet18.resnet18_config import DDConfig
from jl.double_descent.resnet18.resnet18_data import load_cifar10_with_noise
from jl.double_descent.resnet18.resnet18k import make_resnet18k

logger = logging.getLogger(__name__)


def extract_features(model, x: torch.Tensor) -> torch.Tensor:
    """Run forward pass up to (but not including) the final linear layer.

    Returns:
        [B, 8*k] feature tensor.
    """
    out = model.conv1(x)
    out = model.layer1(out)
    out = model.layer2(out)
    out = model.layer3(out)
    out = model.layer4(out)
    out = F.relu(model.bn(out))
    out = F.avg_pool2d(out, 4)
    out = out.view(out.size(0), -1)
    return out


def fine_tune_worker(
    gpu_id: int,
    k: int,
    model_path: str,
    data_path: str,
    output_dir: str,
    l2_lambda: float,
    max_steps: int,
) -> None:
    """Fine-tune a single ResNet18 model on one GPU."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    device = torch.device(f"cuda:{gpu_id}")
    logger.info(f"[k={k}] Starting fine-tuning on GPU {gpu_id}")

    # Load model
    model = make_resnet18k(k=k, num_classes=10).to(device)
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # Load training data (no augmentation for deterministic features)
    config = DDConfig()
    train_loader, _ = load_cifar10_with_noise(
        noise_prob=config.label_noise,
        batch_size=config.batch_size,
        data_augmentation=False,
        data_dir=data_path,
    )

    # Extract features from frozen backbone
    all_features = []
    all_labels = []
    with torch.no_grad():
        for images, labels in train_loader:
            images = images.to(device)
            feats = extract_features(model, images)
            all_features.append(feats)
            all_labels.append(labels.to(device))

    features = torch.cat(all_features, dim=0)
    targets = torch.cat(all_labels, dim=0)
    logger.info(f"[k={k}] Extracted features: {features.shape}")

    # Copy final layer into standalone nn.Linear
    in_features = 8 * k
    linear = nn.Linear(in_features, 10, bias=True).to(device)
    linear.load_state_dict(model.linear.state_dict())

    # Free model memory
    del model
    torch.cuda.empty_cache()

    # Fine-tune
    metadata = fine_tune_final_layer(
        features=features,
        targets=targets,
        linear_layer=linear,
        l2_lambda=l2_lambda,
        max_steps=max_steps,
        device=device,
    )
    metadata["k"] = k

    # Save fine-tuned layer
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    layer_path = out_path / f"layer_k{k}.pt"
    torch.save(linear.state_dict(), layer_path)
    logger.info(f"[k={k}] Saved fine-tuned layer to {layer_path}")

    # Append metadata
    metadata_path = out_path / "fine_tune_metadata.jsonl"
    with open(metadata_path, "a") as f:
        f.write(json.dumps(metadata) + "\n")
    logger.info(
        f"[k={k}] Done: loss={metadata['final_loss']:.6f}, "
        f"grad_norm={metadata['final_grad_norm']:.2e}"
    )


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Fine-tune ResNet18 final layers with L-BFGS + L2"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Directory containing model_k*.pt files",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="./data",
        help="Directory containing CIFAR-10 data",
    )
    parser.add_argument(
        "--l2-lambda",
        type=float,
        default=1e-5,
        help="L2 regularization strength (default: 1e-5)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=100,
        help="Number of L-BFGS steps (default: 100)",
    )
    args = parser.parse_args()

    # Discover models
    models = discover_models(args.model_path)
    if not models:
        raise FileNotFoundError(
            f"No model_k*.pt files found in {args.model_path}"
        )
    logger.info(f"Found models for k values: {list(models.keys())}")

    output_dir = str(Path(args.model_path) / "fine_tuned")

    # Clear existing metadata file
    metadata_path = Path(output_dir) / "fine_tune_metadata.jsonl"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    if metadata_path.exists():
        metadata_path.unlink()

    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("No GPUs available. Fine-tuning requires at least one GPU.")
    logger.info(f"Using {num_gpus} GPUs")

    k_values = list(models.keys())

    # Process in batches of num_gpus
    mp.set_start_method("spawn", force=True)
    for batch_start in range(0, len(k_values), num_gpus):
        batch = k_values[batch_start : batch_start + num_gpus]
        batch_num = batch_start // num_gpus + 1
        total_batches = (len(k_values) + num_gpus - 1) // num_gpus
        logger.info(f"Batch {batch_num}/{total_batches}: k = {batch}")

        processes = []
        for gpu_id, k in enumerate(batch):
            p = mp.Process(
                target=fine_tune_worker,
                args=(
                    gpu_id,
                    k,
                    str(models[k]),
                    args.data_path,
                    output_dir,
                    args.l2_lambda,
                    args.max_steps,
                ),
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
            if p.exitcode != 0:
                logger.error(f"Worker exited with code {p.exitcode}")

        logger.info(f"Batch {batch_num}/{total_batches} complete")

    logger.info(f"Fine-tuning complete. Results in {output_dir}")


if __name__ == "__main__":
    main()
