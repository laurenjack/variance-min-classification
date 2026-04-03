#!/usr/bin/env python3
"""Evaluate fine-tuned final layers for ResNet18 and Transformer models.

Loads each base model, swaps in the fine-tuned final layer, and computes
test loss and test error. Parallelizes across all available GPUs.

Writes fine_tune_evaluation.jsonl into each model's fine_tuned/lambda_*/ directory.

Usage:
    python -m jl.double_descent.fine_tune_evaluation \
        --resnet-path ./output/resnet18/long_double_descent \
        --transformer-path ./output/transformer/long_double_descent_36K \
        --data-path ./data \
        --transformer-data-path ./data/iwslt14.tokenized.de-en \
        --l2-lambda 1e-3
"""

import argparse
import json
import logging
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from jl.double_descent.fine_tune_lib import lambda_dir_name

logger = logging.getLogger(__name__)


def _load_evaluation_jsonl(eval_path: Path) -> Dict:
    """Load evaluation.jsonl into a dict keyed by model width."""
    results = {}
    if not eval_path.exists():
        return results
    with open(eval_path) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                results[r.get("k") or r.get("d_model")] = r
    return results


# --- ResNet18 ---


def _resnet_test_metrics(model, test_loader, device) -> Tuple[float, float, float]:
    """Compute cross-entropy test loss, test error, and ECE for a ResNet18 model.

    Returns:
        (test_loss, test_error, ece)
    """
    from jl.double_descent.resnet18.evaluation import compute_ece

    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    all_confidences = []
    all_correct = []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = F.cross_entropy(logits, labels, reduction="sum")
            total_loss += loss.item()

            probs = F.softmax(logits, dim=-1)
            max_probs, predictions = probs.max(dim=-1)
            correct = predictions == labels

            total_correct += correct.sum().item()
            total_samples += labels.size(0)
            all_confidences.append(max_probs.cpu())
            all_correct.append(correct.cpu())

    test_loss = total_loss / total_samples
    test_error = 1.0 - (total_correct / total_samples)
    ece = compute_ece(torch.cat(all_confidences), torch.cat(all_correct))
    return test_loss, test_error, ece


def _resnet_worker(
    gpu_id: int,
    k: int,
    orig_model_path: str,
    fine_tuned_layer_path: str,
    data_path: str,
    batch_size: int,
    eval_entry: Optional[Dict],
    result_dict: dict,
) -> None:
    """Evaluate one ResNet18 model (original + fine-tuned) on a single GPU."""
    from jl.double_descent.resnet18.resnet18_data import NoisyCIFAR10
    from jl.double_descent.resnet18.resnet18k import make_resnet18k
    import torchvision.transforms as transforms

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    device = torch.device(f"cuda:{gpu_id}")

    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]
    )
    test_dataset = NoisyCIFAR10(
        root=data_path,
        train=False,
        noise_prob=0.0,
        transform=transforms.Compose([transforms.ToTensor(), normalize]),
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )

    # Original test loss/error/ece
    if eval_entry and "test_loss" in eval_entry and "test_error" in eval_entry and "ece" in eval_entry:
        orig_loss = eval_entry["test_loss"]
        orig_error = eval_entry["test_error"]
        orig_ece = eval_entry["ece"]
    else:
        model = make_resnet18k(k=k, num_classes=10).to(device)
        model.load_state_dict(
            torch.load(orig_model_path, map_location=device, weights_only=True)
        )
        orig_loss, orig_error, orig_ece = _resnet_test_metrics(model, test_loader, device)
        del model

    # Fine-tuned test loss/error/ece
    model = make_resnet18k(k=k, num_classes=10).to(device)
    model.load_state_dict(
        torch.load(orig_model_path, map_location=device, weights_only=True)
    )
    layer_state = torch.load(
        fine_tuned_layer_path, map_location=device, weights_only=True
    )
    model.linear.load_state_dict(layer_state)
    ft_loss, ft_error, ft_ece = _resnet_test_metrics(model, test_loader, device)
    del model
    torch.cuda.empty_cache()

    result_dict[k] = (orig_loss, ft_loss, orig_error, ft_error, orig_ece, ft_ece)
    logger.info(
        f"  [GPU {gpu_id}] k={k}: orig_loss={orig_loss:.4f}, ft_loss={ft_loss:.4f}, "
        f"orig_error={orig_error:.4f}, ft_error={ft_error:.4f}, "
        f"orig_ece={orig_ece:.4f}, ft_ece={ft_ece:.4f}"
    )


def evaluate_resnet(
    model_dir: str, data_path: str, l2_lambda: float = 1e-5
) -> Path:
    """Evaluate all ResNet18 models and write fine_tune_evaluation.jsonl.

    Returns:
        Path to the written JSONL file.
    """
    from jl.double_descent.resnet18.evaluation import discover_models
    from jl.double_descent.resnet18.resnet18_config import DDConfig
    import torchvision

    # Pre-download CIFAR-10 before spawning workers to avoid race condition
    torchvision.datasets.CIFAR10(root=data_path, train=False, download=True)

    model_path = Path(model_dir)
    fine_tuned_dir = model_path / "fine_tuned" / lambda_dir_name(l2_lambda)

    models = discover_models(model_dir)
    if not models:
        raise FileNotFoundError(f"No model_k*.pt files in {model_dir}")

    for k in models:
        layer_path = fine_tuned_dir / f"layer_k{k}.pt"
        if not layer_path.exists():
            raise FileNotFoundError(f"Missing fine-tuned layer: {layer_path}")

    eval_results = _load_evaluation_jsonl(model_path / "evaluation.jsonl")
    config = DDConfig()
    num_gpus = torch.cuda.device_count()
    logger.info(f"ResNet evaluation: {len(models)} models across {num_gpus} GPUs")

    sorted_models = sorted(models.items())
    manager = mp.Manager()
    result_dict = manager.dict()

    for batch_start in range(0, len(sorted_models), num_gpus):
        batch = sorted_models[batch_start : batch_start + num_gpus]
        processes = []
        for gpu_id, (k, orig_model_path) in enumerate(batch):
            p = mp.Process(
                target=_resnet_worker,
                args=(
                    gpu_id,
                    k,
                    str(orig_model_path),
                    str(fine_tuned_dir / f"layer_k{k}.pt"),
                    data_path,
                    config.batch_size,
                    eval_results.get(k),
                    result_dict,
                ),
            )
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
            if p.exitcode != 0:
                logger.error(f"ResNet worker exited with code {p.exitcode}")

    # Write results
    output_path = fine_tuned_dir / "fine_tune_evaluation.jsonl"
    with open(output_path, "w") as f:
        for k, _ in sorted_models:
            orig_loss, ft_loss, orig_error, ft_error, orig_ece, ft_ece = result_dict[k]
            entry = {
                "k": k,
                "original_loss": round(orig_loss, 6),
                "fine_tuned_loss": round(ft_loss, 6),
                "original_error": round(orig_error, 6),
                "fine_tuned_error": round(ft_error, 6),
                "original_ece": round(orig_ece, 6),
                "fine_tuned_ece": round(ft_ece, 6),
            }
            f.write(json.dumps(entry) + "\n")

    logger.info(f"Wrote {len(sorted_models)} entries to {output_path}")
    return output_path


# --- Transformer ---


def _transformer_test_metrics(model, test_loader, pad_idx, device) -> Tuple[float, float]:
    """Compute cross-entropy test loss and test error for a Transformer model.

    Returns:
        (test_loss, test_error)
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    with torch.no_grad():
        for src, tgt in test_loader:
            src, tgt = src.to(device), tgt.to(device)
            logits = model(src, tgt[:, :-1])
            target = tgt[:, 1:].contiguous()
            mask = target != pad_idx
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                target.view(-1),
                ignore_index=pad_idx,
                reduction="sum",
            )
            total_loss += loss.item()
            predictions = logits.argmax(dim=-1)
            total_correct += ((predictions == target) & mask).sum().item()
            total_tokens += mask.sum().item()
    test_loss = total_loss / total_tokens
    test_error = 1.0 - (total_correct / total_tokens) if total_tokens > 0 else 1.0
    return test_loss, test_error


def _transformer_worker(
    gpu_id: int,
    d_model: int,
    train_samples: int,
    orig_model_path: str,
    fine_tuned_layer_path: str,
    data_path: str,
    eval_entry: Optional[Dict],
    result_dict: dict,
) -> None:
    """Evaluate one Transformer model (original + fine-tuned) on a single GPU."""
    from jl.double_descent.transformer.transformer_config import TDDConfig
    from jl.double_descent.transformer.transformer_data import (
        build_vocab,
        collate_fn,
        load_split,
        TranslationDataset,
    )
    from jl.double_descent.transformer.transformer_model import TransformerModel

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    device = torch.device(f"cuda:{gpu_id}")
    config = TDDConfig()
    vocab = build_vocab(data_path)
    test_src, test_tgt = load_split(data_path, "test")
    test_dataset = TranslationDataset(test_src, test_tgt, vocab)
    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
        collate_fn=partial(collate_fn, pad_idx=vocab.pad_idx),
    )

    def _make_model():
        return TransformerModel(
            vocab_size=len(vocab),
            d_model=d_model,
            n_layers=config.n_layers,
            n_heads=config.n_heads,
            d_ff_multiplier=config.d_ff_multiplier,
            pad_idx=vocab.pad_idx,
        ).to(device)

    from jl.double_descent.transformer.bleu import compute_bleu

    # Original test loss/error/bleu
    if eval_entry and "test_loss" in eval_entry and "test_acc" in eval_entry and "test_bleu" in eval_entry:
        orig_loss = eval_entry["test_loss"]
        orig_error = 1.0 - eval_entry["test_acc"]
        orig_bleu = eval_entry["test_bleu"]
    else:
        model = _make_model()
        model.load_state_dict(
            torch.load(orig_model_path, map_location=device, weights_only=True)
        )
        orig_loss, orig_error = _transformer_test_metrics(model, test_loader, vocab.pad_idx, device)
        orig_bleu = compute_bleu(model, test_dataset, vocab, device, max_len=128)
        del model

    # Fine-tuned test loss/error/bleu
    model = _make_model()
    model.load_state_dict(
        torch.load(orig_model_path, map_location=device, weights_only=True)
    )
    new_linear = nn.Linear(d_model, len(vocab), bias=False).to(device)
    layer_state = torch.load(
        fine_tuned_layer_path, map_location=device, weights_only=True
    )
    new_linear.load_state_dict(layer_state)
    model.output_proj = new_linear
    ft_loss, ft_error = _transformer_test_metrics(model, test_loader, vocab.pad_idx, device)
    ft_bleu = compute_bleu(model, test_dataset, vocab, device, max_len=128)
    del model
    torch.cuda.empty_cache()

    result_dict[(d_model, train_samples)] = (orig_loss, ft_loss, orig_error, ft_error, orig_bleu, ft_bleu)
    logger.info(
        f"  [GPU {gpu_id}] d_model={d_model}: orig_loss={orig_loss:.4f}, ft_loss={ft_loss:.4f}, "
        f"orig_error={orig_error:.4f}, ft_error={ft_error:.4f}, "
        f"orig_bleu={orig_bleu:.2f}, ft_bleu={ft_bleu:.2f}"
    )


def evaluate_transformer(
    model_dir: str, data_path: str, l2_lambda: float = 1e-5
) -> Path:
    """Evaluate all Transformer models and write fine_tune_evaluation.jsonl.

    Returns:
        Path to the written JSONL file.
    """
    from jl.double_descent.transformer.evaluation import discover_models

    model_path = Path(model_dir)
    fine_tuned_dir = model_path / "fine_tuned" / lambda_dir_name(l2_lambda)

    models = discover_models(model_dir)
    if not models:
        raise FileNotFoundError(f"No model_d*_*k.pt files in {model_dir}")

    for (d_model, train_samples) in models:
        samples_k = train_samples // 1000
        layer_path = fine_tuned_dir / f"layer_d{d_model}_{samples_k}k.pt"
        if not layer_path.exists():
            raise FileNotFoundError(f"Missing fine-tuned layer: {layer_path}")

    eval_results = _load_evaluation_jsonl(model_path / "evaluation.jsonl")
    num_gpus = torch.cuda.device_count()
    logger.info(f"Transformer evaluation: {len(models)} models across {num_gpus} GPUs")

    sorted_models = sorted(models.items())
    manager = mp.Manager()
    result_dict = manager.dict()

    for batch_start in range(0, len(sorted_models), num_gpus):
        batch = sorted_models[batch_start : batch_start + num_gpus]
        processes = []
        for gpu_id, ((d_model, train_samples), orig_model_path) in enumerate(batch):
            samples_k = train_samples // 1000
            p = mp.Process(
                target=_transformer_worker,
                args=(
                    gpu_id,
                    d_model,
                    train_samples,
                    str(orig_model_path),
                    str(fine_tuned_dir / f"layer_d{d_model}_{samples_k}k.pt"),
                    data_path,
                    eval_results.get(d_model),
                    result_dict,
                ),
            )
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
            if p.exitcode != 0:
                logger.error(f"Transformer worker exited with code {p.exitcode}")

    # Write results
    output_path = fine_tuned_dir / "fine_tune_evaluation.jsonl"
    with open(output_path, "w") as f:
        for (d_model, train_samples), _ in sorted_models:
            orig_loss, ft_loss, orig_error, ft_error, orig_bleu, ft_bleu = result_dict[(d_model, train_samples)]
            entry = {
                "d_model": d_model,
                "original_loss": round(orig_loss, 6),
                "fine_tuned_loss": round(ft_loss, 6),
                "original_error": round(orig_error, 6),
                "fine_tuned_error": round(ft_error, 6),
                "original_bleu": round(orig_bleu, 2),
                "fine_tuned_bleu": round(ft_bleu, 2),
            }
            f.write(json.dumps(entry) + "\n")

    logger.info(f"Wrote {len(sorted_models)} entries to {output_path}")
    return output_path


# --- Main ---


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Evaluate fine-tuned final layers (writes fine_tune_evaluation.jsonl)"
    )
    parser.add_argument(
        "--resnet-path",
        type=str,
        default=None,
        help="Directory containing ResNet18 model_k*.pt files",
    )
    parser.add_argument(
        "--transformer-path",
        type=str,
        default=None,
        help="Directory containing Transformer model_d*_*k.pt files",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="./data",
        help="Directory containing CIFAR-10 data (for ResNet evaluation)",
    )
    parser.add_argument(
        "--transformer-data-path",
        type=str,
        default=None,
        help="Directory containing preprocessed IWSLT data (for Transformer evaluation)",
    )
    parser.add_argument(
        "--l2-lambda",
        type=float,
        default=1e-5,
        help="L2 lambda value (determines which fine_tuned/lambda_* subdir to use)",
    )
    args = parser.parse_args()

    if args.resnet_path is None and args.transformer_path is None:
        parser.error("At least one of --resnet-path or --transformer-path is required")

    mp.set_start_method("spawn", force=True)

    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("No GPUs available. Fine-tune evaluation requires at least one GPU.")
    logger.info(f"Using {num_gpus} GPUs")

    if args.resnet_path:
        logger.info("Evaluating ResNet18 fine-tuned models...")
        path = evaluate_resnet(args.resnet_path, args.data_path, args.l2_lambda)
        logger.info(f"ResNet results: {path}")

    if args.transformer_path:
        t_data_path = args.transformer_data_path
        if t_data_path is None:
            parser.error("--transformer-data-path is required when --transformer-path is set")
        logger.info("Evaluating Transformer fine-tuned models...")
        path = evaluate_transformer(args.transformer_path, t_data_path, args.l2_lambda)
        logger.info(f"Transformer results: {path}")

    logger.info("Fine-tune evaluation complete.")


if __name__ == "__main__":
    main()
