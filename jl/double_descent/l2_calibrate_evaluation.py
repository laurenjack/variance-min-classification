#!/usr/bin/env python3
"""Evaluate calibration approaches on trained models.

Two modes (mutually exclusive):
  --l2-calibrate: Evaluates L2-calibrated final layers vs originals.
  --temperature-scaling: Fits per-model temperature T on test NLL, evaluates with logits/T.

Parallelizes across all available GPUs.

Usage:
    # L2 calibrate evaluation (pass layer directories directly)
    python -m jl.double_descent.l2_calibrate_evaluation --l2-calibrate \
        --resnet-path ./output/resnet18/long_double_descent \
        --resnet-layer-dir ./output/resnet18/long_double_descent/l2_calibrated/lambda_1e-03 \
        --transformer-path ./output/transformer/long_double_descent_36K \
        --transformer-layer-dir ./output/transformer/long_double_descent_36K/l2_calibrated/lambda_1e-03 \
        --data-path ./data \
        --transformer-data-path ./data/iwslt14.tokenized.de-en

    # Temperature scaling evaluation
    python -m jl.double_descent.l2_calibrate_evaluation --temperature-scaling \
        --resnet-path ./output/resnet18/long_double_descent \
        --transformer-path ./output/transformer/long_double_descent_36K \
        --data-path ./data \
        --transformer-data-path ./data/iwslt14.tokenized.de-en
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


def _resnet_metrics_from_logits(
    logits: torch.Tensor, labels: torch.Tensor, temperature: float = 1.0
) -> Tuple[float, float, float]:
    """Compute (test_loss, test_error, ece) from precomputed logits/labels.

    Applies optional temperature scaling. Used for both the original
    (T=1.0) and temperature-scaled evaluation on a held-out split.
    """
    from jl.double_descent.resnet18.evaluation import compute_ece

    scaled = logits / temperature
    loss = F.cross_entropy(scaled, labels).item()
    probs = F.softmax(scaled, dim=-1)
    max_probs, preds = probs.max(dim=-1)
    correct = preds == labels
    error = 1.0 - correct.float().mean().item()
    ece = compute_ece(max_probs, correct)
    return loss, error, ece


def _fit_resnet_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Fit scalar temperature T via L-BFGS on precomputed logits/labels.

    Takes a held-out fit split (not the full test set) so callers can
    reserve the rest of the test set for evaluation.
    """
    temperature = nn.Parameter(torch.ones(1))
    # lr=1.0, max_iter=200 (see ca412cc): lr=0.01 bails out at T~1.2 when the
    # true optimum is far away (e.g. T~2.5 for overconfident models).
    optimizer = torch.optim.LBFGS([temperature], lr=1.0, max_iter=200)

    def closure():
        optimizer.zero_grad()
        loss = F.cross_entropy(logits / temperature, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return temperature.item()


def _resnet_worker(
    gpu_id: int,
    k: int,
    orig_model_path: str,
    l2_calibrated_layer_path: str,
    data_path: str,
    batch_size: int,
    eval_entry: Optional[Dict],
    result_dict: dict,
) -> None:
    """Evaluate one ResNet18 model (original + L2-calibrated) on a single GPU."""
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

    # L2-calibrated test loss/error/ece
    model = make_resnet18k(k=k, num_classes=10).to(device)
    model.load_state_dict(
        torch.load(orig_model_path, map_location=device, weights_only=True)
    )
    layer_state = torch.load(
        l2_calibrated_layer_path, map_location=device, weights_only=True
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
    model_dir: str, layer_dir: str, data_path: str
) -> Path:
    """Evaluate all ResNet18 models and write l2_calibrate_evaluation.jsonl.

    Args:
        model_dir: Directory containing model_k*.pt base model files.
        layer_dir: Directory containing layer_k*.pt L2-calibrated layer files.
        data_path: Directory containing CIFAR-10 data.

    Returns:
        Path to the written JSONL file.
    """
    from jl.double_descent.resnet18.evaluation import discover_models
    from jl.double_descent.resnet18.resnet18_config import DDConfig
    import torchvision

    # Pre-download CIFAR-10 before spawning workers to avoid race condition
    torchvision.datasets.CIFAR10(root=data_path, train=False, download=True)

    model_path = Path(model_dir)
    l2_calibrated_dir = Path(layer_dir)

    models = discover_models(model_dir)
    if not models:
        raise FileNotFoundError(f"No model_k*.pt files in {model_dir}")

    for k in models:
        layer_path = l2_calibrated_dir / f"layer_k{k}.pt"
        if not layer_path.exists():
            raise FileNotFoundError(f"Missing L2-calibrated layer: {layer_path}")

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
                    str(l2_calibrated_dir / f"layer_k{k}.pt"),
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
    output_path = l2_calibrated_dir / "l2_calibrate_evaluation.jsonl"
    with open(output_path, "w") as f:
        for k, _ in sorted_models:
            orig_loss, ft_loss, orig_error, ft_error, orig_ece, ft_ece = result_dict[k]
            entry = {
                "k": k,
                "original_loss": round(orig_loss, 6),
                "l2_calibrated_loss": round(ft_loss, 6),
                "original_error": round(orig_error, 6),
                "l2_calibrated_error": round(ft_error, 6),
                "original_ece": round(orig_ece, 6),
                "l2_calibrated_ece": round(ft_ece, 6),
            }
            f.write(json.dumps(entry) + "\n")

    logger.info(f"Wrote {len(sorted_models)} entries to {output_path}")
    return output_path


def ts_evaluate_resnet(
    model_dir: str, data_path: str
) -> Path:
    """Temperature-scale all ResNet18 models and write temperature_scaled_evaluation.jsonl.

    Single-process, single-GPU implementation. Uses the proper Guo et al.
    (2017) protocol:
      - If val.pt exists in model_dir: fit T on the 5K held-out val set
        (noisy labels, same distribution as training), evaluate on the full
        10K clean test set.
      - Fallback (no val.pt): deterministic 5K/5K split of the test set
        for fit / eval (legacy behaviour for older runs).

    The test set (or val set) is loaded once into GPU memory and reused
    across every model.

    Returns:
        Path to the written JSONL file.
    """
    from jl.double_descent.resnet18.evaluation import discover_models
    from jl.double_descent.resnet18.resnet18_data import NoisyCIFAR10
    from jl.double_descent.resnet18.resnet18k import make_resnet18k
    import torchvision
    import torchvision.transforms as transforms

    torchvision.datasets.CIFAR10(root=data_path, train=False, download=True)

    model_path = Path(model_dir)
    models = discover_models(model_dir)
    if not models:
        raise FileNotFoundError(f"No model_k*.pt files in {model_dir}")

    if torch.cuda.device_count() == 0:
        raise RuntimeError("ts_evaluate_resnet requires a CUDA GPU")
    device = torch.device("cuda:0")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # Load the entire CIFAR-10 test set into GPU memory (~30 MB).
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]
    )
    test_dataset = NoisyCIFAR10(
        root=data_path,
        train=False,
        noise_prob=0.0,
        transform=transforms.Compose([transforms.ToTensor(), normalize]),
    )
    loader = DataLoader(test_dataset, batch_size=1000, shuffle=False, num_workers=0)
    all_test_images = []
    all_test_labels = []
    for images, labels in loader:
        all_test_images.append(images)
        all_test_labels.append(labels)
    all_test_images = torch.cat(all_test_images).to(device)  # [10000, 3, 32, 32]
    all_test_labels = torch.cat(all_test_labels).to(device)  # [10000]

    # Check for val.pt (proper Guo et al. protocol).
    val_pt_path = model_path / "val.pt"
    if val_pt_path.exists():
        val_data = torch.load(str(val_pt_path), map_location=device, weights_only=True)
        val_images = val_data["images"].to(device)   # [5000, 3, 32, 32]
        val_labels = val_data["labels"].to(device)    # [5000]
        use_val_split = True
        logger.info(
            f"Loaded val.pt ({val_images.size(0)} samples) — "
            f"fitting T on val, evaluating on full 10K test set"
        )
    else:
        use_val_split = False
        logger.info(
            "No val.pt found — falling back to 5K/5K test-set split "
            "(legacy mode, not proper Guo et al. protocol)"
        )

    sorted_models = sorted(models.items())
    logger.info(
        f"ResNet TS evaluation: {len(sorted_models)} models on {device} "
        f"(single-process)"
    )

    results: List[Tuple[int, float, float, float, float, float, float, float]] = []
    forward_batch = 512
    for k, orig_model_path in sorted_models:
        model = make_resnet18k(k=k, num_classes=10).to(device)
        model.load_state_dict(
            torch.load(str(orig_model_path), map_location=device, weights_only=True)
        )
        model.eval()

        # Forward pass on test images.
        test_logits = []
        with torch.no_grad():
            for i in range(0, all_test_images.size(0), forward_batch):
                test_logits.append(model(all_test_images[i : i + forward_batch]))
        test_logits = torch.cat(test_logits)  # [10000, 10]

        if use_val_split:
            # Forward pass on val images for fitting T.
            val_logits = []
            with torch.no_grad():
                for i in range(0, val_images.size(0), forward_batch):
                    val_logits.append(model(val_images[i : i + forward_batch]))
            val_logits_t = torch.cat(val_logits).cpu()  # [5000, 10]
            val_labels_cpu = val_labels.cpu()

            temp = _fit_resnet_temperature(val_logits_t, val_labels_cpu)

            # Evaluate on the full 10K test set.
            test_logits_cpu = test_logits.cpu()
            test_labels_cpu = all_test_labels.cpu()
            orig_loss, orig_error, orig_ece = _resnet_metrics_from_logits(
                test_logits_cpu, test_labels_cpu, temperature=1.0
            )
            ts_loss, ts_error, ts_ece = _resnet_metrics_from_logits(
                test_logits_cpu, test_labels_cpu, temperature=temp
            )
        else:
            # Legacy fallback: 5K/5K split of the test set.
            gen = torch.Generator()
            gen.manual_seed(42)
            perm = torch.randperm(all_test_images.size(0), generator=gen)
            half = all_test_images.size(0) // 2
            fit_idx = perm[:half]
            eval_idx = perm[half:]

            logits_cpu = test_logits.cpu()
            labels_cpu = all_test_labels.cpu()

            temp = _fit_resnet_temperature(logits_cpu[fit_idx], labels_cpu[fit_idx])
            orig_loss, orig_error, orig_ece = _resnet_metrics_from_logits(
                logits_cpu[eval_idx], labels_cpu[eval_idx], temperature=1.0
            )
            ts_loss, ts_error, ts_ece = _resnet_metrics_from_logits(
                logits_cpu[eval_idx], labels_cpu[eval_idx], temperature=temp
            )

        results.append(
            (k, orig_loss, ts_loss, orig_error, ts_error, orig_ece, ts_ece, temp)
        )
        logger.info(
            f"  k={k}: T={temp:.4f}, orig_loss={orig_loss:.4f}, ts_loss={ts_loss:.4f}, "
            f"orig_error={orig_error:.4f}, ts_error={ts_error:.4f}, "
            f"orig_ece={orig_ece:.4f}, ts_ece={ts_ece:.4f}"
        )

        del model, test_logits
        torch.cuda.empty_cache()

    output_dir = model_path / "temperature_scaled"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "temperature_scaled_evaluation.jsonl"
    with open(output_path, "w") as f:
        for k, orig_loss, ts_loss, orig_error, ts_error, orig_ece, ts_ece, temp in results:
            entry = {
                "k": k,
                "original_loss": round(orig_loss, 6),
                "ts_loss": round(ts_loss, 6),
                "original_error": round(orig_error, 6),
                "ts_error": round(ts_error, 6),
                "original_ece": round(orig_ece, 6),
                "ts_ece": round(ts_ece, 6),
                "temperature": round(temp, 6),
            }
            f.write(json.dumps(entry) + "\n")

    logger.info(f"Wrote {len(results)} entries to {output_path}")
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


def _collect_transformer_flat_logits(
    model, loader, pad_idx: int, device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Forward a transformer loader, return flat (non-pad) logits & targets.

    Returns:
        logits: [N, V] on `device` — logit vectors at non-pad target positions
        targets: [N] on `device` — target token ids at non-pad positions
    """
    model.eval()
    all_logits = []
    all_targets = []
    with torch.no_grad():
        for src, tgt in loader:
            src, tgt = src.to(device), tgt.to(device)
            logits = model(src, tgt[:, :-1])   # [B, L-1, V]
            target = tgt[:, 1:].contiguous()    # [B, L-1]
            mask = target != pad_idx            # [B, L-1]
            all_logits.append(logits[mask])     # [K_batch, V]
            all_targets.append(target[mask])    # [K_batch]
    return torch.cat(all_logits), torch.cat(all_targets)


def _transformer_metrics_from_logits(
    logits: torch.Tensor, targets: torch.Tensor, temperature: float = 1.0
) -> Tuple[float, float]:
    """Compute (loss, token-level error) on flat non-pad logits/targets.

    No ECE — token-level ECE over a large vocabulary is not meaningful here
    (see user note / TRANSFORMER_PLAN.md).
    """
    scaled = logits / temperature
    loss = F.cross_entropy(scaled, targets).item()
    preds = scaled.argmax(dim=-1)
    error = 1.0 - (preds == targets).float().mean().item()
    return loss, error


def _transformer_worker(
    gpu_id: int,
    d_model: int,
    train_samples: int,
    orig_model_path: str,
    l2_calibrated_layer_path: str,
    data_path: str,
    eval_entry: Optional[Dict],
    result_dict: dict,
) -> None:
    """Evaluate one Transformer model (original + L2-calibrated) on a single GPU."""
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

    # L2-calibrated test loss/error/bleu
    model = _make_model()
    model.load_state_dict(
        torch.load(orig_model_path, map_location=device, weights_only=True)
    )
    new_linear = nn.Linear(d_model, len(vocab), bias=False).to(device)
    layer_state = torch.load(
        l2_calibrated_layer_path, map_location=device, weights_only=True
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
    model_dir: str, layer_dir: str, data_path: str
) -> Path:
    """Evaluate all Transformer models and write l2_calibrate_evaluation.jsonl.

    Args:
        model_dir: Directory containing model_d*_*k.pt base model files.
        layer_dir: Directory containing layer_d*_*k.pt L2-calibrated layer files.
        data_path: Directory containing preprocessed IWSLT data.

    Returns:
        Path to the written JSONL file.
    """
    from jl.double_descent.transformer.evaluation import discover_models

    model_path = Path(model_dir)
    l2_calibrated_dir = Path(layer_dir)

    models = discover_models(model_dir)
    if not models:
        raise FileNotFoundError(f"No model_d*_*k.pt files in {model_dir}")

    for (d_model, train_samples) in models:
        samples_k = train_samples // 1000
        layer_path = l2_calibrated_dir / f"layer_d{d_model}_{samples_k}k.pt"
        if not layer_path.exists():
            raise FileNotFoundError(f"Missing L2-calibrated layer: {layer_path}")

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
                    str(l2_calibrated_dir / f"layer_d{d_model}_{samples_k}k.pt"),
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
    output_path = l2_calibrated_dir / "l2_calibrate_evaluation.jsonl"
    with open(output_path, "w") as f:
        for (d_model, train_samples), _ in sorted_models:
            orig_loss, ft_loss, orig_error, ft_error, orig_bleu, ft_bleu = result_dict[(d_model, train_samples)]
            entry = {
                "d_model": d_model,
                "original_loss": round(orig_loss, 6),
                "l2_calibrated_loss": round(ft_loss, 6),
                "original_error": round(orig_error, 6),
                "l2_calibrated_error": round(ft_error, 6),
                "original_bleu": round(orig_bleu, 2),
                "l2_calibrated_bleu": round(ft_bleu, 2),
            }
            f.write(json.dumps(entry) + "\n")

    logger.info(f"Wrote {len(sorted_models)} entries to {output_path}")
    return output_path


def ts_evaluate_transformer(
    model_dir: str, data_path: str
) -> Path:
    """Temperature-scale all Transformer models and write temperature_scaled_evaluation.jsonl.

    Single-process, single-GPU implementation. For each model:
      1. Forward-pass the IWSLT `valid` split, cache flat non-pad logits on GPU
      2. Forward-pass the `test` split, cache flat non-pad logits on GPU
      3. Fit scalar temperature T on the cached valid logits via L-BFGS
         (proper Guo et al. 2017 protocol — no test-set contamination)
      4. Compute original + temperature-scaled metrics on the cached test logits
      5. BLEU is reused from evaluation.jsonl when available (temperature does
         not change argmax, so greedy-decode BLEU is identical for original
         and scaled); otherwise computed once before the model is freed.

    Caching logits on GPU avoids re-running the model forward on every L-BFGS
    closure iteration (the previous implementation did). Expected GPU memory
    for IWSLT14 (~140K non-pad valid tokens + ~130K test tokens × vocab ≈
    ~10K × 4 bytes each) is ~10–11 GB, well within A40 / A100.

    Returns:
        Path to the written JSONL file.
    """
    from jl.double_descent.transformer.transformer_config import TDDConfig
    from jl.double_descent.transformer.transformer_data import (
        build_vocab,
        collate_fn,
        load_split,
        TranslationDataset,
    )
    from jl.double_descent.transformer.transformer_model import TransformerModel
    from jl.double_descent.transformer.evaluation import discover_models
    from jl.double_descent.transformer.bleu import compute_bleu

    model_path = Path(model_dir)
    models = discover_models(model_dir)
    if not models:
        raise FileNotFoundError(f"No model_d*_*k.pt files in {model_dir}")

    if torch.cuda.device_count() == 0:
        raise RuntimeError("ts_evaluate_transformer requires a CUDA GPU")
    device = torch.device("cuda:0")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    config = TDDConfig()
    vocab = build_vocab(data_path)

    # Load valid + test splits once; the same loaders are reused for every model.
    valid_src, valid_tgt = load_split(data_path, "valid")
    test_src, test_tgt = load_split(data_path, "test")
    valid_dataset = TranslationDataset(valid_src, valid_tgt, vocab)
    test_dataset = TranslationDataset(test_src, test_tgt, vocab)
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=64,
        shuffle=False,
        collate_fn=partial(collate_fn, pad_idx=vocab.pad_idx),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
        collate_fn=partial(collate_fn, pad_idx=vocab.pad_idx),
    )

    eval_results = _load_evaluation_jsonl(model_path / "evaluation.jsonl")

    sorted_models = sorted(models.items())
    logger.info(
        f"Transformer TS evaluation: {len(sorted_models)} models on {device} "
        f"(single-process, fit on valid / eval on test)"
    )

    results: List[
        Tuple[int, int, float, float, float, float, float, float, float]
    ] = []
    for (d_model, train_samples), orig_model_path in sorted_models:
        model = TransformerModel(
            vocab_size=len(vocab),
            d_model=d_model,
            n_layers=config.n_layers,
            n_heads=config.n_heads,
            d_ff_multiplier=config.d_ff_multiplier,
            pad_idx=vocab.pad_idx,
        ).to(device)
        model.load_state_dict(
            torch.load(str(orig_model_path), map_location=device, weights_only=True)
        )
        model.eval()

        # Forward valid + test once; cache flat (non-pad) logits on GPU.
        valid_logits, valid_targets = _collect_transformer_flat_logits(
            model, valid_loader, vocab.pad_idx, device
        )
        test_logits, test_targets = _collect_transformer_flat_logits(
            model, test_loader, vocab.pad_idx, device
        )

        # BLEU: reuse from evaluation.jsonl when available (T doesn't change
        # greedy argmax). Otherwise compute once here while the model is live.
        eval_entry = eval_results.get(d_model)
        if eval_entry and "test_bleu" in eval_entry:
            orig_bleu = float(eval_entry["test_bleu"])
        else:
            orig_bleu = compute_bleu(model, test_dataset, vocab, device, max_len=128)
        ts_bleu = orig_bleu  # temperature scaling doesn't change argmax → BLEU unchanged

        del model
        torch.cuda.empty_cache()

        # Fit T on cached valid logits. Chunked closure so autograd
        # intermediates are per-chunk, not full-batch — a naive full-batch
        # F.cross_entropy on [N_valid, V] with backward uses ~5× the tensor
        # size for intermediates and OOMs on realistic IWSLT vocab sizes.
        temperature = nn.Parameter(torch.ones(1, device=device))
        # lr=1.0, max_iter=200 (see commit ca412cc): lr=0.01 bails out at
        # T~1.2 when the true optimum is far away.
        optimizer = torch.optim.LBFGS([temperature], lr=1.0, max_iter=200)

        n_valid = valid_logits.size(0)
        fit_chunk = 8192

        def _closure():
            optimizer.zero_grad()
            total_loss = torch.zeros((), device=device)
            for i in range(0, n_valid, fit_chunk):
                cl = valid_logits[i : i + fit_chunk]
                ct = valid_targets[i : i + fit_chunk]
                loss = (
                    F.cross_entropy(cl / temperature, ct, reduction="sum")
                    / n_valid
                )
                loss.backward()
                total_loss = total_loss + loss.detach()
            return total_loss

        optimizer.step(_closure)
        temp = temperature.item()

        # Free valid cache before computing test metrics — creates headroom
        # for the scaled test logits + softmax intermediates.
        del valid_logits, valid_targets
        torch.cuda.empty_cache()

        # Evaluate both original (T=1) and temperature-scaled metrics on cached
        # test logits. Both use the same flat-non-pad representation, so
        # original_loss here is directly comparable to ts_loss.
        with torch.no_grad():
            orig_loss, orig_error = _transformer_metrics_from_logits(
                test_logits, test_targets, temperature=1.0
            )
            ts_loss, ts_error = _transformer_metrics_from_logits(
                test_logits, test_targets, temperature=temp
            )

        results.append(
            (
                d_model,
                train_samples,
                orig_loss,
                ts_loss,
                orig_error,
                ts_error,
                orig_bleu,
                ts_bleu,
                temp,
            )
        )
        logger.info(
            f"  d_model={d_model}: T={temp:.4f}, "
            f"orig_loss={orig_loss:.4f}, ts_loss={ts_loss:.4f}, "
            f"orig_error={orig_error:.4f}, ts_error={ts_error:.4f}, "
            f"orig_bleu={orig_bleu:.2f}, ts_bleu={ts_bleu:.2f}"
        )

        del test_logits, test_targets, temperature
        torch.cuda.empty_cache()

    output_dir = model_path / "temperature_scaled"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "temperature_scaled_evaluation.jsonl"
    with open(output_path, "w") as f:
        for (
            d_model,
            train_samples,
            orig_loss,
            ts_loss,
            orig_error,
            ts_error,
            orig_bleu,
            ts_bleu,
            temp,
        ) in results:
            entry = {
                "d_model": d_model,
                "original_loss": round(orig_loss, 6),
                "ts_loss": round(ts_loss, 6),
                "original_error": round(orig_error, 6),
                "ts_error": round(ts_error, 6),
                "original_bleu": round(orig_bleu, 2),
                "ts_bleu": round(ts_bleu, 2),
                "temperature": round(temp, 6),
            }
            f.write(json.dumps(entry) + "\n")

    logger.info(f"Wrote {len(results)} entries to {output_path}")
    return output_path


# --- Main ---


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Evaluate calibration approaches on trained models"
    )
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--l2-calibrate",
        action="store_true",
        help="Evaluate L2-calibrated final layers vs originals",
    )
    mode_group.add_argument(
        "--temperature-scaling",
        action="store_true",
        help="Fit per-model temperature T on test NLL and evaluate",
    )
    parser.add_argument(
        "--resnet-path",
        type=str,
        default=None,
        help="Directory containing ResNet18 model_k*.pt files",
    )
    parser.add_argument(
        "--resnet-layer-dir",
        type=str,
        default=None,
        help="Directory containing ResNet18 layer_k*.pt L2-calibrated files (for --l2-calibrate mode)",
    )
    parser.add_argument(
        "--transformer-path",
        type=str,
        default=None,
        help="Directory containing Transformer model_d*_*k.pt files",
    )
    parser.add_argument(
        "--transformer-layer-dir",
        type=str,
        default=None,
        help="Directory containing Transformer layer_d*_*k.pt L2-calibrated files (for --l2-calibrate mode)",
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
    args = parser.parse_args()

    if args.resnet_path is None and args.transformer_path is None:
        parser.error("At least one of --resnet-path or --transformer-path is required")

    if args.l2_calibrate:
        if args.resnet_path and not args.resnet_layer_dir:
            parser.error("--resnet-layer-dir is required when using --l2-calibrate with --resnet-path")
        if args.transformer_path and not args.transformer_layer_dir:
            parser.error("--transformer-layer-dir is required when using --l2-calibrate with --transformer-path")

    mp.set_start_method("spawn", force=True)

    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("No GPUs available. Evaluation requires at least one GPU.")
    logger.info(f"Using {num_gpus} GPUs")

    if args.l2_calibrate:
        if args.resnet_path:
            logger.info("Evaluating ResNet18 L2-calibrated models...")
            path = evaluate_resnet(args.resnet_path, args.resnet_layer_dir, args.data_path)
            logger.info(f"ResNet results: {path}")

        if args.transformer_path:
            t_data_path = args.transformer_data_path
            if t_data_path is None:
                parser.error("--transformer-data-path is required when --transformer-path is set")
            logger.info("Evaluating Transformer L2-calibrated models...")
            path = evaluate_transformer(args.transformer_path, args.transformer_layer_dir, t_data_path)
            logger.info(f"Transformer results: {path}")

    elif args.temperature_scaling:
        if args.resnet_path:
            logger.info("Temperature-scaling ResNet18 models...")
            path = ts_evaluate_resnet(args.resnet_path, args.data_path)
            logger.info(f"ResNet TS results: {path}")

        if args.transformer_path:
            t_data_path = args.transformer_data_path
            if t_data_path is None:
                parser.error("--transformer-data-path is required when --transformer-path is set")
            logger.info("Temperature-scaling Transformer models...")
            path = ts_evaluate_transformer(args.transformer_path, t_data_path)
            logger.info(f"Transformer TS results: {path}")

    logger.info("Evaluation complete.")


if __name__ == "__main__":
    main()
