"""BLEU score computation for translation evaluation."""

from typing import List

import sacrebleu
import torch
from torch.utils.data import DataLoader

from jl.double_descent.transformer.transformer_data import TranslationDataset, Vocab, collate_fn
from jl.double_descent.transformer.transformer_model import TransformerModel


def compute_bleu(
    model: TransformerModel,
    dataset: TranslationDataset,
    vocab: Vocab,
    device: torch.device,
    max_len: int = 128,
    batch_size: int = 256,
    use_bf16: bool = True,
) -> float:
    """Compute corpus-level BLEU score using greedy decoding.

    Args:
        model: Trained Transformer model.
        dataset: Translation dataset.
        vocab: Shared vocabulary.
        device: Device to run inference on.
        max_len: Maximum generation length.
        batch_size: Batch size for inference (default 256 — fine on modern
            GPUs; drop if memory-bound).
        use_bf16: Wrap `generate()` in a BF16 autocast region. ~1.5–2x
            faster on Hopper/Blackwell tensor cores; numerically safe for
            greedy decode since the final argmax is dtype-stable.

    Returns:
        BLEU score (0-100 scale).
    """
    model.eval()

    # Create dataloader (no shuffling for consistent results)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, vocab.pad_idx),
    )

    hypotheses: List[str] = []
    references: List[str] = []

    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if use_bf16 and device.type == "cuda"
        else _NullCtx()
    )

    with torch.no_grad():
        for src, tgt in loader:
            src = src.to(device)

            # Generate translations under BF16 autocast (greedy decode is
            # dtype-stable; argmax of fp32-promoted logits is identical to
            # argmax in bf16 in the vast majority of cases).
            with autocast_ctx:
                generated = model.generate(
                    src,
                    max_len=max_len,
                    bos_idx=vocab.bos_idx,
                    eos_idx=vocab.eos_idx,
                )

            # Decode generated sequences
            for i in range(generated.shape[0]):
                gen_tokens = generated[i].tolist()
                hyp = vocab.decode(gen_tokens, remove_special=True)
                hypotheses.append(hyp)

                # Get reference (remove BOS/EOS)
                ref_tokens = tgt[i].tolist()
                ref = vocab.decode(ref_tokens, remove_special=True)
                references.append(ref)

    # Compute BLEU
    # sacrebleu expects list of hypotheses and list of list of references
    bleu = sacrebleu.corpus_bleu(hypotheses, [references])

    return bleu.score


class _NullCtx:
    """No-op context manager for the use_bf16=False branch."""
    def __enter__(self):
        return self
    def __exit__(self, *exc):
        return False


def remove_bpe(text: str) -> str:
    """Remove BPE markers from text.

    Args:
        text: BPE-tokenized text with @@ markers.

    Returns:
        Detokenized text.
    """
    return text.replace("@@ ", "").replace(" @@", "")
