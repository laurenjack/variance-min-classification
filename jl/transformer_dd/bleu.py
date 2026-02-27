"""BLEU score computation for translation evaluation."""

from typing import List

import sacrebleu
import torch
from torch.utils.data import DataLoader

from jl.transformer_dd.transformer_data import TranslationDataset, Vocab, collate_fn
from jl.transformer_dd.transformer_model import TransformerModel


def compute_bleu(
    model: TransformerModel,
    dataset: TranslationDataset,
    vocab: Vocab,
    device: torch.device,
    max_len: int = 128,
    batch_size: int = 32,
) -> float:
    """Compute corpus-level BLEU score using greedy decoding.

    Args:
        model: Trained Transformer model.
        dataset: Translation dataset.
        vocab: Shared vocabulary.
        device: Device to run inference on.
        max_len: Maximum generation length.
        batch_size: Batch size for inference.

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

    with torch.no_grad():
        for src, tgt in loader:
            src = src.to(device)

            # Generate translations
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


def remove_bpe(text: str) -> str:
    """Remove BPE markers from text.

    Args:
        text: BPE-tokenized text with @@ markers.

    Returns:
        Detokenized text.
    """
    return text.replace("@@ ", "").replace(" @@", "")
