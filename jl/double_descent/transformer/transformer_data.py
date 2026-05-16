"""Data loading for IWSLT'14 German-English translation."""

import json
import random
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from torch.utils.data import Dataset, Sampler


class Vocab:
    """Vocabulary for BPE-tokenized text."""

    PAD_TOKEN = "<pad>"
    UNK_TOKEN = "<unk>"
    BOS_TOKEN = "<s>"
    EOS_TOKEN = "</s>"

    def __init__(self, tokens: List[str]):
        """Initialize vocabulary from list of tokens.

        Args:
            tokens: List of unique tokens (excluding special tokens).
        """
        self.special_tokens = [self.PAD_TOKEN, self.UNK_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN]
        self.tokens = self.special_tokens + tokens

        self.token_to_idx = {t: i for i, t in enumerate(self.tokens)}
        self.idx_to_token = {i: t for i, t in enumerate(self.tokens)}

        self.pad_idx = self.token_to_idx[self.PAD_TOKEN]
        self.unk_idx = self.token_to_idx[self.UNK_TOKEN]
        self.bos_idx = self.token_to_idx[self.BOS_TOKEN]
        self.eos_idx = self.token_to_idx[self.EOS_TOKEN]

    def __len__(self) -> int:
        return len(self.tokens)

    def encode(self, text: str) -> List[int]:
        """Encode text to token indices (without BOS/EOS)."""
        return [self.token_to_idx.get(t, self.unk_idx) for t in text.split()]

    def encode_with_special(self, text: str) -> List[int]:
        """Encode text with BOS and EOS tokens."""
        return [self.bos_idx] + self.encode(text) + [self.eos_idx]

    def decode(self, indices: List[int], remove_special: bool = True) -> str:
        """Decode token indices to text."""
        tokens = []
        for idx in indices:
            if idx in self.idx_to_token:
                token = self.idx_to_token[idx]
                if remove_special and token in self.special_tokens:
                    continue
                tokens.append(token)
        return " ".join(tokens)


class TranslationDataset(Dataset):
    """Dataset for translation pairs."""

    def __init__(
        self,
        src_sentences: List[str],
        tgt_sentences: List[str],
        vocab: Vocab,
    ):
        """Initialize dataset.

        Args:
            src_sentences: Source language sentences (BPE tokenized).
            tgt_sentences: Target language sentences (BPE tokenized).
            vocab: Shared vocabulary.
        """
        assert len(src_sentences) == len(tgt_sentences)
        self.src_sentences = src_sentences
        self.tgt_sentences = tgt_sentences
        self.vocab = vocab

        # Pre-encode all sentences
        self.src_encoded = [vocab.encode_with_special(s) for s in src_sentences]
        self.tgt_encoded = [vocab.encode_with_special(s) for s in tgt_sentences]

    def __len__(self) -> int:
        return len(self.src_sentences)

    def __getitem__(self, idx: int) -> Tuple[List[int], List[int]]:
        return self.src_encoded[idx], self.tgt_encoded[idx]

    def get_lengths(self) -> List[int]:
        """Get total length (src + tgt) for each example."""
        return [len(s) + len(t) for s, t in zip(self.src_encoded, self.tgt_encoded)]


class MaxTokensBatchSampler(Sampler):
    """Batch sampler that groups sentences to fit max_tokens per batch.

    - Sorts sentences by length
    - Greedily fills batches up to max_tokens
    - Shuffles batch order (not within-batch order)
    """

    def __init__(
        self,
        dataset: TranslationDataset,
        max_tokens: int,
        shuffle: bool = True,
        seed: int = 42,
    ):
        """Initialize sampler.

        Args:
            dataset: Translation dataset.
            max_tokens: Maximum tokens per batch.
            shuffle: Whether to shuffle batch order.
            seed: Random seed for shuffling.
        """
        self.dataset = dataset
        self.max_tokens = max_tokens
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

        # Get lengths and sort indices by length
        lengths = dataset.get_lengths()
        self.sorted_indices = sorted(range(len(lengths)), key=lambda i: lengths[i])

        # Create batches
        self.batches = self._create_batches(lengths)

    def _create_batches(self, lengths: List[int]) -> List[List[int]]:
        """Create batches that fit within max_tokens."""
        batches = []
        current_batch = []
        current_max_len = 0

        for idx in self.sorted_indices:
            length = lengths[idx]

            # Check if adding this example would exceed max_tokens
            new_max_len = max(current_max_len, length)
            new_batch_tokens = new_max_len * (len(current_batch) + 1)

            if new_batch_tokens > self.max_tokens and current_batch:
                # Start new batch
                batches.append(current_batch)
                current_batch = [idx]
                current_max_len = length
            else:
                current_batch.append(idx)
                current_max_len = new_max_len

        # Don't forget the last batch
        if current_batch:
            batches.append(current_batch)

        return batches

    def set_epoch(self, epoch: int):
        """Set epoch for deterministic shuffling."""
        self.epoch = epoch

    def __iter__(self):
        batches = self.batches.copy()

        if self.shuffle:
            rng = random.Random(self.seed + self.epoch)
            rng.shuffle(batches)

        for batch in batches:
            yield batch

    def __len__(self) -> int:
        return len(self.batches)


def collate_fn(
    batch: List[Tuple[List[int], List[int]]],
    pad_idx: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Collate batch of translation pairs with padding.

    Args:
        batch: List of (src_indices, tgt_indices) tuples.
        pad_idx: Padding token index.

    Returns:
        src_tensor: [batch_size, max_src_len]
        tgt_tensor: [batch_size, max_tgt_len]
    """
    src_seqs, tgt_seqs = zip(*batch)

    # Find max lengths
    max_src_len = max(len(s) for s in src_seqs)
    max_tgt_len = max(len(t) for t in tgt_seqs)

    # Pad sequences
    src_padded = []
    tgt_padded = []
    for src, tgt in batch:
        src_padded.append(src + [pad_idx] * (max_src_len - len(src)))
        tgt_padded.append(tgt + [pad_idx] * (max_tgt_len - len(tgt)))

    return torch.tensor(src_padded), torch.tensor(tgt_padded)


class GPUBatchedLoader:
    """Pre-materializes all training batches as padded GPU tensors.

    Mirrors the batch-construction logic of MaxTokensBatchSampler (length-
    bucketed, greedy fill to max_tokens) but the resulting padded src/tgt
    pairs live on `device` from construction onward. Iteration is just an
    index walk — no CPU-side collation per step, no .to(device).

    IWSLT-14 train tokens are tiny (~30-40 MB on GPU even with padding), so
    keeping the whole epoch resident is essentially free. Use this for the
    transformer trainer's train/valid/test loaders.

    Args:
        dataset: Must implement __len__, __getitem__ -> (src_ids, tgt_ids),
            and get_lengths() -> list[int]. Both TranslationDataset and
            M2M100TranslationDataset satisfy this.
        pad_idx: Padding token index.
        max_tokens: Max tokens per batch (Vaswani max-tokens batching).
        device: torch.device the batches will live on.
        shuffle: If True, shuffles batch order each epoch (call set_epoch).
        seed: Base RNG seed for shuffle (combined with epoch index).
    """

    def __init__(
        self,
        dataset,
        pad_idx: int,
        max_tokens: int,
        device: torch.device,
        shuffle: bool = True,
        seed: int = 42,
    ):
        self.pad_idx = pad_idx
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self.device = device

        lengths = dataset.get_lengths()
        sorted_indices = sorted(range(len(lengths)), key=lambda i: lengths[i])

        # Greedy batch construction matching MaxTokensBatchSampler.
        batches_idx: List[List[int]] = []
        current_batch: List[int] = []
        current_max_len = 0
        for idx in sorted_indices:
            length = lengths[idx]
            new_max_len = max(current_max_len, length)
            new_batch_tokens = new_max_len * (len(current_batch) + 1)
            if new_batch_tokens > max_tokens and current_batch:
                batches_idx.append(current_batch)
                current_batch = [idx]
                current_max_len = length
            else:
                current_batch.append(idx)
                current_max_len = new_max_len
        if current_batch:
            batches_idx.append(current_batch)

        # Pre-pad and move every batch to device.
        self.batches: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for idx_batch in batches_idx:
            src_seqs = []
            tgt_seqs = []
            for i in idx_batch:
                s, t = dataset[i]
                src_seqs.append(s)
                tgt_seqs.append(t)
            max_src = max(len(s) for s in src_seqs)
            max_tgt = max(len(t) for t in tgt_seqs)
            src_padded = torch.tensor(
                [s + [pad_idx] * (max_src - len(s)) for s in src_seqs],
                dtype=torch.long, device=device,
            )
            tgt_padded = torch.tensor(
                [t + [pad_idx] * (max_tgt - len(t)) for t in tgt_seqs],
                dtype=torch.long, device=device,
            )
            self.batches.append((src_padded, tgt_padded))

    def __len__(self) -> int:
        return len(self.batches)

    def set_epoch(self, epoch: int):
        """Set epoch for deterministic shuffle."""
        self.epoch = epoch

    def __iter__(self):
        if self.shuffle:
            rng = random.Random(self.seed + self.epoch)
            order = list(range(len(self.batches)))
            rng.shuffle(order)
        else:
            order = range(len(self.batches))
        for i in order:
            yield self.batches[i]


def build_vocab(data_dir: str) -> Vocab:
    """Build vocabulary from training data.

    Args:
        data_dir: Directory containing train.de and train.en.

    Returns:
        Shared vocabulary.
    """
    data_path = Path(data_dir)
    tokens = set()

    for lang in ["de", "en"]:
        file_path = data_path / f"train.{lang}"
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                tokens.update(line.strip().split())

    return Vocab(sorted(tokens))


def load_split(data_dir: str, split: str) -> Tuple[List[str], List[str]]:
    """Load source and target sentences for a split.

    Args:
        data_dir: Directory containing data files.
        split: One of "train", "valid", "test".

    Returns:
        (src_sentences, tgt_sentences)
    """
    data_path = Path(data_dir)

    src_path = data_path / f"{split}.de"
    tgt_path = data_path / f"{split}.en"

    with open(src_path, "r", encoding="utf-8") as f:
        src_sentences = [line.strip() for line in f]

    with open(tgt_path, "r", encoding="utf-8") as f:
        tgt_sentences = [line.strip() for line in f]

    return src_sentences, tgt_sentences


def load_iwslt14(
    data_dir: str,
    train_samples: int = 4000,
    subsample_seed: int = 42,
) -> Tuple[TranslationDataset, TranslationDataset, TranslationDataset, Vocab]:
    """Load IWSLT'14 de-en with subsampling.

    Args:
        data_dir: Directory containing preprocessed IWSLT data.
        train_samples: Number of training samples to use.
        subsample_seed: Random seed for subsampling.

    Returns:
        train_dataset: Subsampled training dataset.
        valid_dataset: Full validation dataset.
        test_dataset: Full test dataset.
        vocab: Shared vocabulary.
    """
    # Build vocabulary from full training data
    vocab = build_vocab(data_dir)

    # Load all splits
    train_src, train_tgt = load_split(data_dir, "train")
    valid_src, valid_tgt = load_split(data_dir, "valid")
    test_src, test_tgt = load_split(data_dir, "test")

    # Subsample training data
    if train_samples < len(train_src):
        rng = random.Random(subsample_seed)
        indices = list(range(len(train_src)))
        rng.shuffle(indices)
        indices = indices[:train_samples]
        train_src = [train_src[i] for i in indices]
        train_tgt = [train_tgt[i] for i in indices]

    # Create datasets
    train_dataset = TranslationDataset(train_src, train_tgt, vocab)
    valid_dataset = TranslationDataset(valid_src, valid_tgt, vocab)
    test_dataset = TranslationDataset(test_src, test_tgt, vocab)

    return train_dataset, valid_dataset, test_dataset, vocab


# --- M2M100 tokenizer support ---


class M2M100Vocab:
    """Vocabulary based on M2M100 tokenizer with compact ID remapping.

    Loads a pre-computed vocab_mapping.json that maps between compact IDs
    (0..~18K) and M2M100's native token IDs (sparse over 0..128K).
    Duck-types with Vocab (same interface: pad_idx, __len__, decode).
    """

    def __init__(self, mapping_path: str):
        with open(mapping_path, "r", encoding="utf-8") as f:
            mapping = json.load(f)

        self.compact_to_m2m100: List[int] = mapping["compact_to_m2m100"]
        self.m2m100_to_compact = {
            m2m100_id: compact_id
            for compact_id, m2m100_id in enumerate(self.compact_to_m2m100)
        }

        self.pad_idx: int = mapping["pad_idx"]
        self.bos_idx: int = mapping["bos_idx"]
        self.eos_idx: int = mapping["eos_idx"]
        self.unk_idx: int = mapping["unk_idx"]
        self.de_lang_idx: int = mapping["de_lang_idx"]
        self.en_lang_idx: int = mapping["en_lang_idx"]
        self._vocab_size: int = mapping["vocab_size"]
        self._m2m100_model: str = mapping.get("m2m100_model", "facebook/m2m100-12B-last-ckpt")
        self._tokenizer = None

    def __len__(self) -> int:
        return self._vocab_size

    def _get_tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(self._m2m100_model)
        return self._tokenizer

    def decode(self, indices: List[int], remove_special: bool = True) -> str:
        """Decode compact token IDs to text via M2M100 tokenizer."""
        special = {self.pad_idx, self.bos_idx, self.eos_idx}
        if remove_special:
            indices = [i for i in indices if i not in special]
        m2m100_ids = [self.compact_to_m2m100[i] for i in indices]
        return self._get_tokenizer().decode(m2m100_ids, skip_special_tokens=False)


class M2M100TranslationDataset(Dataset):
    """Dataset for translation pairs using pre-tokenized compact IDs."""

    def __init__(
        self,
        src_id_seqs: List[List[int]],
        tgt_id_seqs: List[List[int]],
        vocab: M2M100Vocab,
    ):
        assert len(src_id_seqs) == len(tgt_id_seqs)
        self.vocab = vocab

        # Wrap with BOS/EOS (same pattern as TranslationDataset.encode_with_special)
        self.src_encoded = [[vocab.bos_idx] + ids + [vocab.eos_idx] for ids in src_id_seqs]
        self.tgt_encoded = [[vocab.bos_idx] + ids + [vocab.eos_idx] for ids in tgt_id_seqs]

    def __len__(self) -> int:
        return len(self.src_encoded)

    def __getitem__(self, idx: int) -> Tuple[List[int], List[int]]:
        return self.src_encoded[idx], self.tgt_encoded[idx]

    def get_lengths(self) -> List[int]:
        return [len(s) + len(t) for s, t in zip(self.src_encoded, self.tgt_encoded)]


def load_m2m100_split_ids(data_dir: str, split: str) -> Tuple[List[List[int]], List[List[int]]]:
    """Load pre-tokenized compact ID sequences for a split.

    Args:
        data_dir: Directory containing .ids files.
        split: One of "train", "valid", "test".

    Returns:
        (src_id_seqs, tgt_id_seqs) as lists of int lists.
    """
    data_path = Path(data_dir)
    src_ids = []
    tgt_ids = []

    with open(data_path / f"{split}.de.ids", "r", encoding="utf-8") as f:
        for line in f:
            src_ids.append([int(x) for x in line.strip().split()])

    with open(data_path / f"{split}.en.ids", "r", encoding="utf-8") as f:
        for line in f:
            tgt_ids.append([int(x) for x in line.strip().split()])

    return src_ids, tgt_ids


def load_m2m100_iwslt14(
    data_dir: str,
    train_samples: int = 4000,
    subsample_seed: int = 42,
) -> Tuple[M2M100TranslationDataset, M2M100TranslationDataset, M2M100TranslationDataset, M2M100Vocab]:
    """Load M2M100-tokenized IWSLT'14 with subsampling.

    Mirrors `load_iwslt14`'s signature/semantics (shuffle full train set with
    `subsample_seed`, take first `train_samples`). Use this for normal/main
    runs; use `load_m2m100_iwslt14_variance_split` for the disjoint-splits
    variance experiment.
    """
    vocab = M2M100Vocab(str(Path(data_dir) / "vocab_mapping.json"))

    train_src, train_tgt = load_m2m100_split_ids(data_dir, "train")
    valid_src, valid_tgt = load_m2m100_split_ids(data_dir, "valid")
    test_src, test_tgt = load_m2m100_split_ids(data_dir, "test")

    if train_samples < len(train_src):
        rng = random.Random(subsample_seed)
        indices = list(range(len(train_src)))
        rng.shuffle(indices)
        indices = indices[:train_samples]
        train_src = [train_src[i] for i in indices]
        train_tgt = [train_tgt[i] for i in indices]

    train_dataset = M2M100TranslationDataset(train_src, train_tgt, vocab)
    valid_dataset = M2M100TranslationDataset(valid_src, valid_tgt, vocab)
    test_dataset = M2M100TranslationDataset(test_src, test_tgt, vocab)
    return train_dataset, valid_dataset, test_dataset, vocab


def load_m2m100_iwslt14_train_chunk_test(
    data_dir: str,
    train_samples: int = 36000,
    holdout_test_samples: int = 6750,
    subsample_seed: int = 42,
) -> Tuple[M2M100TranslationDataset, M2M100TranslationDataset, M2M100TranslationDataset, M2M100Vocab]:
    """Load M2M100-tokenized IWSLT'14 with an in-distribution test chunk
    carved from train.

    Shuffles the full IWSLT train set with `subsample_seed`, takes:
      - train  = shuffled indices [0 : train_samples]
      - test   = shuffled indices [train_samples : train_samples + holdout_test_samples]
    Both come from the same IWSLT-train distribution (no domain shift).
    Valid is the standard IWSLT valid split (kept for early-stop signal).

    Default holdout_test_samples=6750 matches the IWSLT-test split size so
    estimator variance is comparable.
    """
    vocab = M2M100Vocab(str(Path(data_dir) / "vocab_mapping.json"))

    train_src, train_tgt = load_m2m100_split_ids(data_dir, "train")
    valid_src, valid_tgt = load_m2m100_split_ids(data_dir, "valid")

    n_total = len(train_src)
    required = train_samples + holdout_test_samples
    if n_total < required:
        raise ValueError(
            f"IWSLT train has {n_total} sentences; need "
            f"{train_samples} train + {holdout_test_samples} held-out test "
            f"= {required}."
        )

    rng = random.Random(subsample_seed)
    indices = list(range(n_total))
    rng.shuffle(indices)
    train_idx = indices[:train_samples]
    test_idx = indices[train_samples : train_samples + holdout_test_samples]

    train_dataset = M2M100TranslationDataset(
        [train_src[i] for i in train_idx],
        [train_tgt[i] for i in train_idx],
        vocab,
    )
    valid_dataset = M2M100TranslationDataset(valid_src, valid_tgt, vocab)
    test_dataset = M2M100TranslationDataset(
        [train_src[i] for i in test_idx],
        [train_tgt[i] for i in test_idx],
        vocab,
    )
    return train_dataset, valid_dataset, test_dataset, vocab


def _compute_variance_chunks(
    n_total: int,
    num_splits: int,
    samples_per_split: int,
    subsample_seed: int,
    holdout_samples: Optional[int] = None,
) -> List[List[int]]:
    """Shuffle range(n_total) and return num_splits + 1 disjoint chunks.

    Chunks 0..num_splits-1 each contain `samples_per_split` indices — the
    training splits. Chunk `num_splits` is the held-out in-distribution
    test chunk:
      - If `holdout_samples` is given, the held-out chunk has that many
        indices (must fit in n_total alongside the training splits).
      - If None, the held-out chunk is everything left over after the
        training splits, i.e. `n_total - num_splits * samples_per_split`
        indices. Use this when you trained with the legacy (no-held-out)
        loader and want every never-seen index as the test set.
    """
    train_required = num_splits * samples_per_split
    if n_total < train_required:
        raise ValueError(
            f"Not enough data for {num_splits} training splits of "
            f"{samples_per_split} samples each. Need {train_required}, "
            f"have {n_total}."
        )
    if holdout_samples is None:
        held_count = n_total - train_required
        if held_count <= 0:
            raise ValueError(
                f"holdout_samples=None requires n_total > num_splits * "
                f"samples_per_split. Got n_total={n_total}, "
                f"train_required={train_required}."
            )
    else:
        held_count = holdout_samples
        if n_total < train_required + held_count:
            raise ValueError(
                f"Not enough data for {num_splits} training splits + "
                f"{held_count}-sample held-out test chunk. Need "
                f"{train_required + held_count}, have {n_total}."
            )

    rng = random.Random(subsample_seed)
    indices = list(range(n_total))
    rng.shuffle(indices)
    chunks = [
        indices[i * samples_per_split:(i + 1) * samples_per_split]
        for i in range(num_splits)
    ]
    chunks.append(indices[train_required:train_required + held_count])
    return chunks


def load_iwslt14_variance_split(
    data_dir: str,
    split_id: int,
    num_splits: int = 4,
    samples_per_split: int = 32000,
    subsample_seed: int = 42,
    holdout_samples: Optional[int] = None,
) -> Tuple[TranslationDataset, TranslationDataset, TranslationDataset, Vocab]:
    """Load BPE-tokenized IWSLT'14 with a specific disjoint training split
    and a held-out chunk of the same training distribution as the test set.

    Pipeline:
      1. Shuffle the full IWSLT train (~160K) with subsample_seed.
      2. Partition into num_splits + 1 disjoint chunks.
      3. Chunks 0..num_splits-1 are the training splits (one per model)
         of `samples_per_split` sentences each. Chunk num_splits is the
         held-out in-distribution test set:
           - holdout_samples=None → all leftover after the training
             splits (n_total - num_splits * samples_per_split).
           - otherwise           → exactly holdout_samples sentences.
      4. The IWSLT `valid` split is still used as the val set for ES.

    Returns:
        train_dataset: this split's training chunk.
        valid_dataset: full IWSLT valid (shared across all models).
        test_dataset: the held-out in-distribution chunk (shared).
        vocab: shared vocabulary.
    """
    if split_id < 0 or split_id >= num_splits:
        raise ValueError(f"split_id must be in [0, {num_splits-1}], got {split_id}")

    vocab = build_vocab(data_dir)
    train_src_all, train_tgt_all = load_split(data_dir, "train")
    valid_src, valid_tgt = load_split(data_dir, "valid")

    chunks = _compute_variance_chunks(
        n_total=len(train_src_all),
        num_splits=num_splits,
        samples_per_split=samples_per_split,
        subsample_seed=subsample_seed,
        holdout_samples=holdout_samples,
    )
    train_indices = chunks[split_id]
    test_indices = chunks[num_splits]  # held-out chunk

    train_src = [train_src_all[i] for i in train_indices]
    train_tgt = [train_tgt_all[i] for i in train_indices]
    test_src = [train_src_all[i] for i in test_indices]
    test_tgt = [train_tgt_all[i] for i in test_indices]

    train_dataset = TranslationDataset(train_src, train_tgt, vocab)
    valid_dataset = TranslationDataset(valid_src, valid_tgt, vocab)
    test_dataset = TranslationDataset(test_src, test_tgt, vocab)

    return train_dataset, valid_dataset, test_dataset, vocab


def load_m2m100_iwslt14_variance_split(
    data_dir: str,
    split_id: int,
    num_splits: int = 4,
    samples_per_split: int = 32000,
    subsample_seed: int = 42,
    holdout_samples: Optional[int] = None,
) -> Tuple[M2M100TranslationDataset, M2M100TranslationDataset, M2M100TranslationDataset, M2M100Vocab]:
    """M2M100-tokenized counterpart to load_iwslt14_variance_split.

    Same held-out-train-chunk-as-test design — see load_iwslt14_variance_split
    docstring for details.
    """
    if split_id < 0 or split_id >= num_splits:
        raise ValueError(f"split_id must be in [0, {num_splits-1}], got {split_id}")

    vocab = M2M100Vocab(str(Path(data_dir) / "vocab_mapping.json"))

    train_src_all, train_tgt_all = load_m2m100_split_ids(data_dir, "train")
    valid_src, valid_tgt = load_m2m100_split_ids(data_dir, "valid")

    chunks = _compute_variance_chunks(
        n_total=len(train_src_all),
        num_splits=num_splits,
        samples_per_split=samples_per_split,
        subsample_seed=subsample_seed,
        holdout_samples=holdout_samples,
    )
    train_indices = chunks[split_id]
    test_indices = chunks[num_splits]  # held-out chunk

    train_src = [train_src_all[i] for i in train_indices]
    train_tgt = [train_tgt_all[i] for i in train_indices]
    test_src = [train_src_all[i] for i in test_indices]
    test_tgt = [train_tgt_all[i] for i in test_indices]

    train_dataset = M2M100TranslationDataset(train_src, train_tgt, vocab)
    valid_dataset = M2M100TranslationDataset(valid_src, valid_tgt, vocab)
    test_dataset = M2M100TranslationDataset(test_src, test_tgt, vocab)

    return train_dataset, valid_dataset, test_dataset, vocab
