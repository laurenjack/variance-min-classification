#!/usr/bin/env python3
"""Oracle diagnostic: compare M2M100-12B / NLLB-200-3.3B / opus-mt-de-en on
the IWSLT'14 de-en test set. For each model: teacher-forced per-token NLL on
the full vocab (no compact-vocab subsetting), greedy translation + sacrebleu,
and a sample of qualitative outputs.

Per-character NLL is also computed so we can compare across tokenizers fairly:
character count is tokenizer-invariant.

Usage (one model per invocation):
    python -m jl.double_descent.transformer.oracle_diagnostic \\
        --model {m2m100|nllb|opus} \\
        --data-path ./data/iwslt14.m2m100.de-en \\
        --output-dir ./data/oracle_diagnostic
"""

import argparse
import json
import logging
import time
from pathlib import Path

import sacrebleu
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


MODEL_CONFIG = {
    "m2m100": {
        "name": "facebook/m2m100-12B-last-ckpt",
        "src_lang": "de",
        "tgt_lang": "en",
        "skip_first_target": True,   # leading __en__ language tag in target
        "default_batch": 16,
        "fp16": True,
    },
    "nllb": {
        "name": "facebook/nllb-200-3.3B",
        "src_lang": "deu_Latn",
        "tgt_lang": "eng_Latn",
        "skip_first_target": True,   # leading eng_Latn language tag in target
        "default_batch": 32,
        "fp16": True,
    },
    "opus": {
        "name": "Helsinki-NLP/opus-mt-de-en",
        "src_lang": None,
        "tgt_lang": None,
        "skip_first_target": False,  # bilingual model: no language tag prefix
        "default_batch": 64,
        "fp16": False,
    },
}


def load_model_and_tokenizer(model_choice: str, device: torch.device):
    """Returns (tokenizer, model, forced_bos_token_id_or_None)."""
    cfg = MODEL_CONFIG[model_choice]

    if model_choice == "m2m100":
        from transformers import AutoTokenizer, M2M100ForConditionalGeneration

        tokenizer = AutoTokenizer.from_pretrained(cfg["name"])
        tokenizer.src_lang = cfg["src_lang"]
        tokenizer.tgt_lang = cfg["tgt_lang"]
        model = M2M100ForConditionalGeneration.from_pretrained(
            cfg["name"], torch_dtype=torch.float16 if cfg["fp16"] else torch.float32,
        ).to(device)
        forced_bos = tokenizer.get_lang_id(cfg["tgt_lang"])

    elif model_choice == "nllb":
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

        tokenizer = AutoTokenizer.from_pretrained(
            cfg["name"], src_lang=cfg["src_lang"], tgt_lang=cfg["tgt_lang"],
        )
        model = AutoModelForSeq2SeqLM.from_pretrained(
            cfg["name"], torch_dtype=torch.float16 if cfg["fp16"] else torch.float32,
        ).to(device)
        forced_bos = tokenizer.convert_tokens_to_ids(cfg["tgt_lang"])

    elif model_choice == "opus":
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

        tokenizer = AutoTokenizer.from_pretrained(cfg["name"])
        model = AutoModelForSeq2SeqLM.from_pretrained(
            cfg["name"], torch_dtype=torch.float16 if cfg["fp16"] else torch.float32,
        ).to(device)
        forced_bos = None

    else:
        raise ValueError(f"Unknown model: {model_choice}")

    model.eval()
    return tokenizer, model, forced_bos


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=["m2m100", "nllb", "opus"])
    parser.add_argument(
        "--data-path", required=True,
        help="Directory with test.de.txt and test.en.txt",
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="Output directory; per-model subdir will be created",
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-gen-len", type=int, default=256)
    parser.add_argument(
        "--max-sentences", type=int, default=None,
        help="Optional truncation for fast debug",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = MODEL_CONFIG[args.model]
    batch_size = args.batch_size or cfg["default_batch"]
    skip_first_target = cfg["skip_first_target"]

    output_dir = Path(args.output_dir) / args.model
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output dir: {output_dir}")

    logger.info(f"Loading model: {cfg['name']} (fp16={cfg['fp16']})")
    t_load = time.time()
    tokenizer, model, forced_bos = load_model_and_tokenizer(args.model, device)
    logger.info(f"Model loaded in {time.time() - t_load:.1f}s")

    # Test set
    data_path = Path(args.data_path)
    with open(data_path / "test.de.txt", encoding="utf-8") as f:
        src_texts = [line.strip() for line in f]
    with open(data_path / "test.en.txt", encoding="utf-8") as f:
        tgt_texts = [line.strip() for line in f]
    assert len(src_texts) == len(tgt_texts)
    if args.max_sentences:
        src_texts = src_texts[: args.max_sentences]
        tgt_texts = tgt_texts[: args.max_sentences]
    logger.info(f"Test set: {len(src_texts)} sentence pairs, batch_size={batch_size}")

    pad_id = tokenizer.pad_token_id
    n_batches = (len(src_texts) + batch_size - 1) // batch_size

    per_sentence_records = []
    total_nll = 0.0       # sum of per-token NLL on content positions
    total_tokens = 0
    total_chars = 0

    t0 = time.time()
    with torch.no_grad():
        for batch_idx in range(n_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, len(src_texts))
            batch_de = src_texts[start:end]
            batch_en = tgt_texts[start:end]

            # Source
            src_enc = tokenizer(
                batch_de, return_tensors="pt", padding=True, truncation=True, max_length=512,
            ).to(device)

            # Target (for teacher-forced loss)
            tgt_enc = tokenizer(
                text_target=batch_en, return_tensors="pt", padding=True,
                truncation=True, max_length=512,
            ).to(device)
            tgt_ids = tgt_enc["input_ids"]
            tgt_mask = tgt_enc["attention_mask"]

            # Teacher-forced forward (model.shift_right + cross-entropy internally)
            outputs = model(
                input_ids=src_enc["input_ids"],
                attention_mask=src_enc["attention_mask"],
                labels=tgt_ids,
            )
            logits = outputs.logits.float()
            log_probs = F.log_softmax(logits, dim=-1)
            per_token_nll = -log_probs.gather(-1, tgt_ids.unsqueeze(-1)).squeeze(-1)  # [B, T]

            mask = (tgt_ids != pad_id) & tgt_mask.bool()
            if skip_first_target:
                content_mask = mask.clone()
                content_mask[:, 0] = False
            else:
                content_mask = mask

            # Greedy generation
            gen_kwargs = dict(
                input_ids=src_enc["input_ids"],
                attention_mask=src_enc["attention_mask"],
                max_length=args.max_gen_len,
                num_beams=1,
                do_sample=False,
            )
            if forced_bos is not None:
                gen_kwargs["forced_bos_token_id"] = forced_bos
            generated = model.generate(**gen_kwargs)
            hypotheses = tokenizer.batch_decode(generated, skip_special_tokens=True)

            for i in range(len(batch_de)):
                m_i = content_mask[i]
                sent_nll = float(per_token_nll[i][m_i].sum().item())
                n_tok = int(m_i.sum().item())
                ref = batch_en[i]
                n_char = len(ref)

                per_sentence_records.append({
                    "source": batch_de[i],
                    "reference": ref,
                    "hypothesis": hypotheses[i],
                    "nll_total": sent_nll,
                    "n_content_tokens": n_tok,
                    "n_chars": n_char,
                })
                total_nll += sent_nll
                total_tokens += n_tok
                total_chars += n_char

            if (batch_idx + 1) % 10 == 0 or batch_idx == n_batches - 1:
                elapsed = time.time() - t0
                rate = (batch_idx + 1) / elapsed
                eta = (n_batches - batch_idx - 1) / rate if rate > 0 else float("inf")
                logger.info(
                    f"Batch {batch_idx + 1}/{n_batches}: "
                    f"sentences={len(per_sentence_records)}, "
                    f"elapsed={elapsed:.1f}s, eta={eta:.1f}s"
                )

    total_time = time.time() - t0
    nll_per_token = total_nll / max(total_tokens, 1)
    nll_per_char = total_nll / max(total_chars, 1)
    nll_per_sentence = total_nll / len(per_sentence_records)

    hypotheses = [r["hypothesis"] for r in per_sentence_records]
    references = [r["reference"] for r in per_sentence_records]
    bleu = sacrebleu.corpus_bleu(hypotheses, [references])

    summary = {
        "model": args.model,
        "model_name": cfg["name"],
        "n_sentences": len(per_sentence_records),
        "n_content_tokens": total_tokens,
        "n_chars": total_chars,
        "nll_per_token": nll_per_token,
        "nll_per_char": nll_per_char,
        "nll_per_sentence": nll_per_sentence,
        "bleu_score": bleu.score,
        "bleu_signature": str(bleu),
        "wall_time_sec": total_time,
        "batch_size": batch_size,
        "samples": per_sentence_records[:10],
    }
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    per_sentence_path = output_dir / "per_sentence.jsonl"
    with open(per_sentence_path, "w", encoding="utf-8") as f:
        for r in per_sentence_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    logger.info(f"=== {args.model.upper()} RESULTS ===")
    logger.info(f"  NLL/token (content positions): {nll_per_token:.4f}")
    logger.info(f"  NLL/char:                      {nll_per_char:.4f}")
    logger.info(f"  NLL/sentence:                  {nll_per_sentence:.4f}")
    logger.info(f"  BLEU:                          {bleu.score:.2f}")
    logger.info(f"  Wall time:                     {total_time:.1f}s")
    logger.info(f"Wrote summary: {summary_path}")
    logger.info(f"Wrote per-sentence: {per_sentence_path}")


if __name__ == "__main__":
    main()
