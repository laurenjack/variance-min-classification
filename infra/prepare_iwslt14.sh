#!/bin/bash
# Prepare IWSLT'14 German-English data for Transformer training.
# Downloads from HuggingFace datasets and applies BPE tokenization.
#
# This script:
# 1. Downloads IWSLT'14 de-en data from HuggingFace
# 2. Learns joint BPE vocabulary (10K merge operations)
# 3. Applies BPE to train/valid/test splits
# 4. Outputs to data/iwslt14.tokenized.de-en/
#
# Usage:
#   ./infra/prepare_iwslt14.sh
#
# Output: data/iwslt14.tokenized.de-en/
#   train.de, train.en, valid.de, valid.en, test.de, test.en, code

set -e

# Output directory
OUTDIR=data/iwslt14.tokenized.de-en
mkdir -p $OUTDIR

# Temp directory for intermediate files
TMPDIR=$(mktemp -d)
trap "rm -rf $TMPDIR" EXIT

echo "Temporary directory: $TMPDIR"

# Download and extract data using Python/HuggingFace
echo "Downloading IWSLT'14 de-en data from HuggingFace..."
python3 << 'EOF'
import os
import sys

try:
    from datasets import load_dataset
except ImportError:
    print("Installing datasets library...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "datasets"])
    from datasets import load_dataset

tmpdir = os.environ.get('TMPDIR', '/tmp')

print("Loading IWSLT2017 de-en dataset...")
# IWSLT2017 contains IWSLT14 data as a subset
dataset = load_dataset("iwslt2017", "iwslt2017-de-en", trust_remote_code=True)

# Extract train/validation/test splits
for split_name, hf_split in [("train", "train"), ("valid", "validation"), ("test", "test")]:
    print(f"Processing {split_name}...")
    split_data = dataset[hf_split]

    de_file = os.path.join(tmpdir, f"{split_name}.de")
    en_file = os.path.join(tmpdir, f"{split_name}.en")

    with open(de_file, "w", encoding="utf-8") as f_de, \
         open(en_file, "w", encoding="utf-8") as f_en:
        for example in split_data:
            de_text = example["translation"]["de"].lower().strip()
            en_text = example["translation"]["en"].lower().strip()
            if de_text and en_text:  # Skip empty lines
                f_de.write(de_text + "\n")
                f_en.write(en_text + "\n")

    print(f"  {split_name}: {len(split_data)} examples")

print("Data extraction complete!")
EOF

echo "Data downloaded successfully."

# Clone subword-nmt if not present
if [ ! -d "subword-nmt" ]; then
    echo "Cloning subword-nmt..."
    git clone https://github.com/rsennrich/subword-nmt.git --depth 1
fi

BPEROOT=subword-nmt/subword_nmt
BPE_TOKENS=10000

# Learn BPE on training data
echo "Learning BPE with $BPE_TOKENS merge operations..."
cat $TMPDIR/train.de $TMPDIR/train.en > $TMPDIR/train.both
python $BPEROOT/learn_bpe.py -s $BPE_TOKENS < $TMPDIR/train.both > $OUTDIR/code

# Apply BPE
echo "Applying BPE..."
for l in de en; do
    python $BPEROOT/apply_bpe.py -c $OUTDIR/code < $TMPDIR/train.$l > $OUTDIR/train.$l
    python $BPEROOT/apply_bpe.py -c $OUTDIR/code < $TMPDIR/valid.$l > $OUTDIR/valid.$l
    python $BPEROOT/apply_bpe.py -c $OUTDIR/code < $TMPDIR/test.$l > $OUTDIR/test.$l
done

# Print statistics
echo ""
echo "Data preparation complete!"
echo "Output directory: $OUTDIR"
echo ""
echo "Statistics:"
for f in train valid test; do
    for l in de en; do
        lines=$(wc -l < $OUTDIR/$f.$l)
        echo "  $f.$l: $lines lines"
    done
done
