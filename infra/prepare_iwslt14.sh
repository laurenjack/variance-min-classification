#!/bin/bash
# Prepare IWSLT'14 German-English data for Transformer training.
# Downloads pre-tokenized data from Stanford NLP mirror.
#
# This script:
# 1. Downloads pre-tokenized IWSLT'14 de-en data from Stanford NLP mirror
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

BPEROOT=subword-nmt/subword_nmt
BPE_TOKENS=10000

# Output directory
OUTDIR=data/iwslt14.tokenized.de-en
mkdir -p $OUTDIR

src=de
tgt=en
lang=de-en

# Temp directory for intermediate files
TMPDIR=$(mktemp -d)
trap "rm -rf $TMPDIR" EXIT

echo "Temporary directory: $TMPDIR"

# Clone subword-nmt if not present
if [ ! -d "subword-nmt" ]; then
    echo "Cloning subword-nmt..."
    git clone https://github.com/rsennrich/subword-nmt.git --depth 1
fi

# Download IWSLT'14 data from Stanford NLP mirror (pre-tokenized, lowercased)
# This mirror provides already-processed data from fairseq
STANFORD_URL="https://nlp.stanford.edu/projects/nmt/data/iwslt14.en-de"

echo "Downloading IWSLT'14 de-en data from Stanford NLP mirror..."
for f in train valid test; do
    for l in $src $tgt; do
        echo "  Downloading $f.$l..."
        wget -q -O $TMPDIR/$f.tok.lc.$l "$STANFORD_URL/$f.$l"
    done
done

echo "Data downloaded successfully."

# Note: Stanford data is already tokenized and lowercased
# Skip cleaning step - data is already clean

# Learn BPE on training data
echo "Learning BPE with $BPE_TOKENS merge operations..."
cat $TMPDIR/train.tok.lc.$src $TMPDIR/train.tok.lc.$tgt > $TMPDIR/train.tok.lc.both
python $BPEROOT/learn_bpe.py -s $BPE_TOKENS < $TMPDIR/train.tok.lc.both > $OUTDIR/code

# Apply BPE
echo "Applying BPE..."
for l in $src $tgt; do
    python $BPEROOT/apply_bpe.py -c $OUTDIR/code < $TMPDIR/train.tok.lc.$l > $OUTDIR/train.$l
    python $BPEROOT/apply_bpe.py -c $OUTDIR/code < $TMPDIR/valid.tok.lc.$l > $OUTDIR/valid.$l
    python $BPEROOT/apply_bpe.py -c $OUTDIR/code < $TMPDIR/test.tok.lc.$l > $OUTDIR/test.$l
done

# Print statistics
echo ""
echo "Data preparation complete!"
echo "Output directory: $OUTDIR"
echo ""
echo "Statistics:"
for f in train valid test; do
    for l in $src $tgt; do
        lines=$(wc -l < $OUTDIR/$f.$l)
        echo "  $f.$l: $lines lines"
    done
done
