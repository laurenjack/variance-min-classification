#!/bin/bash
# Prepare IWSLT'14 German-English data for Transformer training.
# Based on fairseq's prepare-iwslt14.sh but without fairseq dependency.
#
# This script:
# 1. Clones Moses tokenizer (Perl scripts)
# 2. Installs subword-nmt via pip (for BPE)
# 3. Downloads IWSLT'14 de-en raw data
# 4. Applies Moses tokenization
# 5. Lowercases text
# 6. Learns joint BPE vocabulary (10K merge operations)
# 7. Applies BPE to train/valid/test splits
# 8. Outputs to data/iwslt14.tokenized.de-en/
#
# Usage:
#   ./infra/prepare_iwslt14.sh
#
# Output: data/iwslt14.tokenized.de-en/
#   train.de, train.en, valid.de, valid.en, test.de, test.en, code

set -e

SCRIPTS_DIR=mosesdecoder/scripts
TOKENIZER=$SCRIPTS_DIR/tokenizer/tokenizer.perl
LC=$SCRIPTS_DIR/tokenizer/lowercase.perl
CLEAN=$SCRIPTS_DIR/training/clean-corpus-n.perl
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

# Clone Moses if not present
if [ ! -d "mosesdecoder" ]; then
    echo "Cloning Moses tokenizer..."
    git clone https://github.com/moses-smt/mosesdecoder.git --depth 1
fi

# Clone subword-nmt if not present
if [ ! -d "subword-nmt" ]; then
    echo "Cloning subword-nmt..."
    git clone https://github.com/rsennrich/subword-nmt.git --depth 1
fi

# Check for perl
if ! command -v perl &> /dev/null; then
    echo "Error: perl is required but not installed."
    exit 1
fi

# Download IWSLT'14 data
URL="https://wit3.fbk.eu/archive/2014-01/texts/de/en/de-en.tgz"
GZ=$TMPDIR/de-en.tgz

if [ ! -f "$GZ" ]; then
    echo "Downloading IWSLT'14 de-en data..."
    wget -O $GZ $URL
fi

echo "Extracting data..."
tar -xzf $GZ -C $TMPDIR

# Prepare raw data
echo "Preparing raw data..."
cd $TMPDIR/de-en
for l in $src $tgt; do
    # Concatenate all training talks
    cat train.tags.$lang.$l | \
        grep -v '<url>' | \
        grep -v '<talkid>' | \
        grep -v '<keywords>' | \
        sed -e 's/<title>//g' | \
        sed -e 's/<\/title>//g' | \
        sed -e 's/<description>//g' | \
        sed -e 's/<\/description>//g' > train.$l

    # Validation set (2010 dev set)
    cat IWSLT14.TED.dev2010.$lang.$l.xml | \
        grep '<seg id' | \
        sed -e 's/<seg id="[0-9]*">\s*//g' | \
        sed -e 's/\s*<\/seg>\s*//g' | \
        sed -e "s/\'/\'/g" > valid.$l

    # Test set (2010-2012 test sets combined)
    cat IWSLT14.TED.tst2010.$lang.$l.xml \
        IWSLT14.TED.tst2011.$lang.$l.xml \
        IWSLT14.TED.tst2012.$lang.$l.xml \
        IWSLT14.TED.tst2013.$lang.$l.xml \
        IWSLT14.TED.tst2014.$lang.$l.xml 2>/dev/null | \
        grep '<seg id' | \
        sed -e 's/<seg id="[0-9]*">\s*//g' | \
        sed -e 's/\s*<\/seg>\s*//g' | \
        sed -e "s/\'/\'/g" > test.$l || true
done
cd -

# Tokenize
echo "Tokenizing..."
for l in $src $tgt; do
    for f in train valid test; do
        perl $TOKENIZER -threads 8 -l $l < $TMPDIR/de-en/$f.$l > $TMPDIR/$f.tok.$l
    done
done

# Lowercase
echo "Lowercasing..."
for l in $src $tgt; do
    for f in train valid test; do
        perl $LC < $TMPDIR/$f.tok.$l > $TMPDIR/$f.tok.lc.$l
    done
done

# Clean training data (remove empty lines, lines > 175 tokens)
echo "Cleaning training data..."
perl $CLEAN $TMPDIR/train.tok.lc $src $tgt $TMPDIR/train.tok.lc.clean 1 175

# Learn BPE on training data
echo "Learning BPE with $BPE_TOKENS merge operations..."
cat $TMPDIR/train.tok.lc.clean.$src $TMPDIR/train.tok.lc.clean.$tgt > $TMPDIR/train.tok.lc.clean.both
python $BPEROOT/learn_bpe.py -s $BPE_TOKENS < $TMPDIR/train.tok.lc.clean.both > $OUTDIR/code

# Apply BPE
echo "Applying BPE..."
for l in $src $tgt; do
    python $BPEROOT/apply_bpe.py -c $OUTDIR/code < $TMPDIR/train.tok.lc.clean.$l > $OUTDIR/train.$l
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
