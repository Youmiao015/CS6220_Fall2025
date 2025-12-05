#!/bin/bash
###############################################################################
# SLAKE Index Building Script
# Prepares dataset and builds text + image indexes
#
# Usage: ./build_indexes.sh
#
# This script will:
# 1. Prepare full SLAKE dataset (9,835 train samples)
# 2. Build text index (~10-15 min)
# 3. Build image index (~3-4 hours on CPU with M4 compatibility)
#
# Estimated time: 3.5-4.5 hours total
###############################################################################

set -e  # Exit on error

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

log() { echo -e "${GREEN}[$(date +'%H:%M:%S')]${NC} $1"; }
error() { echo -e "${RED}[$(date +'%H:%M:%S')] ERROR:${NC} $1"; exit 1; }
warn() { echo -e "${YELLOW}[$(date +'%H:%M:%S')] WARN:${NC} $1"; }

# Check virtual environment
[[ -z "$VIRTUAL_ENV" ]] && error "Activate venv first: source .venv/bin/activate"

# Set M4 compatibility
export PYTORCH_ENABLE_MPS_FALLBACK=1

mkdir -p logs

log "════════════════════════════════════════════════════════════════"
log "        SLAKE Index Building (Full Dataset)"
log "════════════════════════════════════════════════════════════════"

###############################################################################
# Step 1: Prepare Dataset
###############################################################################

log "Step 1/3: Preparing SLAKE dataset (downloading images + creating JSON)..."
log "Expected: 9,835 train, 2,099 validation, 2,094 test"

python scripts/prepare_slake.py --splits train validation test 2>&1 | tee logs/prepare_data.log

[ ! -f "dataset/slake/json_files/train.json" ] && error "Dataset prep failed. Check logs/prepare_data.log"

log "✓ Dataset ready"

###############################################################################
# Step 2: Build Text Index
###############################################################################

log "Step 2/3: Building text index..."
log "Time: ~10-15 minutes"

python build_corpus.py \
  --train-file dataset/slake/json_files/train.json \
  --index-file retriever/slake/text.index \
  --texts-file retriever/slake/texts.json \
  2>&1 | tee logs/build_text_index.log

[ ! -f "retriever/slake/text.index" ] && error "Text index failed. Check logs/build_text_index.log"

log "✓ Text index built: retriever/slake/text.index"

###############################################################################
# Step 3: Build Image Index (CPU mode with sleep prevention)
###############################################################################

log "Step 3/3: Building image index (CPU mode for M4 compatibility)..."
log "Time: ~3-4 hours"
warn "Using caffeinate to prevent Mac sleep"

caffeinate -i python build_image_index.py \
  --img-dir dataset/slake/imgs \
  --train-file dataset/slake/json_files/train.json \
  --index-out retriever/slake/image.index \
  --metadata-out retriever/slake/image_texts.json \
  2>&1 | tee logs/build_image_index.log

if [ ! -f "retriever/slake/image.index" ]; then
    warn "Image index failed. Evaluations will use text-only (still valid)."
else
    log "✓ Image index built: retriever/slake/image.index"
fi

###############################################################################
# Summary
###############################################################################

log "════════════════════════════════════════════════════════════════"
log "                   INDEX BUILDING COMPLETE"
log "════════════════════════════════════════════════════════════════"
log ""
log "Built indexes:"
log "  ✓ retriever/slake/text.index (9,835 samples)"
log "  ✓ retriever/slake/texts.json"
if [ -f "retriever/slake/image.index" ]; then
    log "  ✓ retriever/slake/image.index"
    log "  ✓ retriever/slake/image_texts.json"
fi
log ""
log "Next steps:"
log "  1. Run baseline eval:  ./eval_baseline.sh"
log "  2. Run stage2 eval:    ./eval_stage2.sh"
log "════════════════════════════════════════════════════════════════"
