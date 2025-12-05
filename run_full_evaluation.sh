#!/bin/bash
###############################################################################
# SLAKE Full Evaluation Script
# Runs baseline and Stage 2 evaluations on full dataset with sleep prevention
#
# Usage: ./run_full_evaluation.sh
#
# This script will:
# 1. Prepare full SLAKE dataset (9,835 train samples)
# 2. Build text and image indexes
# 3. Run baseline evaluation (text-only, no rewriting, no reranking)
# 4. Run Stage 2 evaluation (HyDE + MiniLM reranker)
#
# Total estimated time: 24-30 hours on MacBook M4
###############################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR:${NC} $1"
    exit 1
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING:${NC} $1"
}

# Check if virtual environment is activated
if [[ -z "$VIRTUAL_ENV" ]]; then
    error "Virtual environment not activated. Run: source .venv/bin/activate"
fi

# Create log directory
mkdir -p logs

# Set environment variables for M4 compatibility
export PYTORCH_ENABLE_MPS_FALLBACK=1

log "Starting SLAKE full evaluation pipeline"
log "Estimated total time: 24-30 hours"
log "Progress bars will show real-time status"

###############################################################################
# Step 1: Prepare Full Dataset
###############################################################################

log "Step 1/5: Preparing full SLAKE dataset (9,835 train, 2,099 val, 2,094 test)..."
python scripts/prepare_slake.py --splits train validation test 2>&1 | tee logs/01_prepare_data.log

if [ ! -f "dataset/slake/json_files/train.json" ]; then
    error "Dataset preparation failed. Check logs/01_prepare_data.log"
fi

log "✓ Dataset preparation complete"

###############################################################################
# Step 2: Build Text Index
###############################################################################

log "Step 2/5: Building text index (9,835 training samples)..."
log "Estimated time: 10-15 minutes"

python build_corpus.py \
  --train-file dataset/slake/json_files/train.json \
  --index-file retriever/slake/text.index \
  --texts-file retriever/slake/texts.json \
  2>&1 | tee logs/02_build_text_index.log

if [ ! -f "retriever/slake/text.index" ]; then
    error "Text index build failed. Check logs/02_build_text_index.log"
fi

log "✓ Text index complete"

###############################################################################
# Step 3: Build Image Index (with caffeinate to prevent sleep)
###############################################################################

log "Step 3/5: Building image index (CPU mode, with sleep prevention)..."
log "Estimated time: 3-4 hours"
warn "Mac will stay awake during this step (caffeinate enabled)"

caffeinate -i python build_image_index.py \
  --img-dir dataset/slake/imgs \
  --train-file dataset/slake/json_files/train.json \
  --index-out retriever/slake/image.index \
  --metadata-out retriever/slake/image_texts.json \
  2>&1 | tee logs/03_build_image_index.log

# Image index is optional - warn but continue if it fails
if [ ! -f "retriever/slake/image.index" ]; then
    warn "Image index build failed. Evaluations will use text-only retrieval."
    warn "This is OK - results are still valid for comparison."
else
    log "✓ Image index complete"
fi

###############################################################################
# Step 4: Run Baseline Evaluation (with caffeinate)
###############################################################################

log "Step 4/5: Running baseline evaluation (text-only, no rewriting, no reranking)..."
log "Estimated time: 8-10 hours"
log "Config: configs/slake_baseline.yaml"
warn "Mac will stay awake during this step (caffeinate enabled)"

caffeinate -i python eval_slake.py \
  --config configs/slake_baseline.yaml \
  --split test \
  --top-k 1 \
  --output result/slake_baseline_predictions.json \
  2>&1 | tee logs/04_eval_baseline.log

if [ ! -f "result/slake_baseline_predictions.json" ]; then
    error "Baseline evaluation failed. Check logs/04_eval_baseline.log"
fi

# Extract and display baseline accuracy
BASELINE_ACC=$(python -c "import json; data=json.load(open('result/slake_baseline_predictions.json')); print(f'{data[\"accuracy\"]:.4f}')")
log "✓ Baseline evaluation complete"
log "  Baseline Accuracy: ${BASELINE_ACC}"

###############################################################################
# Step 5: Run Stage 2 Evaluation (with caffeinate)
###############################################################################

log "Step 5/5: Running Stage 2 evaluation (HyDE + MiniLM reranker)..."
log "Estimated time: 12-15 hours"
log "Config: configs/slake_stage2.yaml"
warn "Mac will stay awake during this step (caffeinate enabled)"

caffeinate -i python eval_slake.py \
  --config configs/slake_stage2.yaml \
  --split test \
  --top-k 3 \
  --output result/slake_stage2_predictions.json \
  2>&1 | tee logs/05_eval_stage2.log

if [ ! -f "result/slake_stage2_predictions.json" ]; then
    error "Stage 2 evaluation failed. Check logs/05_eval_stage2.log"
fi

# Extract and display stage2 accuracy
STAGE2_ACC=$(python -c "import json; data=json.load(open('result/slake_stage2_predictions.json')); print(f'{data[\"accuracy\"]:.4f}')")
log "✓ Stage 2 evaluation complete"
log "  Stage 2 Accuracy: ${STAGE2_ACC}"

###############################################################################
# Summary
###############################################################################

log "════════════════════════════════════════════════════════════════"
log "                    EVALUATION COMPLETE                          "
log "════════════════════════════════════════════════════════════════"
log ""
log "Results Summary:"
log "  Dataset: SLAKE (9,835 train, 2,094 test)"
log "  Baseline Accuracy:  ${BASELINE_ACC}"
log "  Stage 2 Accuracy:   ${STAGE2_ACC}"
log ""
log "Output Files:"
log "  - result/slake_baseline_predictions.json"
log "  - result/slake_stage2_predictions.json"
log ""
log "Logs saved to:"
log "  - logs/01_prepare_data.log"
log "  - logs/02_build_text_index.log"
log "  - logs/03_build_image_index.log"
log "  - logs/04_eval_baseline.log"
log "  - logs/05_eval_stage2.log"
log ""
log "════════════════════════════════════════════════════════════════"
