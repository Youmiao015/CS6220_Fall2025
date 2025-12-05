#!/bin/bash
###############################################################################
# SLAKE Stage 2 Evaluation Script
# Runs Stage 2 evaluation (HyDE + MiniLM reranker)
#
# Usage: ./eval_stage2.sh
#
# Prerequisites:
#   - Indexes must be built first (run ./build_indexes.sh)
#
# This script runs:
#   - HyDE query rewriting (2 hypothetical answers)
#   - MiniLM cross-encoder reranking
#   - Top-3 answer voting
#   - Like VQA-RAD Stage 2 (34.15% accuracy)
#
# Estimated time: 12-15 hours (2,094 test examples with query rewriting)
###############################################################################

set -e  # Exit on error

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log() { echo -e "${GREEN}[$(date +'%H:%M:%S')]${NC} $1"; }
error() { echo -e "${RED}[$(date +'%H:%M:%S')] ERROR:${NC} $1"; exit 1; }
warn() { echo -e "${YELLOW}[$(date +'%H:%M:%S')] WARN:${NC} $1"; }
info() { echo -e "${BLUE}[$(date +'%H:%M:%S')] INFO:${NC} $1"; }

# Check prerequisites
[[ -z "$VIRTUAL_ENV" ]] && error "Activate venv first: source .venv/bin/activate"
[ ! -f "retriever/slake/text.index" ] && error "Text index not found. Run ./build_indexes.sh first"
[ ! -f "dataset/slake/json_files/test.json" ] && error "Test data not found. Run ./build_indexes.sh first"

mkdir -p logs result

log "════════════════════════════════════════════════════════════════"
log "        SLAKE Stage 2 Evaluation"
log "════════════════════════════════════════════════════════════════"
info "Method: HyDE query rewriting + MiniLM reranker + voting"
info "Config: configs/slake_stage2.yaml"
info "Test samples: 2,094"
info "Estimated time: 12-15 hours (slower due to query rewriting)"
warn "Mac will stay awake (caffeinate enabled)"
log "════════════════════════════════════════════════════════════════"
log ""

# Record start time
START_TIME=$(date +%s)

# Run evaluation with sleep prevention
log "Starting Stage 2 evaluation..."
log "Progress bar will show real-time status"
log ""

caffeinate -i python eval_slake.py \
  --config configs/slake_stage2.yaml \
  --split test \
  --top-k 3 \
  --output result/slake_stage2_predictions.json \
  2>&1 | tee logs/eval_stage2.log

# Check if evaluation completed
[ ! -f "result/slake_stage2_predictions.json" ] && error "Evaluation failed. Check logs/eval_stage2.log"

# Calculate elapsed time
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(((ELAPSED % 3600) / 60))

# Extract results
ACCURACY=$(python -c "import json; d=json.load(open('result/slake_stage2_predictions.json')); print(f'{d[\"accuracy\"]:.4f}')")
TOTAL=$(python -c "import json; d=json.load(open('result/slake_stage2_predictions.json')); print(d['total_examples'])")

# Get baseline for comparison if exists
if [ -f "result/slake_baseline_predictions.json" ]; then
    BASELINE_ACC=$(python -c "import json; d=json.load(open('result/slake_baseline_predictions.json')); print(f'{d[\"accuracy\"]:.4f}')")
    IMPROVEMENT=$(python -c "print(f'{(${ACCURACY} - ${BASELINE_ACC}):.4f}')")
fi

###############################################################################
# Summary
###############################################################################

log ""
log "════════════════════════════════════════════════════════════════"
log "              STAGE 2 EVALUATION COMPLETE"
log "════════════════════════════════════════════════════════════════"
log ""
log "Results:"
log "  Method: Stage 2 (HyDE + MiniLM reranker + voting)"
log "  Test samples: ${TOTAL}"
log "  Accuracy: ${ACCURACY} (compare to VQA-RAD: 0.3415)"
log ""

if [ ! -z "$BASELINE_ACC" ]; then
    log "Comparison:"
    log "  Baseline: ${BASELINE_ACC}"
    log "  Stage 2:  ${ACCURACY}"
    log "  Difference: ${IMPROVEMENT}"
    log ""
fi

log "Time taken: ${HOURS}h ${MINUTES}m"
log ""
log "Output:"
log "  - result/slake_stage2_predictions.json"
log "  - logs/eval_stage2.log"
log ""
log "════════════════════════════════════════════════════════════════"
