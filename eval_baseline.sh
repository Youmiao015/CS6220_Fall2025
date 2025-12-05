#!/bin/bash
###############################################################################
# SLAKE Baseline Evaluation Script
# Runs baseline evaluation (text-only, no rewriting, no reranking)
#
# Usage: ./eval_baseline.sh
#
# Prerequisites:
#   - Indexes must be built first (run ./build_indexes.sh)
#
# This script runs:
#   - Text-only retrieval
#   - No query rewriting
#   - No reranking
#   - Top-1 answer (like VQA-RAD baseline: 35.03%)
#
# Estimated time: 8-10 hours (2,094 test examples)
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
log "        SLAKE Baseline Evaluation"
log "════════════════════════════════════════════════════════════════"
info "Method: Text-only retrieval, no rewriting, no reranking"
info "Config: configs/slake_baseline.yaml"
info "Test samples: 2,094"
info "Estimated time: 8-10 hours"
warn "Mac will stay awake (caffeinate enabled)"
log "════════════════════════════════════════════════════════════════"
log ""

# Record start time
START_TIME=$(date +%s)

# Run evaluation with sleep prevention
log "Starting baseline evaluation..."
log "Progress bar will show real-time status"
log ""

caffeinate -i python eval_slake.py \
  --config configs/slake_baseline.yaml \
  --split test \
  --top-k 1 \
  --output result/slake_baseline_predictions.json \
  2>&1 | tee logs/eval_baseline.log

# Check if evaluation completed
[ ! -f "result/slake_baseline_predictions.json" ] && error "Evaluation failed. Check logs/eval_baseline.log"

# Calculate elapsed time
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(((ELAPSED % 3600) / 60))

# Extract results
ACCURACY=$(python -c "import json; d=json.load(open('result/slake_baseline_predictions.json')); print(f'{d[\"accuracy\"]:.4f}')")
TOTAL=$(python -c "import json; d=json.load(open('result/slake_baseline_predictions.json')); print(d['total_examples'])")

###############################################################################
# Summary
###############################################################################

log ""
log "════════════════════════════════════════════════════════════════"
log "              BASELINE EVALUATION COMPLETE"
log "════════════════════════════════════════════════════════════════"
log ""
log "Results:"
log "  Method: Baseline (text-only, no rewriting, no reranking)"
log "  Test samples: ${TOTAL}"
log "  Accuracy: ${ACCURACY} (compare to VQA-RAD: 0.3503)"
log ""
log "Time taken: ${HOURS}h ${MINUTES}m"
log ""
log "Output:"
log "  - result/slake_baseline_predictions.json"
log "  - logs/eval_baseline.log"
log ""
log "Next step:"
log "  Run Stage 2 evaluation: ./eval_stage2.sh"
log "════════════════════════════════════════════════════════════════"
