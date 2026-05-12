#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-$ROOT/CueFilter/results/formal_sample_parallel}"
LOG_DIR="$RUN_ROOT/logs"
MASTER_LOG="$LOG_DIR/queue_${RUN_TS}.log"
GPU_IDS="${GPU_IDS:-0 1}"
GPU_JOBS_PER_GPU="${GPU_JOBS_PER_GPU:-2}"
CPU_JOBS="${CPU_JOBS:-1}"
SKIP_TABLE1="${SKIP_TABLE1:-1}"
DRY_RUN="${DRY_RUN:-0}"
SEEDS="${SEEDS:-42}"
RERUN_EXISTING="${RERUN_EXISTING:-0}"

mkdir -p "$LOG_DIR"
exec > >(tee -a "$MASTER_LOG") 2>&1

echo "Formal parallel CueFilter queue started at $(date '+%F %T %Z')"
echo "Run root: $RUN_ROOT"
echo "Master log: $MASTER_LOG"
echo "GPU ids: $GPU_IDS"
echo "GPU jobs per GPU: $GPU_JOBS_PER_GPU"
echo "CPU jobs: $CPU_JOBS"
echo "Seeds: $SEEDS"
echo "Rerun existing shards: $RERUN_EXISTING"
echo "Skip Table 1: $SKIP_TABLE1"
echo "Dry run: $DRY_RUN"

EXTRA_ARGS=()
if [[ "$SKIP_TABLE1" == "1" ]]; then
    EXTRA_ARGS+=(--skip-table1)
fi
if [[ "$DRY_RUN" == "1" ]]; then
    EXTRA_ARGS+=(--dry-run)
fi
if [[ "$RERUN_EXISTING" == "1" ]]; then
    EXTRA_ARGS+=(--rerun-existing)
fi

python -u scripts/run_cue_formal_experiments_parallel.py \
    --run-root "$RUN_ROOT" \
    --gpus $GPU_IDS \
    --gpu-jobs-per-gpu "$GPU_JOBS_PER_GPU" \
    --cpu-jobs "$CPU_JOBS" \
    --seeds $SEEDS \
    "${EXTRA_ARGS[@]}"

echo "Formal parallel CueFilter queue finished at $(date '+%F %T %Z')"
