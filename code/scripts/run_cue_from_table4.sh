#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-$ROOT/CueFilter/results/formal_sample_parallel}"
LOG_DIR="$RUN_ROOT/logs"
MASTER_LOG="$LOG_DIR/queue_from_table4_${RUN_TS}.log"
GPU_IDS="${GPU_IDS:-0 1}"
GPU_JOBS_PER_GPU="${GPU_JOBS_PER_GPU:-1}"
CPU_JOBS="${CPU_JOBS:-1}"
SEEDS="${SEEDS:-42}"
RERUN_EXISTING="${RERUN_EXISTING:-0}"
SKIP_FIGURES="${SKIP_FIGURES:-1}"

mkdir -p "$LOG_DIR"
exec > >(tee -a "$MASTER_LOG") 2>&1

echo "Formal CueFilter queue resumed from Table 4 at $(date '+%F %T %Z')"
echo "Run root: $RUN_ROOT"
echo "Master log: $MASTER_LOG"
echo "GPU ids: $GPU_IDS"
echo "GPU jobs per GPU: $GPU_JOBS_PER_GPU"
echo "CPU jobs: $CPU_JOBS"
echo "Seeds: $SEEDS"
echo "Rerun existing shards: $RERUN_EXISTING"
echo "Skip figures: $SKIP_FIGURES"

EXTRA_ARGS=(--skip-table1 --skip-serial --start-phase 04_role_specific_audit)
if [[ "$RERUN_EXISTING" == "1" ]]; then
    EXTRA_ARGS+=(--rerun-existing)
fi
if [[ "$SKIP_FIGURES" == "1" ]]; then
    EXTRA_ARGS+=(--skip-figures)
fi

python3 -u scripts/run_cue_formal_experiments_parallel.py \
    --run-root "$RUN_ROOT" \
    --gpus $GPU_IDS \
    --gpu-jobs-per-gpu "$GPU_JOBS_PER_GPU" \
    --cpu-jobs "$CPU_JOBS" \
    --seeds $SEEDS \
    "${EXTRA_ARGS[@]}"

echo "Formal CueFilter queue finished at $(date '+%F %T %Z')"
