#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-CueFilter/results/formal_sample_parallel}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
SHARD_DIR="$ROOT/shards/table4_svm_fast"
LOG_DIR="$ROOT/logs"
RAW_OUT="$ROOT/raw/table4_svm_fast_raw.csv"

mkdir -p "$SHARD_DIR" "$LOG_DIR" "$ROOT/raw"

for dataset in edaic pdch cmdc; do
  src="$ROOT/shards/table4/${dataset}__SVM.csv"
  if [[ -f "$src" ]]; then
    cp "$src" "$SHARD_DIR/${dataset}__SVM.csv"
  else
    "$PYTHON_BIN" -u CueFilter/Baseline/run_segment_experiments.py \
      --datasets "$dataset" \
      --experiment role-effect \
      --train-variant pre \
      --variants pre cue-removed random-removed \
      --models SVM \
      --seeds 42 \
      --eval-level sample \
      --segment-first \
      --output "$SHARD_DIR/${dataset}__SVM.csv" \
      --device cpu \
      2>&1 | tee "$LOG_DIR/table4_fast_${dataset}__SVM.log"
  fi
done

for role in patient doctor all; do
  for variant in pre cue-removed random-removed; do
    "$PYTHON_BIN" -u CueFilter/Baseline/run_segment_experiments.py \
      --datasets mandic \
      --experiment role-effect \
      --role-effect-roles "$role" \
      --train-variant pre \
      --variants "$variant" \
      --models SVM \
      --seeds 42 \
      --eval-level sample \
      --segment-first \
      --output "$SHARD_DIR/mandic__SVM__${role}__${variant}.csv" \
      --device cpu \
      2>&1 | tee "$LOG_DIR/table4_fast_mandic__SVM__${role}__${variant}.log"
  done
done

awk 'FNR == 1 && NR != 1 { next } { print }' "$SHARD_DIR"/*.csv > "$RAW_OUT"
echo "Wrote $RAW_OUT"
