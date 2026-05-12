#!/bin/bash
# LODO cross-dataset runner: 2 GPU, sequential pairs
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OUT_DIR="$ROOT/CueFilter/results/lodo_cross_dataset"
mkdir -p "$OUT_DIR/logs" "$OUT_DIR/shards" "$OUT_DIR/raw" "$OUT_DIR/details"
rm -f "$OUT_DIR/shards"/*.csv

run_one() {
    local target=$1 model=$2 gpu=$3
    local train=""
    for d in edaic pdch cmdc mandic; do
        [ "$d" != "$target" ] && train="$train $d"
    done
    echo "[$(date +%H:%M:%S)] START target=$target model=$model gpu=cuda:$gpu"
    python -u CueFilter/run_generalization_experiments.py \
        --mode cross-dataset --train-datasets $train --test-datasets $target \
        --models $model --seeds 42 \
        --cue-role patient --speech-scope participant \
        --boundary-tolerance-sec 1.0 --eval-level sample --segment-first \
        --device cuda:$gpu \
        --detail-output-dir "$OUT_DIR/details" \
        --output "$OUT_DIR/shards/${target}__${model}.csv" \
        > "$OUT_DIR/logs/${target}__${model}.log" 2>&1
    echo "[$(date +%H:%M:%S)] DONE  target=$target model=$model gpu=cuda:$gpu"
}

MODELS="DepAudioNet DMPF DALF STFN DisNet"
TARGETS="edaic pdch cmdc mandic"

# Build all task pairs: (target, model, gpu)
TASKS=()
for TARGET in $TARGETS; do
    for MODEL in $MODELS; do
        TASKS+=("$TARGET|$MODEL")
    done
done

echo "=============================================="
echo "LODO Cross-Dataset: ${#TASKS[@]} tasks, 1 per GPU"
echo "Start: $(date)"
echo "=============================================="

total=${#TASKS[@]}
idx=0
while [ $idx -lt $total ]; do
    IFS='|' read -r t0 m0 <<< "${TASKS[$idx]}"
    run_one "$t0" "$m0" 0 &
    pid0=$!
    
    idx=$((idx + 1))
    if [ $idx -lt $total ]; then
        IFS='|' read -r t1 m1 <<< "${TASKS[$idx]}"
        run_one "$t1" "$m1" 1 &
        pid1=$!
        idx=$((idx + 1))
        wait $pid0 $pid1
    else
        wait $pid0
    fi
done

echo ""
echo "=============================================="
echo "All training done: $(date)"
echo "=============================================="

# Merge shards per target
echo ""
echo "=== Merging shards ==="
FIRST_MODEL="${MODELS%% *}"
for TARGET in $TARGETS; do
    out="$OUT_DIR/raw/table7_generalization_cross_lodo_${TARGET}.csv"
    head -1 "$OUT_DIR/shards/${TARGET}__${FIRST_MODEL}.csv" > "$out"
    for MODEL in $MODELS; do
        f="$OUT_DIR/shards/${TARGET}__${MODEL}.csv"
        [ -f "$f" ] && tail -n+2 "$f" >> "$out"
    done
    nrows=$(tail -n+2 "$out" 2>/dev/null | wc -l)
    echo "  $TARGET: $nrows rows"
done

# Assemble final paper table
echo ""
echo "=== Assembling paper table ==="
python -u scripts/assemble_formal_tables.py --run-root "$OUT_DIR"

echo ""
echo "=============================================="
echo "=== Cross-Dataset LODO Table ==="
echo "=============================================="
cat "$OUT_DIR/tables/Table_cross_dataset_cf.md" 2>/dev/null || echo "(table not found)"
echo ""
echo "Done: $(date)"
