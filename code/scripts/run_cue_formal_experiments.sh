#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="$ROOT/CueFilter/results/formal"
TABLE_DIR="$RUN_ROOT/tables"
LOG_DIR="$RUN_ROOT/logs"
CKPT_DIR="$RUN_ROOT/shared_stage1_ckpts"
MASTER_LOG="$LOG_DIR/queue_${RUN_TS}.log"

mkdir -p "$TABLE_DIR" "$LOG_DIR" "$CKPT_DIR"
exec > >(tee -a "$MASTER_LOG") 2>&1

BASELINE_DATASETS=(edaic mandic)
BASELINE_MODELS=(SVM RF DepAudioNet DisNet DMPF DALF STFN)
CF_MODELS=(DepAudioNet DisNet DMPF DALF STFN)
SEEDS=(0 1 2 3 4)
CKPT_TEMPLATE="$CKPT_DIR/{model}_seed{seed}.pt"

run_step() {
    local step_name="$1"
    shift
    local step_log="$LOG_DIR/${step_name}.log"

    echo
    echo "===== ${step_name} | START $(date '+%F %T %Z') ====="
    "$@" 2>&1 | tee "$step_log"
    local status=${PIPESTATUS[0]}
    echo "===== ${step_name} | END $(date '+%F %T %Z') | status=${status} ====="
    return "${status}"
}

echo "Formal CueFilter queue started at $(date '+%F %T %Z')"
echo "Results root: $RUN_ROOT"
echo "Master log: $MASTER_LOG"
echo
echo "Execution order follows paper.txt Section V core results:"
echo "1. Cue validity analysis"
echo "2. Patient-cue-centered auditing"
echo "3. Role-specific cue impact analysis"
echo "4. Functional CueFilter controls"
echo "5. Frozen-feature reuse"
echo "6. Suppression strength and semantic probes"
echo "7. Cross-dataset / cross-gender / cross-age generalization"
echo
echo "Current available cue datasets: E-DAIC, ManDIC"
echo "Cross-age runs are restricted to adolescent <-> young-adult because the adult bin is currently too small for a stable formal main-table transfer setting."

run_step "01_cue_validity_edaic" \
    python -u Experiments/Comparison/emotion_classification.py \
    -d edaic \
    --output "$TABLE_DIR/01_cue_validity_edaic.md"

run_step "01_cue_validity_mandic" \
    python -u Experiments/Comparison/emotion_classification.py \
    -d mandic \
    --output "$TABLE_DIR/01_cue_validity_mandic.md"

run_step "02_patient_baseline" \
    python -u CueFilter/Baseline/run_segment_experiments.py \
    --datasets "${BASELINE_DATASETS[@]}" \
    --experiment patient-baseline \
    --models "${BASELINE_MODELS[@]}" \
    --seeds "${SEEDS[@]}" \
    --output "$TABLE_DIR/02_patient_baseline.md"

run_step "03_role_effect" \
    python -u CueFilter/Baseline/run_segment_experiments.py \
    --datasets "${BASELINE_DATASETS[@]}" \
    --experiment role-effect \
    --models "${BASELINE_MODELS[@]}" \
    --seeds "${SEEDS[@]}" \
    --output "$TABLE_DIR/03_role_effect.md"

run_step "04_06_cuefilter_functional" \
    python -u CueFilter/run_functional_experiments.py \
    --datasets edaic pdch cmdc mandic \
    --models "${CF_MODELS[@]}" \
    --seeds "${SEEDS[@]}" \
    --shared-pretrain-datasets edaic pdch cmdc mandic \
    --load-shared-cuefilter "$CKPT_TEMPLATE" \
    --save-shared-cuefilter "$CKPT_TEMPLATE" \
    --output-dir "$RUN_ROOT/cuefilter_functional"

run_step "06a_cross_dataset_edaic_to_mandic" \
    python -u CueFilter/run_generalization_experiments.py \
    --mode cross-dataset \
    --train-datasets edaic \
    --test-datasets mandic \
    --models "${CF_MODELS[@]}" \
    --seeds "${SEEDS[@]}" \
    --shared-pretrain-datasets "${BASELINE_DATASETS[@]}" \
    --load-shared-cuefilter "$CKPT_TEMPLATE" \
    --save-shared-cuefilter "$CKPT_TEMPLATE" \
    --output "$TABLE_DIR/06a_cross_dataset_edaic_to_mandic.md"

run_step "06b_cross_dataset_mandic_to_edaic" \
    python -u CueFilter/run_generalization_experiments.py \
    --mode cross-dataset \
    --train-datasets mandic \
    --test-datasets edaic \
    --models "${CF_MODELS[@]}" \
    --seeds "${SEEDS[@]}" \
    --shared-pretrain-datasets "${BASELINE_DATASETS[@]}" \
    --load-shared-cuefilter "$CKPT_TEMPLATE" \
    --save-shared-cuefilter "$CKPT_TEMPLATE" \
    --output "$TABLE_DIR/06b_cross_dataset_mandic_to_edaic.md"

run_step "06c_cross_gender_male_to_female" \
    python -u CueFilter/run_generalization_experiments.py \
    --mode cross-gender \
    --dataset mandic \
    --train-gender male \
    --test-gender female \
    --models "${CF_MODELS[@]}" \
    --seeds "${SEEDS[@]}" \
    --shared-pretrain-datasets "${BASELINE_DATASETS[@]}" \
    --load-shared-cuefilter "$CKPT_TEMPLATE" \
    --save-shared-cuefilter "$CKPT_TEMPLATE" \
    --output "$TABLE_DIR/06c_cross_gender_male_to_female.md"

run_step "06d_cross_gender_female_to_male" \
    python -u CueFilter/run_generalization_experiments.py \
    --mode cross-gender \
    --dataset mandic \
    --train-gender female \
    --test-gender male \
    --models "${CF_MODELS[@]}" \
    --seeds "${SEEDS[@]}" \
    --shared-pretrain-datasets "${BASELINE_DATASETS[@]}" \
    --load-shared-cuefilter "$CKPT_TEMPLATE" \
    --save-shared-cuefilter "$CKPT_TEMPLATE" \
    --output "$TABLE_DIR/06d_cross_gender_female_to_male.md"

run_step "06e_cross_age_adolescent_to_young_adult" \
    python -u CueFilter/run_generalization_experiments.py \
    --mode cross-age \
    --dataset mandic \
    --train-age-bin adolescent \
    --test-age-bin young-adult \
    --models "${CF_MODELS[@]}" \
    --seeds "${SEEDS[@]}" \
    --shared-pretrain-datasets "${BASELINE_DATASETS[@]}" \
    --load-shared-cuefilter "$CKPT_TEMPLATE" \
    --save-shared-cuefilter "$CKPT_TEMPLATE" \
    --output "$TABLE_DIR/06e_cross_age_adolescent_to_young_adult.md"

run_step "06f_cross_age_young_adult_to_adolescent" \
    python -u CueFilter/run_generalization_experiments.py \
    --mode cross-age \
    --dataset mandic \
    --train-age-bin young-adult \
    --test-age-bin adolescent \
    --models "${CF_MODELS[@]}" \
    --seeds "${SEEDS[@]}" \
    --shared-pretrain-datasets "${BASELINE_DATASETS[@]}" \
    --load-shared-cuefilter "$CKPT_TEMPLATE" \
    --save-shared-cuefilter "$CKPT_TEMPLATE" \
    --output "$TABLE_DIR/06f_cross_age_young_adult_to_adolescent.md"

echo
echo "Formal CueFilter queue finished at $(date '+%F %T %Z')"
