#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="$ROOT/CueFilter/results/formal_sample"
RAW_DIR="$RUN_ROOT/raw"
TABLE_DIR="$RUN_ROOT/tables"
LOG_DIR="$RUN_ROOT/logs"
CKPT_DIR="$RUN_ROOT/shared_stage1_ckpts"
MASTER_LOG="$LOG_DIR/queue_${RUN_TS}.log"

mkdir -p "$RAW_DIR" "$TABLE_DIR" "$LOG_DIR" "$CKPT_DIR"
exec > >(tee -a "$MASTER_LOG") 2>&1

FORMAL_DATASETS=(edaic pdch cmdc mandic)
BASELINE_DATASETS=("${FORMAL_DATASETS[@]}")
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

echo "Formal sample-level CueFilter queue started at $(date '+%F %T %Z')"
echo "Results root: $RUN_ROOT"
echo "Master log: $MASTER_LOG"
echo
echo "Execution order follows the sample-level paper structure:"
echo "1. Cue validity analysis"
echo "2. Cue coverage statistics"
echo "3. Main patient-cue auditing"
echo "4. Role-specific cue impact analysis"
echo "5. CueFilter localization + plug-and-play mitigation"
echo "6. Generalization analysis"
echo "7. Assemble final paper tables"
echo

run_step "01_table1_cue_validity" \
    python -u Experiments/Comparison/emotion_classification.py \
    -d all \
    --output "$RAW_DIR/table1_cue_validity.csv"

run_step "02_table2_cue_coverage" \
    python -u Experiments/CueAnaly/cue_coverage_stats.py \
    -d all \
    --output "$RAW_DIR/table2_cue_coverage.csv"

run_step "02b_table4_role_coverage_patient" \
    python -u Experiments/CueAnaly/cue_coverage_stats.py \
    -d all \
    --cue-role patient \
    --speech-scope dialogue \
    --output "$RAW_DIR/table4_role_coverage_patient.csv"

run_step "02c_table4_role_coverage_doctor" \
    python -u Experiments/CueAnaly/cue_coverage_stats.py \
    -d all \
    --cue-role doctor \
    --speech-scope dialogue \
    --output "$RAW_DIR/table4_role_coverage_doctor.csv"

run_step "02d_table4_role_coverage_all" \
    python -u Experiments/CueAnaly/cue_coverage_stats.py \
    -d all \
    --cue-role all \
    --speech-scope dialogue \
    --output "$RAW_DIR/table4_role_coverage_all.csv"

run_step "03_main_patient_audit" \
    python -u CueFilter/Baseline/run_segment_experiments.py \
    --datasets "${BASELINE_DATASETS[@]}" \
    --experiment patient-baseline \
    --train-variant pre \
    --variants pre cue-only cue-removed random-removed \
    --models "${BASELINE_MODELS[@]}" \
    --seeds "${SEEDS[@]}" \
    --eval-level sample \
    --segment-first \
    --output "$RAW_DIR/table3_main_patient_audit_raw.csv"

run_step "04_role_specific_audit" \
    python -u CueFilter/Baseline/run_segment_experiments.py \
    --datasets "${BASELINE_DATASETS[@]}" \
    --experiment role-effect \
    --train-variant pre \
    --variants pre cue-removed random-removed \
    --models "${BASELINE_MODELS[@]}" \
    --seeds "${SEEDS[@]}" \
    --eval-level sample \
    --segment-first \
    --output "$RAW_DIR/table4_role_specific_raw.csv"

run_step "05_localization_mitigation" \
    python -u CueFilter/run_mitigation_experiments.py \
    --datasets "${BASELINE_DATASETS[@]}" \
    --models "${CF_MODELS[@]}" \
    --seeds "${SEEDS[@]}" \
    --shared-pretrain-datasets "${FORMAL_DATASETS[@]}" \
    --load-shared-cuefilter "$CKPT_TEMPLATE" \
    --save-shared-cuefilter "$CKPT_TEMPLATE" \
    --eval-level sample \
    --segment-first \
    --output "$RAW_DIR/table5_6_localization_mitigation_raw.csv"

run_step "06a_cross_dataset_edaic_to_mandic" \
    python -u CueFilter/run_generalization_experiments.py \
    --mode cross-dataset \
    --train-datasets edaic \
    --test-datasets mandic \
    --models "${CF_MODELS[@]}" \
    --seeds "${SEEDS[@]}" \
    --shared-pretrain-datasets "${FORMAL_DATASETS[@]}" \
    --load-shared-cuefilter "$CKPT_TEMPLATE" \
    --save-shared-cuefilter "$CKPT_TEMPLATE" \
    --eval-level sample \
    --segment-first \
    --output "$RAW_DIR/table7_generalization_06a_cross_dataset_edaic_to_mandic.csv"

run_step "06b_cross_dataset_mandic_to_edaic" \
    python -u CueFilter/run_generalization_experiments.py \
    --mode cross-dataset \
    --train-datasets mandic \
    --test-datasets edaic \
    --models "${CF_MODELS[@]}" \
    --seeds "${SEEDS[@]}" \
    --shared-pretrain-datasets "${FORMAL_DATASETS[@]}" \
    --load-shared-cuefilter "$CKPT_TEMPLATE" \
    --save-shared-cuefilter "$CKPT_TEMPLATE" \
    --eval-level sample \
    --segment-first \
    --output "$RAW_DIR/table7_generalization_06b_cross_dataset_mandic_to_edaic.csv"

run_step "06c_cross_gender_male_to_female" \
    python -u CueFilter/run_generalization_experiments.py \
    --mode cross-gender \
    --dataset mandic \
    --train-gender male \
    --test-gender female \
    --models "${CF_MODELS[@]}" \
    --seeds "${SEEDS[@]}" \
    --shared-pretrain-datasets "${FORMAL_DATASETS[@]}" \
    --load-shared-cuefilter "$CKPT_TEMPLATE" \
    --save-shared-cuefilter "$CKPT_TEMPLATE" \
    --eval-level sample \
    --segment-first \
    --output "$RAW_DIR/table7_generalization_06c_cross_gender_male_to_female.csv"

run_step "06d_cross_gender_female_to_male" \
    python -u CueFilter/run_generalization_experiments.py \
    --mode cross-gender \
    --dataset mandic \
    --train-gender female \
    --test-gender male \
    --models "${CF_MODELS[@]}" \
    --seeds "${SEEDS[@]}" \
    --shared-pretrain-datasets "${FORMAL_DATASETS[@]}" \
    --load-shared-cuefilter "$CKPT_TEMPLATE" \
    --save-shared-cuefilter "$CKPT_TEMPLATE" \
    --eval-level sample \
    --segment-first \
    --output "$RAW_DIR/table7_generalization_06d_cross_gender_female_to_male.csv"

run_step "06e_cross_age_adolescent_to_young_adult" \
    python -u CueFilter/run_generalization_experiments.py \
    --mode cross-age \
    --dataset mandic \
    --train-age-bin adolescent \
    --test-age-bin young-adult \
    --models "${CF_MODELS[@]}" \
    --seeds "${SEEDS[@]}" \
    --shared-pretrain-datasets "${FORMAL_DATASETS[@]}" \
    --load-shared-cuefilter "$CKPT_TEMPLATE" \
    --save-shared-cuefilter "$CKPT_TEMPLATE" \
    --eval-level sample \
    --segment-first \
    --output "$RAW_DIR/table7_generalization_06e_cross_age_adolescent_to_young_adult.csv"

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
    --eval-level sample \
    --segment-first \
    --output "$RAW_DIR/table7_generalization_06f_cross_age_young_adult_to_adolescent.csv"

run_step "07_assemble_formal_tables" \
    python -u scripts/assemble_formal_tables.py \
    --run-root "$RUN_ROOT"

echo
echo "Formal sample-level CueFilter queue finished at $(date '+%F %T %Z')"
