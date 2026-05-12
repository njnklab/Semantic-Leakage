#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

DATASETS=("$@")
if [ ${#DATASETS[@]} -eq 0 ]; then
    DATASETS=(cmdc pdch)
fi

for dataset in "${DATASETS[@]}"; do
    echo
    echo "===== Cue score analysis | ${dataset} ====="
    python Experiments/CueAnaly/cue_score_analysis.py --dataset "$dataset"

    echo
    echo "===== Cue wordcloud | ${dataset} ====="
    python Experiments/CueAnaly/cue_wordcloud.py --dataset "$dataset"
done

mkdir -p Experiments/CueAnaly/outputs/shared

echo
echo "===== Cue coverage | all datasets ====="
python Experiments/CueAnaly/cue_coverage_stats.py \
    -d all \
    --output Experiments/CueAnaly/outputs/shared/cue_coverage_all.csv

python Experiments/CueAnaly/cue_coverage_stats.py \
    -d all \
    --output Experiments/CueAnaly/outputs/shared/cue_coverage_all.md
