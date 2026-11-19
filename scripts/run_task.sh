#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: FEATURE_DIR=... PATHWAY_DIR=... CSV_PATH=... bash scripts/run_task.sh <task> [extra train.py args]"
    echo "Tasks: $(ls configs | sed 's/\.json//' | tr '\n' ' ')"
    exit 1
fi

TASK="$1"
shift
: "${FEATURE_DIR:?set FEATURE_DIR to the directory of UNI2 .h5 features}"
: "${PATHWAY_DIR:?set PATHWAY_DIR to the patient_matrices directory of the cohort}"
: "${CSV_PATH:?set CSV_PATH to the label table of the task}"
RESULTS_DIR="${RESULTS_DIR:-results/${TASK}}"

python train.py --config "configs/${TASK}.json" \
    --feature_dir "$FEATURE_DIR" --pathway_dir "$PATHWAY_DIR" --csv_path "$CSV_PATH" \
    --results_dir "$RESULTS_DIR" "$@"

python evaluate.py --config "configs/${TASK}.json" \
    --feature_dir "$FEATURE_DIR" --pathway_dir "$PATHWAY_DIR" --csv_path "$CSV_PATH" \
    --results_dir "$RESULTS_DIR" "$@"
