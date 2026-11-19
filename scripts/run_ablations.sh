#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: FEATURE_DIR=... PATHWAY_DIR=... CSV_PATH=... bash scripts/run_ablations.sh <task>"
    exit 1
fi

TASK="$1"
ROOT="${RESULTS_ROOT:-results/ablations}/${TASK}"

RESULTS_DIR="${ROOT}/direct_alignment" bash scripts/run_task.sh "$TASK" --kd_weight 0 --align_weight 1
RESULTS_DIR="${ROOT}/no_pathway_supervision" bash scripts/run_task.sh "$TASK" --kd_weight 0 --align_weight 0
RESULTS_DIR="${ROOT}/no_slot_diversity" bash scripts/run_task.sh "$TASK" --diversity_weight 0

for MEMORY in 4 8 16 24; do
    for SLOTS in 4 8 16 24 48 96; do
        RESULTS_DIR="${ROOT}/memory${MEMORY}_slots${SLOTS}" bash scripts/run_task.sh "$TASK" \
            --num_memory "$MEMORY" --num_slots "$SLOTS"
    done
done
