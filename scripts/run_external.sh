#!/usr/bin/env bash
set -euo pipefail

: "${FEATURE_DIR:?set FEATURE_DIR to the UNI2 .h5 features of the TCGA-BRCA slides}"
: "${PATHWAY_DIR:?set PATHWAY_DIR to the TCGA-BRCA patient_matrices directory}"
: "${CSV_PATH:?set CSV_PATH to the TCGA-BRCA Oncotype DX label table}"
: "${EXTERNAL_CSV:?set EXTERNAL_CSV to a table with slide_id, label and cohort of the external slides}"
: "${EXTERNAL_FEATURE_DIR:?set EXTERNAL_FEATURE_DIR to the UNI2 .h5 features of the external slides}"
RESULTS_DIR="${RESULTS_DIR:-results/brca_odx_external}"

python train.py --config configs/brca_odx_external.json \
    --feature_dir "$FEATURE_DIR" --pathway_dir "$PATHWAY_DIR" --csv_path "$CSV_PATH" --results_dir "$RESULTS_DIR"

python evaluate.py --config configs/brca_odx_external.json \
    --feature_dir "$FEATURE_DIR" --pathway_dir "$PATHWAY_DIR" --csv_path "$CSV_PATH" --results_dir "$RESULTS_DIR"

python infer_external.py --csv_path "$EXTERNAL_CSV" --feature_dir "$EXTERNAL_FEATURE_DIR" \
    --ckpt_dir "$RESULTS_DIR" --out_dir "${RESULTS_DIR}/external"
