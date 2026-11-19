import argparse
import os

import pandas as pd
import torch

from mope.config import DEFAULTS
from mope.engine import safe_auc
from mope.inference import checkpoint_path, load_fold_model, predict_slides


def parse_args():
    parser = argparse.ArgumentParser(description="H&E-only inference of trained MoPE models on an external cohort.")
    parser.add_argument(
        "--csv_path", required=True, help="CSV with slide_id and optional label (0/1) and cohort columns."
    )
    parser.add_argument("--feature_dir", required=True, help="Directory with UNI2 .h5 features of the external slides.")
    parser.add_argument("--ckpt_dir", required=True, help="Directory with the fold checkpoints.")
    parser.add_argument("--out_dir", default="results/external")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--num_pathways", type=int, default=50)
    parser.add_argument("--num_genes", type=int, default=4188)
    for key in ("wsi_input_dim", "token_dim", "rank_dim", "num_memory", "num_slots"):
        parser.add_argument(f"--{key}", type=int, default=DEFAULTS[key])
    args = parser.parse_args()
    args.task_type = "binary"
    args.student_temperature = DEFAULTS["student_temperature"]
    args.teacher_temperature = DEFAULTS["teacher_temperature"]
    return args


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    table = pd.read_csv(args.csv_path, dtype={"slide_id": str}).drop_duplicates("slide_id")
    has_label = "label" in table.columns
    table["label"] = table["label"].fillna(-1).astype(int) if has_label else -1

    for fold in range(args.folds):
        ckpt = checkpoint_path(args.ckpt_dir, fold)
        if not os.path.exists(ckpt):
            print(f"Fold {fold}: missing checkpoint, skipped.")
            continue
        model = load_fold_model(args, ckpt, args.num_pathways, args.num_genes, 2, 4, device)
        preds = predict_slides(model, table, args.feature_dir, device, extra_cols=("cohort",))
        preds.to_csv(os.path.join(args.out_dir, f"external_preds_fold_{fold}.csv"), index=False)
        labelled = preds[preds["label"] >= 0]
        if has_label and labelled["label"].nunique() == 2:
            print(f"Fold {fold}: AUC {safe_auc(labelled['label'], labelled['p_1']):.4f} (n={len(labelled)})")
        else:
            print(f"Fold {fold}: scored {len(preds)} slides")


if __name__ == "__main__":
    main()
