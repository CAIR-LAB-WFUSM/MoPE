import os

import numpy as np
import pandas as pd
import torch

from mope.config import load_config
from mope.data import SurvivalDataset, index_pathway_files, pathway_shape
from mope.engine import safe_auc
from mope.inference import checkpoint_path, load_fold_model, predict_slides, predict_survival, select_slides


def label_table(cfg) -> pd.DataFrame:
    table = pd.read_csv(cfg.csv_path, dtype={"slide_id": str})
    positive = str(cfg.positive_label).lower()
    table["label"] = table[cfg.label_col].astype(str).str.lower().eq(positive).astype(int)
    return table


def evaluate_classification(cfg, device):
    files = index_pathway_files(cfg.pathway_dir)
    num_pathways, num_genes = pathway_shape(cfg.pathway_dir, next(iter(files.values())))
    table = label_table(cfg)
    rows = []
    for fold in range(cfg.folds):
        ckpt = checkpoint_path(cfg.results_dir, fold)
        split_csv = os.path.join(cfg.split_dir, f"splits_{fold}.csv")
        if not os.path.exists(ckpt) or not os.path.exists(split_csv):
            print(f"Fold {fold}: missing checkpoint or split file, skipped.")
            continue
        splits = pd.read_csv(split_csv)
        model = load_fold_model(cfg, ckpt, num_pathways, num_genes, 2, cfg.num_time_bins, device)
        result = {"fold": fold}
        for name in ("val", "test"):
            if name not in splits.columns:
                continue
            preds = predict_slides(model, select_slides(table, splits[name]), cfg.feature_dir, device)
            preds.to_csv(os.path.join(cfg.results_dir, f"{name}_preds_fold_{fold}.csv"), index=False)
            result[f"{name}_auc"] = safe_auc(preds["label"], preds["p_1"].values)
        rows.append(result)
        print(result)
    return pd.DataFrame(rows)


def evaluate_survival(cfg, device):
    dataset = SurvivalDataset.from_csv(
        cfg.csv_path,
        cfg.feature_dir,
        cfg.pathway_dir,
        time_col=cfg.time_col,
        censorship_col=cfg.censorship_col,
        case_id_col=cfg.case_id_col,
        slide_id_col=cfg.slide_id_col,
        n_bins=cfg.num_time_bins,
    )
    rows = []
    for fold in range(cfg.folds):
        ckpt = checkpoint_path(cfg.results_dir, fold)
        if not os.path.exists(ckpt):
            print(f"Fold {fold}: missing checkpoint, skipped.")
            continue
        train_split, val_split = dataset.return_splits(os.path.join(cfg.split_dir, f"splits_{fold}.csv"))
        num_pathways, num_genes = train_split.rna_shape
        model = load_fold_model(cfg, ckpt, num_pathways, num_genes, 2, train_split.num_time_bins, device)
        preds, c_index = predict_survival(model, val_split, device)
        preds.to_csv(os.path.join(cfg.results_dir, f"val_survival_preds_fold_{fold}.csv"), index=False)
        rows.append({"fold": fold, "val_c_index": c_index, "n": len(preds)})
        print(rows[-1])
    return pd.DataFrame(rows)


def main():
    cfg = load_config("Evaluate trained MoPE fold checkpoints.")
    os.makedirs(cfg.results_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    summary = evaluate_survival(cfg, device) if cfg.task_type == "survival" else evaluate_classification(cfg, device)
    if summary.empty:
        print("No folds evaluated.")
        return
    summary.to_csv(os.path.join(cfg.results_dir, "summary_metrics.csv"), index=False)
    for col in summary.columns.drop("fold"):
        values = summary[col].astype(float)
        print(f"{col}: {np.mean(values):.4f} +/- {np.std(values, ddof=1):.4f}")


if __name__ == "__main__":
    main()
