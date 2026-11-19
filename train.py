import json
import os

import numpy as np
import pandas as pd
import torch

from mope.config import load_config
from mope.data import ClassificationDataset, SurvivalDataset
from mope.engine import seed_everything, train_fold


def build_dataset(cfg):
    if cfg.task_type == "survival":
        return SurvivalDataset.from_csv(
            cfg.csv_path,
            cfg.feature_dir,
            cfg.pathway_dir,
            time_col=cfg.time_col,
            censorship_col=cfg.censorship_col,
            case_id_col=cfg.case_id_col,
            slide_id_col=cfg.slide_id_col,
            n_bins=cfg.num_time_bins,
        )
    return ClassificationDataset.from_csv(
        cfg.csv_path, cfg.feature_dir, cfg.pathway_dir, cfg.label_col, cfg.positive_label
    )


def main():
    cfg = load_config("Train MoPE with five-fold cross-validation.")
    os.makedirs(cfg.results_dir, exist_ok=True)
    with open(os.path.join(cfg.results_dir, "config.json"), "w") as handle:
        json.dump(vars(cfg), handle, indent=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seed_everything(cfg.seed)
    dataset = build_dataset(cfg)
    scores = []
    for fold in range(cfg.folds):
        seed_everything(cfg.seed)
        train_split, val_split = dataset.return_splits(os.path.join(cfg.split_dir, f"splits_{fold}.csv"))
        print(f"Fold {fold}: train {len(train_split)}, val {len(val_split)}")
        scores.append(train_fold(train_split, val_split, fold, cfg, device))

    metric = "c_index" if cfg.task_type == "survival" else "auc"
    summary = pd.DataFrame({"fold": range(len(scores)), f"val_{metric}": scores})
    summary.to_csv(os.path.join(cfg.results_dir, f"summary_val_{metric}.csv"), index=False)
    print(f"Validation {metric} by fold: {np.round(scores, 4).tolist()}")
    print(f"Mean {metric}: {np.mean(scores):.4f} +/- {np.std(scores, ddof=1):.4f}")


if __name__ == "__main__":
    main()
