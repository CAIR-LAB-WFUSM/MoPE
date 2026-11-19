import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from mope.data import SlideDataset, collate_survival, normalize_case_id, normalize_slide_id
from mope.engine import build_model, safe_cindex, survival_risk


def load_fold_model(cfg, ckpt_path, num_pathways, num_genes, n_classes, num_time_bins, device):
    model = build_model(cfg, num_pathways, num_genes, n_classes, num_time_bins)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    return model.to(device).eval()


def select_slides(slide_table: pd.DataFrame, slide_ids) -> pd.DataFrame:
    keys = {normalize_slide_id(x) for x in pd.Series(slide_ids).dropna().astype(str) if str(x).strip()}
    table = slide_table.copy()
    table["slide_key"] = table["slide_id"].astype(str).map(normalize_slide_id)
    selected = table[table["slide_key"].isin(keys)]
    if selected.empty and keys:
        patients = {normalize_case_id(x) for x in keys}
        selected = table[table["slide_id"].astype(str).map(normalize_case_id).isin(patients)]
    return selected


@torch.no_grad()
def predict_slides(model, slide_table: pd.DataFrame, feature_dir: str, device, extra_cols=()) -> pd.DataFrame:
    dataset = SlideDataset(slide_table, feature_dir)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    rows = []
    for features, label, slide_id in loader:
        logits = model.forward_inference(features.to(device))["logits"]
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        row = {"slide_id": slide_id[0], "label": int(label.item()), "Y_hat_argmax": int(probs.argmax())}
        row.update({f"p_{c}": float(p) for c, p in enumerate(probs)})
        rows.append(row)
    preds = pd.DataFrame(rows)
    for col in extra_cols:
        if col in slide_table.columns and not preds.empty:
            preds[col] = preds["slide_id"].map(dict(zip(slide_table["slide_id"].astype(str), slide_table[col])))
    return preds


@torch.no_grad()
def predict_survival(model, split, device):
    loader = DataLoader(
        split, batch_size=1, shuffle=False, num_workers=4 if device.type == "cuda" else 0, collate_fn=collate_survival
    )
    rows = []
    for idx, (features, _, label, event_time, censorship) in enumerate(loader):
        out = model.forward_inference(features.to(device))
        rows.append(
            {
                "case_id": split.get_case_id(idx),
                "label_bin": int(label.item()),
                "event_time": float(np.asarray(event_time)[0]),
                "censorship": float(censorship.item()),
                "risk_score": survival_risk(out["survival"]),
                "hazards": out["hazards"].squeeze(0).cpu().numpy().tolist(),
                "survival": out["survival"].squeeze(0).cpu().numpy().tolist(),
            }
        )
    preds = pd.DataFrame(rows)
    if preds.empty:
        return preds, 0.5
    return preds, safe_cindex(preds["event_time"].tolist(), preds["risk_score"].tolist(), preds["censorship"].tolist())


def checkpoint_path(ckpt_dir: str, fold: int) -> str:
    return os.path.join(ckpt_dir, f"s_{fold}_checkpoint.pt")
