import os
from typing import Dict, List, Optional, Sequence

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def normalize_case_id(case_id) -> str:
    case_id = str(case_id).strip().upper()
    if case_id.lower() == "nan":
        return ""
    if case_id.startswith("TCGA-") and len(case_id) >= 12:
        return case_id[:12]
    return case_id


def normalize_slide_id(slide_id) -> str:
    slide_id = str(slide_id).strip().upper()
    if slide_id.lower() == "nan":
        return ""
    return slide_id


def feature_path(feature_dir: str, slide_id) -> str:
    return os.path.join(feature_dir, f"{str(slide_id).strip().replace('.svs', '')}.h5")


def load_features(path: str) -> torch.Tensor:
    with h5py.File(path, "r") as handle:
        features = torch.from_numpy(handle["features"][:]).float()
    return features.unsqueeze(0) if features.dim() == 1 else features


def index_pathway_files(pathway_dir: str) -> Dict[str, str]:
    files = [name for name in os.listdir(pathway_dir) if name.endswith(".npy")]
    return {normalize_case_id(os.path.splitext(name)[0]): name for name in files}


def pathway_shape(pathway_dir: str, filename: str):
    return tuple(np.load(os.path.join(pathway_dir, filename), mmap_mode="r").shape)


class ClassificationDataset(Dataset):
    def __init__(
        self,
        feature_dir: str,
        pathway_dir: str,
        slide_data: pd.DataFrame,
        pathway_files: Dict[str, str],
    ):
        self.feature_dir = feature_dir
        self.pathway_dir = pathway_dir
        self.pathway_files = pathway_files
        slide_data = slide_data[slide_data["patient_key"].isin(pathway_files.keys())]
        exists = slide_data["slide_id"].apply(lambda s: os.path.exists(feature_path(feature_dir, s)))
        if (~exists).any():
            print(f"Skipping {int((~exists).sum())} slides without feature files.")
        self.slide_data = slide_data.loc[exists].reset_index(drop=True)
        self.slide_cls_ids = [self.slide_data.index[self.slide_data["label"] == c].tolist() for c in (0, 1)]

    @classmethod
    def from_csv(cls, csv_path, feature_dir, pathway_dir, label_col, positive_label):
        pathway_files = index_pathway_files(pathway_dir)
        slide_data = pd.read_csv(csv_path)
        slide_data["patient_key"] = slide_data["case_id"].apply(normalize_case_id)
        slide_data["slide_key"] = slide_data["slide_id"].apply(normalize_slide_id)
        positive = str(positive_label).lower()
        slide_data["label"] = slide_data[label_col].apply(lambda x: int(str(x).lower() == positive))
        return cls(feature_dir, pathway_dir, slide_data, pathway_files)

    def subset(self, slide_ids: Sequence[str]) -> "ClassificationDataset":
        keys = [normalize_slide_id(x) for x in slide_ids]
        subset = self.slide_data[self.slide_data["slide_key"].isin(keys)]
        if len(subset) == 0 and keys:
            patients = {normalize_case_id(x) for x in keys if x}
            subset = self.slide_data[self.slide_data["patient_key"].isin(patients)]
        return ClassificationDataset(
            self.feature_dir, self.pathway_dir, subset.reset_index(drop=True), self.pathway_files
        )

    def return_splits(self, split_csv: str):
        splits = pd.read_csv(split_csv)
        return tuple(
            self.subset(splits[key].dropna().unique() if key in splits.columns else []) for key in ("train", "val")
        )

    @property
    def rna_shape(self):
        if len(self.slide_data) == 0:
            return None
        return pathway_shape(self.pathway_dir, self.pathway_files[self.slide_data["patient_key"].iloc[0]])

    def getlabel(self, idx: int) -> int:
        return self.slide_data["label"].iloc[idx]

    def __len__(self):
        return len(self.slide_data)

    def __getitem__(self, idx):
        row = self.slide_data.iloc[idx]
        try:
            features = load_features(feature_path(self.feature_dir, row["slide_id"]))
        except Exception as exc:
            print(f"Failed to load features for {row['slide_id']}: {exc}")
            return torch.zeros(1, 1536), torch.zeros(self.rna_shape or (1,)), -1, "LOAD_ERROR"
        rna = torch.from_numpy(np.load(os.path.join(self.pathway_dir, self.pathway_files[row["patient_key"]]))).float()
        return features, rna, row["label"], row["slide_id"]


class SurvivalDataset(Dataset):
    def __init__(
        self,
        feature_dir: str,
        pathway_dir: str,
        patient_df: pd.DataFrame,
        patient_to_slides: Dict[str, List[str]],
        pathway_files: Dict[str, str],
        time_col: str,
        censorship_col: str,
        case_id_col: str,
        time_bins: Optional[Sequence[float]] = None,
        n_bins: int = 4,
        eps: float = 1e-6,
    ):
        self.feature_dir = feature_dir
        self.pathway_dir = pathway_dir
        self.patient_df = patient_df.reset_index(drop=True).copy()
        self.patient_to_slides = patient_to_slides
        self.pathway_files = pathway_files
        self.time_col = time_col
        self.censorship_col = censorship_col
        self.case_id_col = case_id_col
        self.n_bins = int(n_bins)
        self.eps = eps
        self.time_bins = None
        self.num_time_bins = self.n_bins
        if time_bins is not None:
            self._assign_bins(np.asarray(time_bins, dtype=np.float64))
        else:
            self.patient_df["label"] = -1

    @classmethod
    def from_csv(
        cls,
        csv_path,
        feature_dir,
        pathway_dir,
        time_col="survival_months",
        censorship_col="censorship",
        case_id_col="case_id",
        slide_id_col="slide_id",
        n_bins=4,
    ):
        pathway_files = index_pathway_files(pathway_dir)
        data = pd.read_csv(csv_path, low_memory=False)
        missing = [c for c in (case_id_col, slide_id_col, time_col, censorship_col) if c not in data.columns]
        if missing:
            raise ValueError(f"Missing required columns in survival CSV: {missing}")
        data[case_id_col] = data[case_id_col].astype(str)
        data[slide_id_col] = data[slide_id_col].astype(str)
        data["patient_key"] = data[case_id_col].apply(normalize_case_id)
        data["slide_key"] = data[slide_id_col].apply(normalize_slide_id)
        data[time_col] = pd.to_numeric(data[time_col], errors="coerce")
        data[censorship_col] = pd.to_numeric(data[censorship_col], errors="coerce")
        data = data.dropna(subset=["patient_key", "slide_key", time_col, censorship_col])
        data = data[(data["patient_key"] != "") & (data["slide_key"] != "")]
        data = data[data["patient_key"].isin(pathway_files.keys())].reset_index(drop=True)
        patient_to_slides = {k: list(v) for k, v in data.groupby("patient_key")[slide_id_col].apply(list).items()}
        patient_df = data.sort_values([case_id_col, slide_id_col]).drop_duplicates("patient_key")
        patient_df = patient_df[[case_id_col, "patient_key", censorship_col, time_col]]
        return cls(
            feature_dir,
            pathway_dir,
            patient_df,
            patient_to_slides,
            pathway_files,
            time_col,
            censorship_col,
            case_id_col,
            n_bins=n_bins,
        )

    def _fit_time_bins(self, train_df: pd.DataFrame) -> np.ndarray:
        if len(train_df) == 0:
            raise ValueError("Cannot fit survival time bins on an empty training split.")
        uncensored = train_df[train_df[self.censorship_col] < 1]
        source_times = (uncensored if len(uncensored) >= 2 else train_df)[self.time_col].astype(float)
        full_times = self.patient_df[self.time_col].astype(float)
        if source_times.nunique() <= 1:
            center = float(source_times.iloc[0])
            return np.array([center - self.eps, center + self.eps], dtype=np.float64)
        q = min(self.n_bins, int(source_times.nunique()))
        _, bins = pd.qcut(source_times, q=q, retbins=True, labels=False, duplicates="drop")
        bins = np.asarray(bins, dtype=np.float64)
        bins[0] = float(full_times.min()) - self.eps
        bins[-1] = float(full_times.max()) + self.eps
        bins = np.unique(bins)
        if len(bins) < 2:
            bins = np.array([float(full_times.min()) - self.eps, float(full_times.max()) + self.eps])
        return bins

    def _assign_bins(self, bins: np.ndarray) -> None:
        disc = pd.cut(
            self.patient_df[self.time_col].astype(float), bins=bins, labels=False, right=False, include_lowest=True
        )
        disc = pd.Series(disc).fillna(len(bins) - 2).astype(int).clip(lower=0, upper=max(len(bins) - 2, 0))
        self.patient_df["label"] = disc.values.astype(int)
        self.time_bins = bins
        self.num_time_bins = int(self.patient_df["label"].max()) + 1 if len(self.patient_df) else self.n_bins

    def _subset(self, patient_keys, time_bins) -> "SurvivalDataset":
        keys = list(dict.fromkeys(patient_keys))
        subset = self.patient_df[self.patient_df["patient_key"].isin(keys)]
        return SurvivalDataset(
            self.feature_dir,
            self.pathway_dir,
            subset,
            self.patient_to_slides,
            self.pathway_files,
            self.time_col,
            self.censorship_col,
            self.case_id_col,
            time_bins=time_bins,
            n_bins=self.n_bins,
            eps=self.eps,
        )

    def return_splits(self, split_csv: str):
        splits = pd.read_csv(split_csv)
        keys = {
            name: [normalize_case_id(v) for v in splits[name].dropna().astype(str)] if name in splits.columns else []
            for name in ("train", "val")
        }
        train_df = self.patient_df[self.patient_df["patient_key"].isin(keys["train"])]
        bins = self._fit_time_bins(train_df)
        return self._subset(keys["train"], bins), self._subset(keys["val"], bins)

    @property
    def rna_shape(self):
        if len(self.patient_df) == 0:
            return None
        return pathway_shape(self.pathway_dir, self.pathway_files[self.patient_df["patient_key"].iloc[0]])

    def get_case_id(self, idx: int) -> str:
        return str(self.patient_df[self.case_id_col].iloc[idx])

    def __len__(self):
        return len(self.patient_df)

    def _load_patient_features(self, slide_ids) -> torch.Tensor:
        features = []
        for slide_id in slide_ids:
            path = feature_path(self.feature_dir, slide_id)
            if not os.path.exists(path):
                continue
            try:
                features.append(load_features(path))
            except Exception as exc:
                print(f"Failed to load {path}: {exc}")
        return torch.cat(features, dim=0) if features else torch.zeros(1, 1536)

    def __getitem__(self, idx):
        row = self.patient_df.iloc[idx]
        features = self._load_patient_features(self.patient_to_slides.get(row["patient_key"], []))
        rna = torch.from_numpy(np.load(os.path.join(self.pathway_dir, self.pathway_files[row["patient_key"]]))).float()
        return features, rna, int(row["label"]), float(row[self.time_col]), float(row[self.censorship_col])


def collate_survival(batch):
    if len(batch) != 1:
        raise ValueError("Survival training expects batch_size=1.")
    features, rna, label, event_time, censorship = batch[0]
    return (
        features,
        rna.unsqueeze(0).float(),
        torch.LongTensor([label]),
        np.array([event_time], dtype=np.float32),
        torch.FloatTensor([censorship]),
    )


class SlideDataset(Dataset):
    def __init__(self, slide_data: pd.DataFrame, feature_dir: str):
        exists = slide_data["slide_id"].apply(lambda s: os.path.exists(feature_path(feature_dir, s)))
        if (~exists).any():
            print(f"Skipping {int((~exists).sum())} slides without feature files.")
        self.slide_data = slide_data.loc[exists].reset_index(drop=True)
        self.feature_dir = feature_dir

    def __len__(self):
        return len(self.slide_data)

    def __getitem__(self, idx):
        row = self.slide_data.iloc[idx]
        return load_features(feature_path(self.feature_dir, row["slide_id"])), int(row["label"]), str(row["slide_id"])
