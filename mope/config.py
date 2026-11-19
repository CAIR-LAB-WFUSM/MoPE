import argparse
import json

DEFAULTS = {
    "task_type": "binary",
    "csv_path": None,
    "feature_dir": None,
    "pathway_dir": None,
    "split_dir": None,
    "results_dir": "results",
    "label_col": "label",
    "positive_label": "1",
    "time_col": "survival_months",
    "censorship_col": "censorship",
    "case_id_col": "case_id",
    "slide_id_col": "slide_id",
    "num_time_bins": 4,
    "folds": 5,
    "seed": 4,
    "max_epochs": 100,
    "patience": 10,
    "stop_epoch": 5,
    "lr": 2e-4,
    "weight_decay": 1e-4,
    "num_workers": 8,
    "wsi_input_dim": 1536,
    "token_dim": 64,
    "rank_dim": 64,
    "num_memory": 24,
    "num_slots": 16,
    "student_temperature": 0.1,
    "teacher_temperature": 0.2,
    "task_weight": 1.0,
    "kd_weight": 1.0,
    "align_weight": 0.0,
    "memory_weight": 0.05,
    "recon_weight": 0.01,
    "diversity_weight": 0.1,
}

REQUIRED = ("csv_path", "feature_dir", "pathway_dir", "split_dir")


def _parse_value(text: str, reference):
    if isinstance(reference, bool):
        return text.lower() in ("1", "true", "yes")
    if isinstance(reference, int):
        return int(text)
    if isinstance(reference, float):
        return float(text)
    if reference is None or isinstance(reference, dict):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text
    return text


def load_config(description: str, extra_required=REQUIRED, extra_args=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", help="JSON file with task settings (see configs/).")
    for key, value in DEFAULTS.items():
        parser.add_argument(f"--{key}", default=None, help=f"default: {value}")
    for name, kwargs in (extra_args or {}).items():
        parser.add_argument(f"--{name}", **kwargs)
    args = parser.parse_args()

    cfg = dict(DEFAULTS)
    if args.config:
        with open(args.config) as handle:
            loaded = json.load(handle)
        unknown = set(loaded) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        cfg.update(loaded)
    for key in DEFAULTS:
        value = getattr(args, key)
        if value is not None:
            cfg[key] = _parse_value(value, DEFAULTS[key] if DEFAULTS[key] is not None else cfg[key])
    for name in extra_args or {}:
        cfg[name] = getattr(args, name)

    missing = [key for key in extra_required if not cfg.get(key)]
    if missing:
        parser.error("missing required settings: " + ", ".join(f"--{key}" for key in missing))
    if cfg["task_type"] not in ("binary", "survival"):
        parser.error("task_type must be 'binary' or 'survival'")
    return argparse.Namespace(**cfg)
