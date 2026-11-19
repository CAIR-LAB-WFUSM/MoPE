import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from lifelines.utils import concordance_index
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, WeightedRandomSampler
from tqdm import tqdm

from mope.data import collate_survival
from mope.model import MoPE


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def build_model(cfg, num_pathways: int, num_genes: int, n_classes: int, num_time_bins: int) -> MoPE:
    return MoPE(
        num_pathways=num_pathways,
        num_genes=num_genes,
        wsi_input_dim=cfg.wsi_input_dim,
        token_dim=cfg.token_dim,
        rank_dim=cfg.rank_dim,
        num_memory=cfg.num_memory,
        num_slots=cfg.num_slots,
        task_type=cfg.task_type,
        n_classes=n_classes,
        num_time_bins=num_time_bins,
        student_temperature=cfg.student_temperature,
        teacher_temperature=cfg.teacher_temperature,
    )


def nll_survival_loss(hazards, survival, labels, censorship, eps=1e-7):
    batch_size = len(labels)
    labels = labels.view(batch_size, 1)
    censorship = censorship.view(batch_size, 1).float()
    survival_padded = torch.cat([torch.ones_like(censorship), survival], 1)
    uncensored = -(1 - censorship) * (
        torch.log(torch.gather(survival_padded, 1, labels).clamp(min=eps))
        + torch.log(torch.gather(hazards, 1, labels).clamp(min=eps))
    )
    censored = -censorship * torch.log(torch.gather(survival_padded, 1, labels + 1).clamp(min=eps))
    return (censored + uncensored).mean()


def survival_risk(survival: torch.Tensor) -> float:
    return float((-survival[:, -1].clamp_min(1e-7).log()).detach().cpu().item())


def safe_auc(labels, probs) -> float:
    labels = np.asarray(labels)
    probs = np.asarray(probs)
    if len(np.unique(labels)) < 2:
        return 0.5
    if probs.ndim == 2:
        probs = probs[:, 1]
    return float(roc_auc_score(labels, probs))


def safe_cindex(event_times, risks, censorship) -> float:
    if len(event_times) < 2:
        return 0.5
    try:
        return float(concordance_index(event_times, -np.asarray(risks), event_observed=1 - np.asarray(censorship)))
    except ZeroDivisionError:
        return 0.5


def class_balanced_weights(dataset) -> torch.DoubleTensor:
    n = float(len(dataset))
    per_class = [n / len(ids) if len(ids) > 0 else 1e-8 for ids in dataset.slide_cls_ids]
    return torch.DoubleTensor([per_class[dataset.getlabel(i)] for i in range(len(dataset))])


def make_loaders(train_split, val_split, task_type: str, num_workers: int):
    if task_type == "survival":
        kwargs = {"num_workers": 4} if torch.cuda.is_available() else {}
        train_loader = DataLoader(
            train_split, batch_size=1, sampler=RandomSampler(train_split), collate_fn=collate_survival, **kwargs
        )
        val_loader = DataLoader(
            val_split, batch_size=1, sampler=SequentialSampler(val_split), collate_fn=collate_survival, **kwargs
        )
        return train_loader, val_loader
    sampler = WeightedRandomSampler(class_balanced_weights(train_split), num_samples=len(train_split), replacement=True)
    train_loader = DataLoader(
        train_split, batch_size=1, sampler=sampler, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(val_split, batch_size=1, shuffle=False, num_workers=4)
    return train_loader, val_loader


def _kmeans(features, num_centers: int, iters: int = 20) -> torch.Tensor:
    x = F.normalize(torch.cat(features, dim=0), dim=-1)
    n = x.size(0)
    if n == 0:
        raise ValueError("Cannot initialize the memory bank with zero features.")
    indices = torch.randperm(n)[:num_centers] if n >= num_centers else torch.randint(0, n, (num_centers,))
    centers = x[indices].clone()
    for _ in range(iters):
        assignment = (x @ centers.T).argmax(dim=1)
        for k in range(num_centers):
            members = assignment == k
            if members.any():
                centers[k] = F.normalize(x[members].mean(0, keepdim=True), dim=-1)
    return centers


@torch.no_grad()
def initialize_memory(loader, model: MoPE, device, max_features: int = 50000, iters: int = 20) -> None:
    model.eval()
    features = []
    for batch in loader:
        if torch.is_tensor(batch[2]) and torch.any(batch[2] < 0).item():
            continue
        tokens = model.expert_tokens(model.prepare_wsi(batch[0].to(device)))
        features.append(F.normalize(tokens, p=2, dim=-1, eps=1e-6).reshape(-1, tokens.size(-1)).cpu())
        if sum(f.size(0) for f in features) >= max_features:
            break
    if not features:
        return
    centers = _kmeans(features, model.num_memory, iters=iters)
    model.memory.codebook.copy_(F.normalize(centers, dim=-1).to(model.memory.codebook.device))


def train_one_epoch(epoch, model, loader, optimizer, cfg, device):
    model.train()
    ce = nn.CrossEntropyLoss()
    totals = dict(task=0.0, align=0.0, kd=0.0, mem=0.0, recon=0.0, div=0.0)
    probs, labels, risks, times, censor = [], [], [], [], []
    steps = 0
    for batch in tqdm(loader, desc=f"Epoch {epoch} [train]"):
        if torch.is_tensor(batch[2]) and torch.any(batch[2] < 0).item():
            continue
        features, rna, label = batch[0].to(device), batch[1].to(device), batch[2].to(device)
        out = model.forward_train(rna, features)
        if cfg.task_type == "survival":
            censorship = batch[4].to(device)
            loss_task = nll_survival_loss(out["hazards"], out["survival"], label, censorship)
            risks.append(survival_risk(out["survival"]))
            times.append(float(np.asarray(batch[3])[0]))
            censor.append(float(censorship.item()))
        else:
            loss_task = ce(out["logits"], label)
            probs.append(torch.softmax(out["logits"].detach(), dim=-1).cpu().numpy()[0])
            labels.append(label.item())

        loss = (
            cfg.task_weight * loss_task
            + cfg.kd_weight * out["distillation_loss"]
            + cfg.align_weight * out["alignment_loss"]
            + cfg.memory_weight * out["memory_loss"]
            + cfg.recon_weight * out["reconstruction_loss"]
            + cfg.diversity_weight * out["diversity_loss"]
        )
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        steps += 1
        for key, value in (
            ("task", loss_task),
            ("align", out["alignment_loss"]),
            ("kd", out["distillation_loss"]),
            ("mem", out["memory_loss"]),
            ("recon", out["reconstruction_loss"]),
            ("div", out["diversity_loss"]),
        ):
            totals[key] += float(value.detach().item())

    summary = {key: value / max(steps, 1) for key, value in totals.items()}
    if cfg.task_type == "survival":
        summary["metric"] = safe_cindex(times, risks, censor)
    else:
        summary["metric"] = safe_auc(labels, probs)
    return summary


@torch.no_grad()
def validate(model, loader, task_type, device):
    model.eval()
    probs, labels, risks, times, censor = [], [], [], [], []
    for batch in tqdm(loader, desc="[val]"):
        if torch.is_tensor(batch[2]) and torch.any(batch[2] < 0).item():
            continue
        out = model.forward_inference(batch[0].to(device))
        if task_type == "survival":
            risks.append(survival_risk(out["survival"]))
            times.append(float(np.asarray(batch[3])[0]))
            censor.append(float(batch[4].item()))
        else:
            probs.append(torch.softmax(out["logits"], dim=-1).cpu().numpy()[0])
            labels.append(batch[2].item())
    if task_type == "survival":
        return safe_cindex(times, risks, censor)
    return safe_auc(labels, probs)


class EarlyStopping:
    def __init__(self, patience: int, stop_epoch: int):
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, epoch, score, model, ckpt_path):
        if self.best_score is not None and score < self.best_score:
            self.counter += 1
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
            return
        self.best_score = score
        self.counter = 0
        torch.save(model.state_dict(), ckpt_path)


def train_fold(train_split, val_split, fold, cfg, device):
    rna_shape = train_split.rna_shape
    num_time_bins = getattr(train_split, "num_time_bins", cfg.num_time_bins)
    train_loader, val_loader = make_loaders(train_split, val_split, cfg.task_type, cfg.num_workers)

    model = build_model(cfg, rna_shape[0], rna_shape[1], 2, num_time_bins).to(device)
    initialize_memory(train_loader, model, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.2, patience=5)
    stopper = EarlyStopping(cfg.patience, cfg.stop_epoch)
    ckpt_path = os.path.join(cfg.results_dir, f"s_{fold}_checkpoint.pt")
    metric_name = "C-index" if cfg.task_type == "survival" else "AUC"

    for epoch in range(cfg.max_epochs):
        stats = train_one_epoch(epoch, model, train_loader, optimizer, cfg, device)
        val_metric = validate(model, val_loader, cfg.task_type, device)
        print(
            f"Fold {fold} epoch {epoch}: task {stats['task']:.4f}, kd {stats['kd']:.4f}, align {stats['align']:.4f}, "
            f"mem {stats['mem']:.4f}, recon {stats['recon']:.4f}, div {stats['div']:.4f}, "
            f"train {metric_name} {stats['metric']:.4f}, val {metric_name} {val_metric:.4f}"
        )
        scheduler.step(val_metric)
        stopper(epoch, val_metric, model, ckpt_path)
        if stopper.early_stop:
            print("Early stopping.")
            break
    print(f"Fold {fold} best validation {metric_name}: {stopper.best_score:.4f}")
    return stopper.best_score
