"""
dh_train_immune_boost.py (drop-in)

Shared SMP encoder + shared Unet decoder + 2 heads:
  - semantic head: 5 classes (MoNuSAC nucleus typing: 0..4)
  - ternary head:  3 classes (0=bg, 1=inside, 2=boundary)

This script:
  - trains BOTH heads (semantic + ternary)  [NOTE: semantic head can be frozen]
  - prints epoch summaries with BOTH semantic + ternary metrics
  - writes metrics.csv + history.json
  - saves a single dashboard.png at the end
  - uses immune-aware WeightedRandomSampler logic
  - computes ternary class weights from TRAIN ternary masks and uses them in CE loss
  - **NEW** freeze semantic head weights; keep encoder/decoder trainable; upweight ternary boundary

Lightning-ready updates:
  - Robust PROJECT_ROOT anchoring so assets/scripts resolve regardless of CWD
  - TRAIN_MANIFEST / VAL_MANIFEST / BASE_DIR / OUT_ROOT / NUM_WORKERS / DATA_ROOT via env vars
  - Default OUT_ROOT="./outputs"
"""

import json
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

import numpy as np
import matplotlib.pyplot as plt

import sys
import os
from dataclasses import dataclass
import csv
from datetime import datetime
import random

from PIL import Image

import albumentations as A
from albumentations.pytorch import ToTensorV2


# ImageNet normalization constants (for pretrained ResNet encoder)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)
IMAGENET_MEAN_NP = np.array(IMAGENET_MEAN)
IMAGENET_STD_NP  = np.array(IMAGENET_STD)


# -----------------------------
# Project root + robust paths
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parent  # script lives at repo root

# Ensure local imports (scripts/...) work from anywhere
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def env_str(key: str, default: str) -> str:
    v = os.environ.get(key, "").strip()
    return v if v else default


def resolve_path(p: str | Path, base: Path) -> Path:
    """
    Resolve 'p' robustly:
        - absolute -> as is
        - relative -> try PROJECT_ROOT/p first (repo-relative)
        - else -> try base/p (data-root-relative)
    """
    p = Path(p)
    if p.is_absolute():
        return p
    cand1 = (PROJECT_ROOT / p)
    if cand1.exists():
        return cand1.resolve()
    return (base / p).resolve()


DEFAULT_DATA_ROOT = Path(env_str("DATA_ROOT", "/teamspace/studios/this_studio/data/monusac")).resolve()
if not DEFAULT_DATA_ROOT.exists():
    DEFAULT_DATA_ROOT = (PROJECT_ROOT / "data" / "monusac").resolve()


from scripts.export_manifest_dataset import ExportManifestDataset, ExportManifestConfig
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.base import SegmentationHead


# -----------------------------
# Globals / constants
# -----------------------------
BASE_SEED = 7

SEMANTIC_CLASS_NAMES = {
    0: "background",
    1: "epithelial",
    2: "lymphocyte",
    3: "neutrophil",
    4: "macrophage",
}
SEM_CLASSES = (0, 1, 2, 3, 4)

TER_CLASS_NAMES = {
    0: "bg",
    1: "inside",
    2: "boundary",
}
TER_CLASSES = (0, 1, 2)

# Base for repo-relative asset resolution
_ASSET_BASE = PROJECT_ROOT

SEM_WEIGHTS_JSON = resolve_path("assets/class_weights.json", base=_ASSET_BASE)
with open(SEM_WEIGHTS_JSON, "r", encoding="utf-8") as f:
    wobj = json.load(f)


# -----------------------------
# Device + seeds
# -----------------------------
def get_device(force: str | None = None) -> torch.device:
    if force is not None:
        return torch.device(force)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# -----------------------------
# JSON helpers
# -----------------------------
def to_jsonable(x: Any) -> Any:
    if x is None or isinstance(x, (bool, int, float, str)):
        return x
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, dict):
        return {str(k): to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    if isinstance(x, set):
        return [to_jsonable(v) for v in sorted(x)]
    if torch.is_tensor(x):
        return {"__tensor__": True, "shape": list(x.shape), "dtype": str(x.dtype)}
    return str(x)


def write_json(path: Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(obj), f, indent=2, sort_keys=True)


def update_alias(alias_path: Path, target_path: Path):
    """Symlink if possible; else copy."""
    alias_path = Path(alias_path)
    target_path = Path(target_path)
    try:
        if alias_path.exists() or alias_path.is_symlink():
            alias_path.unlink()
        alias_path.symlink_to(target_path.name)  # relative within same dir
    except OSError:
        import shutil
        shutil.copy2(target_path, alias_path)


# -----------------------------
# Checkpoint (.pt + sidecar .json)
# -----------------------------
def save_checkpoint(
    ckpt_path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scheduler: Any | None,
    epoch: int,
    global_step: int,
    config: dict,
    extra_pt: dict | None = None,
    extra_json: dict | None = None,
) -> None:
    ckpt_path = Path(ckpt_path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    json_path = ckpt_path.with_suffix(".json")
    write_json(json_path, {"config": config, "extra_json": extra_json or {}})

    ckpt = {
        "model_state": model.state_dict(),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "config_path": str(json_path),
    }
    if optimizer is not None:
        ckpt["optimizer_state"] = optimizer.state_dict()
    if scheduler is not None:
        ckpt["scheduler_state"] = scheduler.state_dict()
    if extra_pt:
        ckpt.update(extra_pt)

    torch.save(ckpt, ckpt_path)


# -----------------------------
# Metrics helpers
# -----------------------------
def _bin_stats(pred: torch.Tensor, gt: torch.Tensor):
    inter = (pred & gt).sum().float()
    ps = pred.sum().float()
    gs = gt.sum().float()
    return inter, ps, gs


def _dice_from_stats(inter, ps, gs, eps=1e-6):
    return (2 * inter + eps) / (ps + gs + eps)


@torch.no_grad()
def semantic_dice_per_class(
    sem_logits: torch.Tensor,   # [B,C,H,W]
    sem_gt: torch.Tensor,       # [B,H,W]
    classes=SEM_CLASSES,
    fg_only: bool = False,
    eps: float = 1e-6,
):
    pred = torch.argmax(sem_logits, dim=1)  # [B,H,W]
    region = (sem_gt > 0) if fg_only else torch.ones_like(sem_gt, dtype=torch.bool)

    dice_macro, dice_micro, class_counts = {}, {}, {}

    for c in classes:
        per_img = []
        for b in range(pred.shape[0]):
            gt_c = (sem_gt[b] == c) & region[b]
            if gt_c.sum() == 0:
                continue
            pred_c = (pred[b] == c) & region[b]
            inter, ps, gs = _bin_stats(pred_c, gt_c)
            per_img.append(_dice_from_stats(inter, ps, gs, eps))
        dice_macro[c] = float(torch.stack(per_img).mean().item()) if len(per_img) > 0 else 0.0

        gt_c_all = (sem_gt == c) & region
        pred_c_all = (pred == c) & region
        inter, ps, gs = _bin_stats(pred_c_all, gt_c_all)
        dice_micro[c] = float(_dice_from_stats(inter, ps, gs, eps).item())

        class_counts[c] = {
            "gt_pixels": int(gt_c_all.sum().item()),
            "pred_pixels": int(pred_c_all.sum().item()),
        }

    return dice_macro, dice_micro, class_counts


@torch.no_grad()
def semantic_iou_all(
    sem_logits: torch.Tensor,  # [B,C,H,W]
    sem_gt: torch.Tensor,      # [B,H,W]
    classes=SEM_CLASSES,
    eps: float = 1e-6,
):
    pred = torch.argmax(sem_logits, dim=1)  # [B,H,W]
    B = pred.shape[0]

    per_img_macro = []
    inter_c = {c: torch.zeros((), device=pred.device) for c in classes}
    union_c = {c: torch.zeros((), device=pred.device) for c in classes}

    for b in range(B):
        ious = []
        for c in classes:
            gt_c = (sem_gt[b] == c)
            if gt_c.sum() == 0:
                continue
            pred_c = (pred[b] == c)
            inter = (pred_c & gt_c).sum().float()
            union = (pred_c | gt_c).sum().float()
            ious.append((inter + eps) / (union + eps))
        per_img_macro.append(
            torch.stack(ious).mean() if len(ious) > 0 else torch.tensor(0.0, device=pred.device)
        )

        for c in classes:
            gt_c = (sem_gt[b] == c)
            pred_c = (pred[b] == c)
            inter_c[c] += (pred_c & gt_c).sum().float()
            union_c[c] += (pred_c | gt_c).sum().float()

    miou_macro_per_img = torch.stack(per_img_macro)  # [B]

    total_inter = torch.zeros((), device=pred.device)
    total_union = torch.zeros((), device=pred.device)
    iou_by_class_micro = {}
    for c in classes:
        total_inter += inter_c[c]
        total_union += union_c[c]
        iou_by_class_micro[c] = float(((inter_c[c] + eps) / (union_c[c] + eps)).item())

    miou_micro = float(((total_inter + eps) / (total_union + eps)).item())
    return miou_macro_per_img, miou_micro, iou_by_class_micro


@torch.no_grad()
def ternary_dice_per_class(
    ter_logits: torch.Tensor,  # [B,3,H,W]
    ter_gt: torch.Tensor,      # [B,H,W]
    classes=TER_CLASSES,
    eps: float = 1e-6,
):
    pred = torch.argmax(ter_logits, dim=1)  # [B,H,W]

    dice_macro, dice_micro, class_counts = {}, {}, {}
    for c in classes:
        per_img = []
        for b in range(pred.shape[0]):
            gt_c = (ter_gt[b] == c)
            if gt_c.sum() == 0:
                continue
            pred_c = (pred[b] == c)
            inter, ps, gs = _bin_stats(pred_c, gt_c)
            per_img.append(_dice_from_stats(inter, ps, gs, eps))
        dice_macro[c] = float(torch.stack(per_img).mean().item()) if len(per_img) > 0 else 0.0

        gt_c_all = (ter_gt == c)
        pred_c_all = (pred == c)
        inter, ps, gs = _bin_stats(pred_c_all, gt_c_all)
        dice_micro[c] = float(_dice_from_stats(inter, ps, gs, eps).item())

        class_counts[c] = {
            "gt_pixels": int(gt_c_all.sum().item()),
            "pred_pixels": int(pred_c_all.sum().item()),
        }

    return dice_macro, dice_micro, class_counts


@torch.no_grad()
def ternary_iou_all(
    ter_logits: torch.Tensor,  # [B,3,H,W]
    ter_gt: torch.Tensor,      # [B,H,W]
    classes=TER_CLASSES,
    eps: float = 1e-6,
):
    pred = torch.argmax(ter_logits, dim=1)  # [B,H,W]
    B = pred.shape[0]

    per_img_macro = []
    inter_c = {c: torch.zeros((), device=pred.device) for c in classes}
    union_c = {c: torch.zeros((), device=pred.device) for c in classes}

    for b in range(B):
        ious = []
        for c in classes:
            gt_c = (ter_gt[b] == c)
            if gt_c.sum() == 0:
                continue
            pred_c = (pred[b] == c)
            inter = (pred_c & gt_c).sum().float()
            union = (pred_c | gt_c).sum().float()
            ious.append((inter + eps) / (union + eps))
        per_img_macro.append(
            torch.stack(ious).mean() if len(ious) > 0 else torch.tensor(0.0, device=pred.device)
        )

        for c in classes:
            gt_c = (ter_gt[b] == c)
            pred_c = (pred[b] == c)
            inter_c[c] += (pred_c & gt_c).sum().float()
            union_c[c] += (pred_c | gt_c).sum().float()

    miou_macro_per_img = torch.stack(per_img_macro)  # [B]

    total_inter = torch.zeros((), device=pred.device)
    total_union = torch.zeros((), device=pred.device)
    iou_by_class_micro = {}
    for c in classes:
        total_inter += inter_c[c]
        total_union += union_c[c]
        iou_by_class_micro[c] = float(((inter_c[c] + eps) / (union_c[c] + eps)).item())

    miou_micro = float(((total_inter + eps) / (total_union + eps)).item())
    return miou_macro_per_img, miou_micro, iou_by_class_micro

def unpack_batch(batch):
    """
    Supports datasets that return:
      (x, sem, ter) OR (x, sem, ter, *extras)
    """
    if not isinstance(batch, (list, tuple)):
        raise TypeError(f"Expected batch tuple/list, got {type(batch)}")
    if len(batch) < 3:
        raise ValueError(f"Expected at least 3 items (x, sem, ter), got len={len(batch)}")
    x, sem, ter = batch[0], batch[1], batch[2]
    extras = batch[3:] if len(batch) > 3 else ()
    return x, sem, ter, extras


# -----------------------------
# Losses
# -----------------------------
def multiclass_soft_dice_loss(
    logits: torch.Tensor,          # [B,C,H,W]
    target: torch.Tensor,          # [B,H,W]
    num_classes: int,
    ignore_index: int | None = None,
    class_weights: torch.Tensor | None = None,  # [C]
    eps: float = 1e-6,
) -> torch.Tensor:
    probs = F.softmax(logits, dim=1)  # [B,C,H,W]

    if ignore_index is not None:
        valid = (target != ignore_index)
    else:
        valid = torch.ones_like(target, dtype=torch.bool)

    target_clamped = target.clone()
    target_clamped[~valid] = 0
    onehot = F.one_hot(target_clamped, num_classes=num_classes).permute(0, 3, 1, 2).float()

    valid_f = valid.unsqueeze(1).float()
    probs = probs * valid_f
    onehot = onehot * valid_f

    dims = (0, 2, 3)
    inter = (probs * onehot).sum(dims)
    den = (probs + onehot).sum(dims)
    dice = (2.0 * inter + eps) / (den + eps)  # [C]
    loss_per_class = 1.0 - dice

    if class_weights is not None:
        w = class_weights.to(loss_per_class.device).float()
        w = w / (w.sum() + eps)
        return (loss_per_class * w).sum()
    return loss_per_class.mean()


def semantic_loss_dice_only(
    logits: torch.Tensor,
    target: torch.Tensor,
    dice_w: torch.Tensor | None = None,
    ignore_index: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    loss_dice = multiclass_soft_dice_loss(
        logits,
        target,
        num_classes=logits.shape[1],
        ignore_index=ignore_index,
        class_weights=dice_w,
    )
    return loss_dice, loss_dice


def focal_cross_entropy(
    logits: torch.Tensor,          # [B, C, H, W]
    target: torch.Tensor,          # [B, H, W]  (class indices)
    weight: torch.Tensor | None = None,  # [C] per-class alpha weights
    gamma: float = 2.0,
    reduction: str = "mean",
    ignore_index: int = -100,
) -> torch.Tensor:
    """
    Focal loss: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Drop-in replacement for F.cross_entropy with an extra `gamma` parameter.
    When gamma=0 this is numerically identical to F.cross_entropy.
    """
    # standard CE per pixel (unreduced)
    ce = F.cross_entropy(
        logits, target,
        weight=weight,
        reduction="none",
        ignore_index=ignore_index,
    )  # [B, H, W]

    if gamma > 0:
        # p_t = probability of the true class
        log_p = F.log_softmax(logits, dim=1)                      # [B, C, H, W]
        p_t = torch.gather(
            log_p, dim=1,
            index=target.unsqueeze(1).clamp(min=0),               # clamp for ignore_index safety
        ).squeeze(1).exp()                                         # [B, H, W]

        # focal modulator: (1 - p_t)^gamma
        focal_weight = (1.0 - p_t).pow(gamma)

        # zero out ignored positions so they don't contribute
        if ignore_index >= 0:
            valid = (target != ignore_index)
            focal_weight = focal_weight * valid.float()

        ce = focal_weight * ce

    if reduction == "mean":
        if ignore_index >= 0:
            valid = (target != ignore_index)
            return ce.sum() / valid.sum().clamp(min=1)
        return ce.mean()
    elif reduction == "sum":
        return ce.sum()
    return ce


def ternary_combined_loss(
    ter_logits: torch.Tensor,       # [B, 3, H, W]
    ter_target: torch.Tensor,       # [B, H, W]
    focal_ce_w: torch.Tensor | None = None,   # [3] per-class alpha for focal CE
    dice_w: torch.Tensor | None = None,        # [3] per-class weights for Dice
    gamma: float = 2.0,
    dice_weight: float = 0.5,       # lambda for Dice term: total = focal_CE + dice_weight * Dice
    ignore_index: int = -100,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Combined ternary loss: focal CE + weighted soft Dice.

    Returns (total_loss, focal_ce_component, dice_component) so both can be logged.
    """
    loss_fce = focal_cross_entropy(
        ter_logits, ter_target,
        weight=focal_ce_w,
        gamma=gamma,
        reduction="mean",
        ignore_index=ignore_index,
    )

    loss_dice = multiclass_soft_dice_loss(
        ter_logits, ter_target,
        num_classes=3,
        ignore_index=ignore_index if ignore_index >= 0 else None,
        class_weights=dice_w,
    )

    total = loss_fce + dice_weight * loss_dice
    return total, loss_fce, loss_dice


# -----------------------------
# Compute ternary CE weights from TRAIN masks
# -----------------------------
def compute_ternary_ce_weights_from_manifest(df, base_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Reads ter_gt_path masks from the TRAIN manifest and returns:
      - counts: [3] int64 pixel counts for (0=bg,1=inside,2=boundary)
      - weights: [3] float64 weights normalized to have mean ~1
    """
    counts = np.zeros(3, dtype=np.int64)

    for rel in df["ter_gt_path"].astype(str).tolist():
        p = (base_dir / rel).resolve()
        arr = np.asarray(Image.open(p))
        if arr.ndim == 3:
            arr = arr[..., 0]

        arr = arr.astype(np.int64, copy=False)

        # fast bincount
        bc = np.bincount(arr.reshape(-1), minlength=3)[:3]
        counts += bc.astype(np.int64, copy=False)

    total = int(counts.sum())
    if total <= 0:
        raise RuntimeError("Ternary masks appear empty; cannot compute CE weights.")

    freq = counts / total
    w = 1.0 / np.maximum(freq, 1e-12)

    # normalize so average weight ~ 1 (keeps loss scale stable)
    w = w / w.mean()

    return counts, w


# -----------------------------
# Logging + plotting
# -----------------------------
def append_metrics_row(path: Path, fieldnames: list[str], row: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore", restval="")
        if write_header:
            w.writeheader()
        w.writerow(row)


def _fmt_per_class(d: dict, names: dict[int, str]) -> str:
    parts = []
    for c in sorted(d.keys()):
        parts.append(f"{names.get(c, str(c))}:{d[c]:.4f}")
    return " ".join(parts)


def _history_append(history: dict[str, list], row: dict, fieldnames: list[str]) -> None:
    for k in fieldnames:
        history.setdefault(k, [])
        history[k].append(row.get(k, None))


def save_preview_panel_dual(
    out_path: Path,
    x: torch.Tensor,
    sem_gt: torch.Tensor,
    sem_logits: torch.Tensor,
    ter_gt: torch.Tensor,
    ter_logits: torch.Tensor,
    b: int = 0,
):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    img = x[b].detach().cpu().numpy()
    img = np.transpose(img, (1, 2, 0))
    # Undo ImageNet normalization for display
    img = img * IMAGENET_STD_NP + IMAGENET_MEAN_NP
    img = np.clip(img, 0, 1)

    sem_gt_np = sem_gt[b].detach().cpu().numpy()
    sem_pred = torch.argmax(sem_logits[b], dim=0).detach().cpu().numpy()

    ter_gt_np = ter_gt[b].detach().cpu().numpy()
    ter_pred = torch.argmax(ter_logits[b], dim=0).detach().cpu().numpy()

    fig, ax = plt.subplots(1, 5, figsize=(18, 4))
    ax[0].imshow(img); ax[0].set_title("RGB"); ax[0].axis("off")
    ax[1].imshow(sem_gt_np); ax[1].set_title("SEM GT (0-4)"); ax[1].axis("off")
    ax[2].imshow(sem_pred); ax[2].set_title("SEM Pred"); ax[2].axis("off")
    ax[3].imshow(ter_gt_np); ax[3].set_title("TER GT (0-2)"); ax[3].axis("off")
    ax[4].imshow(ter_pred); ax[4].set_title("TER Pred"); ax[4].axis("off")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_dashboard(plots_dir: Path, history: dict[str, list]):
    """
    Single dashboard figure (3x2):
      [0,0] Losses (semantic + ternary)
      [0,1] Semantic mIoU (macro+micro)
      [1,0] Semantic Dice (macro) by class
      [1,1] Semantic IoU (micro) by class
      [2,0] LR
      [2,1] Ternary mIoU (macro+micro)
    """
    epochs = history.get("epoch", [])
    if not epochs or len(epochs) < 2:
        return

    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    def _series(key: str):
        xs, ys = [], []
        for e, v in zip(history.get("epoch", []), history.get(key, [])):
            if v is None or v == "":
                continue
            try:
                xs.append(int(e))
                ys.append(float(v))
            except Exception:
                continue
        return xs, ys

    def _plot_lines(ax, keys: list[str], title: str, ylabel: str):
        any_plotted = False
        for k in keys:
            xs, ys = _series(k)
            if len(xs) >= 2:
                ax.plot(xs, ys, label=k)
                any_plotted = True
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.set_ylabel(ylabel)
        if any_plotted:
            ax.legend()

    fig, axs = plt.subplots(3, 2, figsize=(16, 12))

    _plot_lines(
        axs[0, 0],
        [
            "train_loss_total",
            "val_loss_total",
            "train_loss_sem",
            "val_loss_sem",
            "train_loss_ter",
            "val_loss_ter",
        ],
        "Loss (total + per-head)",
        "loss",
    )

    _plot_lines(axs[0, 1], ["miou_all_macro", "miou_all_micro"], "Semantic mIoU", "IoU")

    ax = axs[1, 0]
    any_plotted = False
    for c in SEM_CLASSES:
        nm = SEMANTIC_CLASS_NAMES[c]
        k = f"dice_macro_{nm}"
        xs, ys = _series(k)
        if len(xs) >= 2:
            ax.plot(xs, ys, label=nm)
            any_plotted = True
    ax.set_title("Semantic Dice (macro) by class")
    ax.set_xlabel("epoch")
    ax.set_ylabel("dice")
    if any_plotted:
        ax.legend(ncol=2)

    ax = axs[1, 1]
    any_plotted = False
    for c in SEM_CLASSES:
        nm = SEMANTIC_CLASS_NAMES[c]
        k = f"iou_{nm}"
        xs, ys = _series(k)
        if len(xs) >= 2:
            ax.plot(xs, ys, label=nm)
            any_plotted = True
    ax.set_title("Semantic IoU (micro) by class")
    ax.set_xlabel("epoch")
    ax.set_ylabel("IoU")
    if any_plotted:
        ax.legend(ncol=2)

    _plot_lines(axs[2, 0], ["lr"], "Learning Rate", "lr")
    _plot_lines(axs[2, 1], ["ter_miou_macro", "ter_miou_micro"], "Ternary mIoU", "IoU")
    _plot_lines(axs[2, 1], ["combo_inside_boundary"], "Combo (inside+boundary)", "score")

    plt.tight_layout()
    out_path = plots_dir / "dashboard.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[PLOT] {out_path}")


# -----------------------------
# Plateau stopper (monitor-based)
# -----------------------------
@dataclass
class PlateauStopper:
    patience: int = 80
    min_delta: float = 5e-4
    min_epochs: int = 120
    mode: str = "min"  # "min" or "max"
    ema_alpha: float = 0.3

    best: float = 0.0
    global_step: int = 0
    bad_epochs: int = 0
    ema: Optional[float] | None = None

    def __post_init__(self):
        # Proper mode-aware init
        self.best = float("-inf") if self.mode == "max" else float("inf")

    def step(self, value: float, epoch: int) -> tuple[bool, dict]:
        if self.ema_alpha and self.ema_alpha > 0:
            self.ema = value if self.ema is None else (self.ema_alpha * value + (1 - self.ema_alpha) * self.ema)
            v = self.ema
        else:
            v = value

        improved = (v < self.best - self.min_delta) if self.mode == "min" else (v > self.best + self.min_delta)

        if improved:
            self.best = v
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1

        should_stop = (epoch >= self.min_epochs) and (self.bad_epochs >= self.patience)
        info = {
            "raw": float(value),
            "smoothed": float(v),
            "best": float(self.best),
            "bad_epochs": int(self.bad_epochs),
            "patience": int(self.patience),
        }
        return should_stop, info


# -----------------------------
# Augmentation wrapper (joint image + mask transforms)
# -----------------------------
class AugmentedDataset(torch.utils.data.Dataset):
    """
    Wraps ExportManifestDataset to apply joint albumentations to
    image + sem_mask + ter_mask with identical spatial transforms.

    The base dataset returns tensors; this wrapper converts back to numpy
    for albumentations, then re-converts to tensors.
    """
    def __init__(self, base_dataset, transform=None):
        self.base = base_dataset
        self.transform = transform
        # Forward DataFrame access for sampler weight computation
        self.df = base_dataset.df

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        result = self.base[idx]
        # base returns (x, sem, ter, meta) when return_meta=True
        # or (x, sem, ter) when return_meta=False
        if len(result) == 4:
            x, sem, ter, meta = result
        else:
            x, sem, ter = result
            meta = None

        if self.transform is not None:
            # Convert tensors to numpy for albumentations
            # x is [3, H, W] float32 [0,1] -> [H, W, 3] uint8 [0,255]
            img_np = (x.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
            sem_np = sem.numpy().astype(np.int32)
            ter_np = ter.numpy().astype(np.int32)

            augmented = self.transform(
                image=img_np,
                masks=[sem_np, ter_np],
            )

            # ToTensorV2 converts image to [3,H,W] float32 tensor
            x = augmented["image"]
            # albumentations 2.x: ToTensorV2 may convert masks to tensors too
            m0, m1 = augmented["masks"][0], augmented["masks"][1]
            sem = m0.long() if isinstance(m0, torch.Tensor) else torch.from_numpy(m0).long()
            ter = m1.long() if isinstance(m1, torch.Tensor) else torch.from_numpy(m1).long()

        if meta is not None:
            return x, sem, ter, meta
        return x, sem, ter


# -----------------------------
# Model: shared encoder+decoder, two heads
# -----------------------------
class SharedUnetTwoHead(nn.Module):
    """
    Version-robust dual-head U-Net:
      - shared SMP Unet encoder+decoder
      - two segmentation heads on top of decoder output

    Returns:
      sem_logits: [B, sem_classes, H, W]
      ter_logits: [B, ter_classes, H, W]
    """
    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_weights: str | None = "imagenet",
        in_channels: int = 3,
        decoder_channels=(256, 128, 64, 32, 16),
        sem_classes: int = 5,
        ter_classes: int = 3,
        activation=None,
    ):
        super().__init__()

        self.decoder_channels = tuple(decoder_channels)

        self.base = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=1,  # dummy
            activation=None,
            decoder_channels=self.decoder_channels,
        )

        dec_out_ch = self.decoder_channels[-1]

        # Spatial dropout on shared decoder features (regularization)
        self.drop = nn.Dropout2d(p=0.2)

        self.sem_head = SegmentationHead(
            in_channels=dec_out_ch,
            out_channels=sem_classes,
            activation=activation,
            kernel_size=3,
        )
        self.ter_head = SegmentationHead(
            in_channels=dec_out_ch,
            out_channels=ter_classes,
            activation=activation,
            kernel_size=3,
        )

    def forward(self, x: torch.Tensor):
        feats = self.base.encoder(x)
        try:
            dec = self.base.decoder(*feats)
        except TypeError:
            dec = self.base.decoder(feats)
        dec = self.drop(dec)  # spatial dropout before heads
        sem = self.sem_head(dec)
        ter = self.ter_head(dec)
        return sem, ter


# -----------------------------
# Eval
# -----------------------------
@torch.no_grad()
def evaluate(
    model: nn.Module,
    dl,
    device: torch.device,
    sem_dice_w: torch.Tensor | None = None,
    ter_ce_w: torch.Tensor | None = None,
    ter_loss_weight: float = 1.0,
    ignore_index_sem: int | None = None,
    ignore_index_ter: int | None = None,
    dice_fg_only: bool = False,
    focal_gamma: float = 0.0,
    ter_dice_weight: float = 0.0,
):
    model.eval()
    eps = 1e-6

    sem_loss_sum = 0.0
    ter_loss_sum = 0.0
    total_loss_sum = 0.0
    n = 0

    dice_macro_sum = {c: 0.0 for c in SEM_CLASSES}
    dice_micro_sum = {c: 0.0 for c in SEM_CLASSES}
    n_batches = 0

    miou_all_macro_img_sum = 0.0
    n_all_imgs = 0
    total_inter_all = torch.zeros((), device=device)
    total_union_all = torch.zeros((), device=device)
    inter_c_sum = {c: torch.zeros((), device=device) for c in SEM_CLASSES}
    union_c_sum = {c: torch.zeros((), device=device) for c in SEM_CLASSES}
    gt_pixels_sum = {c: 0 for c in SEM_CLASSES}
    pr_pixels_sum = {c: 0 for c in SEM_CLASSES}

    ter_dice_macro_sum = {c: 0.0 for c in TER_CLASSES}
    ter_dice_micro_sum = {c: 0.0 for c in TER_CLASSES}
    ter_gt_pixels_sum = {c: 0 for c in TER_CLASSES}
    ter_pr_pixels_sum = {c: 0 for c in TER_CLASSES}
    ter_n_batches = 0

    ter_miou_macro_img_sum = 0.0
    n_ter_imgs = 0
    ter_total_inter = torch.zeros((), device=device)
    ter_total_union = torch.zeros((), device=device)
    ter_inter_c_sum = {c: torch.zeros((), device=device) for c in TER_CLASSES}
    ter_union_c_sum = {c: torch.zeros((), device=device) for c in TER_CLASSES}

    for batch in dl:
        x, sem, ter, _ = unpack_batch(batch)
        x = x.to(device)
        sem = sem.to(device)
        ter = ter.to(device)

        sem_logits, ter_logits = model(x)

        sem_loss, _ = semantic_loss_dice_only(
            sem_logits,
            sem,
            dice_w=sem_dice_w,
            ignore_index=ignore_index_sem,
        )

        ter_loss, _ter_fce, _ter_dice = ternary_combined_loss(
            ter_logits,
            ter,
            focal_ce_w=ter_ce_w,
            dice_w=ter_ce_w,
            gamma=focal_gamma,
            dice_weight=ter_dice_weight,
            ignore_index=ignore_index_ter if ignore_index_ter is not None else -100,
        )

        total = sem_loss + float(ter_loss_weight) * ter_loss

        bs = x.size(0)
        sem_loss_sum += float(sem_loss.item()) * bs
        ter_loss_sum += float(ter_loss.item()) * bs
        total_loss_sum += float(total.item()) * bs
        n += bs

        miou_per_img, _miou_micro_unused, _ = semantic_iou_all(sem_logits, sem, classes=SEM_CLASSES, eps=eps)
        miou_all_macro_img_sum += float(miou_per_img.sum().item())
        n_all_imgs += int(miou_per_img.numel())

        sem_pred = torch.argmax(sem_logits, dim=1)
        batch_inter_all = torch.zeros((), device=device)
        batch_union_all = torch.zeros((), device=device)
        for c in SEM_CLASSES:
            gt_c = (sem == c)
            pr_c = (sem_pred == c)
            inter = (gt_c & pr_c).sum().float()
            union = (gt_c | pr_c).sum().float()
            inter_c_sum[c] += inter
            union_c_sum[c] += union
            batch_inter_all += inter
            batch_union_all += union
        total_inter_all += batch_inter_all
        total_union_all += batch_union_all

        dm, di, cc = semantic_dice_per_class(sem_logits, sem, classes=SEM_CLASSES, fg_only=dice_fg_only, eps=eps)
        for c in SEM_CLASSES:
            dice_macro_sum[c] += dm[c]
            dice_micro_sum[c] += di[c]
            gt_pixels_sum[c] += int(cc[c]["gt_pixels"])
            pr_pixels_sum[c] += int(cc[c]["pred_pixels"])
        n_batches += 1

        ter_miou_per_img, _ter_miou_micro_unused, _ = ternary_iou_all(ter_logits, ter, classes=TER_CLASSES, eps=eps)
        ter_miou_macro_img_sum += float(ter_miou_per_img.sum().item())
        n_ter_imgs += int(ter_miou_per_img.numel())

        ter_pred = torch.argmax(ter_logits, dim=1)
        batch_inter_ter = torch.zeros((), device=device)
        batch_union_ter = torch.zeros((), device=device)
        for c in TER_CLASSES:
            gt_c = (ter == c)
            pr_c = (ter_pred == c)
            inter = (gt_c & pr_c).sum().float()
            union = (gt_c | pr_c).sum().float()
            ter_inter_c_sum[c] += inter
            ter_union_c_sum[c] += union
            batch_inter_ter += inter
            batch_union_ter += union
        ter_total_inter += batch_inter_ter
        ter_total_union += batch_union_ter

        tdm, tdi, tcc = ternary_dice_per_class(ter_logits, ter, classes=TER_CLASSES, eps=eps)
        for c in TER_CLASSES:
            ter_dice_macro_sum[c] += tdm[c]
            ter_dice_micro_sum[c] += tdi[c]
            ter_gt_pixels_sum[c] += int(tcc[c]["gt_pixels"])
            ter_pr_pixels_sum[c] += int(tcc[c]["pred_pixels"])
        ter_n_batches += 1

    out: dict[str, Any] = {}

    out["loss_sem"] = sem_loss_sum / max(1, n)
    out["loss_ter"] = ter_loss_sum / max(1, n)
    out["loss_total"] = total_loss_sum / max(1, n)

    out["miou_all_macro"] = miou_all_macro_img_sum / max(1, n_all_imgs)
    out["miou_all_micro"] = float(((total_inter_all + eps) / (total_union_all + eps)).item())

    for c in SEM_CLASSES:
        nm = SEMANTIC_CLASS_NAMES[c]
        out[f"iou_{nm}"] = float(((inter_c_sum[c] + eps) / (union_c_sum[c] + eps)).item())
        out[f"dice_macro_{nm}"] = dice_macro_sum[c] / max(1, n_batches)
        out[f"dice_micro_{nm}"] = dice_micro_sum[c] / max(1, n_batches)
        out[f"gt_pixels_{nm}"] = gt_pixels_sum[c]
        out[f"pred_pixels_{nm}"] = pr_pixels_sum[c]

    out["dice_macro_by_class"] = {c: dice_macro_sum[c] / max(1, n_batches) for c in SEM_CLASSES}
    out["dice_micro_by_class"] = {c: dice_micro_sum[c] / max(1, n_batches) for c in SEM_CLASSES}

    out["ter_miou_macro"] = ter_miou_macro_img_sum / max(1, n_ter_imgs)
    out["ter_miou_micro"] = float(((ter_total_inter + eps) / (ter_total_union + eps)).item())

    for c in TER_CLASSES:
        nm = TER_CLASS_NAMES[c]
        out[f"ter_iou_{nm}"] = float(((ter_inter_c_sum[c] + eps) / (ter_union_c_sum[c] + eps)).item())
        out[f"ter_dice_macro_{nm}"] = ter_dice_macro_sum[c] / max(1, ter_n_batches)
        out[f"ter_dice_micro_{nm}"] = ter_dice_micro_sum[c] / max(1, ter_n_batches)
        out[f"ter_gt_pixels_{nm}"] = ter_gt_pixels_sum[c]
        out[f"ter_pred_pixels_{nm}"] = ter_pr_pixels_sum[c]

    out["ter_dice_macro_by_class"] = {c: ter_dice_macro_sum[c] / max(1, ter_n_batches) for c in TER_CLASSES}
    out["ter_dice_micro_by_class"] = {c: ter_dice_micro_sum[c] / max(1, ter_n_batches) for c in TER_CLASSES}

    return out


# -----------------------------
# Main
# -----------------------------
def main():
    seed_everything(BASE_SEED)

    # Hyperparams
    batch_size = 8
    lr = 1e-4
    max_epochs = 120
    weight_decay = 3e-4              # Run 4 best (was 1e-4)

    # ---- NEW: freeze semantic head + focus loss on ternary ----
    FREEZE_SEMANTIC_HEAD = False
    SEM_LOSS_WEIGHT = 0.3            # Run 4 best (was 0.7)
    TER_LOSS_WEIGHT = 1.0            # keep aligned with best run (lambda_ter=2.0) unless you override here

    # ---- NEW: extra boundary upweighting (in addition to inverse-frequency weights) ----
    TER_BOUNDARY_MULT = 1.0          # multiplies class-2 weight after frequency weighting
    TER_WEIGHT_CAP_MIN = 0.05
    TER_WEIGHT_CAP_MAX = 10.0
    TER_BOUNDARY_HARD_CAP = 2.0

    # ---- Focal loss for ternary head ----
    FOCAL_GAMMA = 2.0                # 0.0 = plain CE; 2.0 = standard focal loss
    TER_DICE_WEIGHT = 0.5            # lambda for Dice in combined ternary loss: focalCE + TER_DICE_WEIGHT * Dice

    MONITOR_KEY = "combo_inside_boundary"
    MONITOR_MODE = "max"

    SELECTION_KEY = "combo_inside_boundary"
    SELECTION_MODE = "max"

    plateau_patience = 20
    plateau_min_delta = 5e-4
    plateau_min_epochs = 25

    PREVIEW_EVERY_STEPS = 5000

    ENCODER = "resnet34"
    ENCODER_WEIGHTS = "imagenet"

    # -----------------------------
    # Data
    # -----------------------------
    DATA_ROOT = Path(env_str("DATA_ROOT", "/teamspace/studios/this_studio/data/monusac_clean")).resolve()

    train_manifest_path = Path(env_str(
        "TRAIN_MANIFEST",
        str(DATA_ROOT / "MoNuSAC_outputs/export_patches/train_P256_S128_fg0.01/export_manifest_immune_counts.csv"),
    )).resolve()

    val_manifest_path = Path(env_str(
        "VAL_MANIFEST",
        str(DATA_ROOT / "MoNuSAC_outputs/export_patches/val_P256_S128_fg0.01/export_manifest_immune_counts.csv"),
    )).resolve()

    base_dir = Path(env_str("BASE_DIR", str(DATA_ROOT))).resolve()

    if not train_manifest_path.exists():
        raise FileNotFoundError(f"TRAIN_MANIFEST not found: {train_manifest_path}")
    if not val_manifest_path.exists():
        raise FileNotFoundError(f"VAL_MANIFEST not found: {val_manifest_path}")

    print("[DATA_ROOT]", DATA_ROOT)
    print("[TRAIN_MANIFEST]", train_manifest_path)
    print("[VAL_MANIFEST]", val_manifest_path)
    print("[BASE_DIR]", base_dir)

    # -----------------------------
    # Outputs
    # -----------------------------
    default_out = Path(env_str("OUT_ROOT", str(PROJECT_ROOT / "outputs")))
    out_root = default_out if default_out.is_absolute() else (PROJECT_ROOT / default_out)
    out_root = out_root.resolve()

    tag = f"FROZENSEM__BOUNDW__{ENCODER}__terLam{TER_LOSS_WEIGHT}__bmult{TER_BOUNDARY_MULT}__select_{SELECTION_KEY}"
    run_name = f"{tag}__{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    ckpt_dir = out_root / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_best_path = ckpt_dir / f"{run_name}__best.pt"
    ckpt_last_path = ckpt_dir / f"{run_name}__last.pt"

    runs_dir = out_root / "runs" / run_name
    runs_dir.mkdir(parents=True, exist_ok=True)

    metrics_csv = runs_dir / "metrics.csv"
    config_json = runs_dir / "config.json"

    preview_dir = out_root / "preview" / run_name
    preview_dir.mkdir(parents=True, exist_ok=True)

    if metrics_csv.exists():
        metrics_csv.unlink()

    # Device + determinism
    device = get_device()
    print("device:", device)
    print("[PROJECT_ROOT]", PROJECT_ROOT)
    print("[OUT_ROOT]", out_root.resolve())
    print("[SEM_WEIGHTS_JSON]", SEM_WEIGHTS_JSON, "exists=", SEM_WEIGHTS_JSON.exists())

    torch.use_deterministic_algorithms(True, warn_only=True)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    # DataLoader performance knobs
    num_workers = int(env_str("NUM_WORKERS", "4"))
    pin_memory = (device.type == "cuda")
    persistent_workers = (num_workers > 0)

    # Data (raw datasets — no augmentation yet)
    train_data_config = ExportManifestConfig(csv_path=train_manifest_path, base_dir=base_dir)
    val_data_config   = ExportManifestConfig(csv_path=val_manifest_path,   base_dir=base_dir)
    train_ds_raw = ExportManifestDataset(train_data_config)
    val_ds_raw   = ExportManifestDataset(val_data_config)

    # ---- Joint augmentation pipelines (albumentations) ----
    # Train: geometric + color + mild affine + ImageNet normalize
    train_aug = A.Compose([
        # Geometric (free for histology — no orientation bias)
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        # Color (image-only; does not affect masks)
        A.ColorJitter(
            brightness=0.15, contrast=0.15,
            saturation=0.1, hue=0.04, p=0.5,
        ),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        # Mild affine (conservative to preserve thin 3px boundaries)
        A.Affine(
            translate_percent=0.05,
            scale=(0.9, 1.1),
            rotate=(-15, 15),
            interpolation=1,        # cv2.INTER_LINEAR
            mask_interpolation=0,   # cv2.INTER_NEAREST (critical for masks)
            p=0.3,
        ),
        # ImageNet normalization + tensor conversion
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])

    # Validation: normalize only, no augmentation
    val_aug = A.Compose([
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])

    # Wrap with augmentation
    train_ds = AugmentedDataset(train_ds_raw, transform=train_aug)
    val_ds   = AugmentedDataset(val_ds_raw,   transform=val_aug)

    # Semantic weights (unused for optimization when SEM_LOSS_WEIGHT=0.0, but kept for completeness)
    sem_w = torch.tensor(wobj["weights"], dtype=torch.float32, device=device).clone()
    sem_w[3] = min(sem_w[3].item(), 6.0)  # neutrophil cap
    sem_w[4] = min(sem_w[4].item(), 2.0)  # macrophage cap
    print("[sem weights (capped)]", sem_w.detach().cpu().numpy().tolist())

    # -----------------------------
    # Ternary CE weights from TRAIN ternary masks + boundary multiplier
    # Compute raw inverse-frequency weights
    ter_counts, ter_w_raw = compute_ternary_ce_weights_from_manifest(
        train_ds.df, base_dir=base_dir
    )
    ter_w_raw = ter_w_raw.astype(np.float64)

    # ---- Stage 1: initial clip (safety) ----
    ter_w_stage1 = np.clip(
        ter_w_raw,
        TER_WEIGHT_CAP_MIN,
        TER_WEIGHT_CAP_MAX,
    )

    # ---- Stage 2: optional boundary upweight ----
    ter_w_stage2 = ter_w_stage1.copy()
    ter_w_stage2[2] *= float(TER_BOUNDARY_MULT)

    # ---- Stage 3: clip + hard cap (halo control) ----
    ter_w_clip = np.clip(
        ter_w_stage2,
        TER_WEIGHT_CAP_MIN,
        TER_WEIGHT_CAP_MAX,
    )
    ter_w_clip[2] = min(ter_w_clip[2], float(TER_BOUNDARY_HARD_CAP))

    # ---- Stage 4: renormalize (keep mean ≈ 1) ----
    ter_w_final = ter_w_clip / ter_w_clip.mean()

    ter_ce_w = torch.tensor(ter_w_final, dtype=torch.float32, device=device)

    # ---- Logging ----
    print("[ternary pixel counts bg/inside/boundary]", ter_counts.tolist())
    print("[ternary CE weights RAW]", ter_w_raw.tolist())
    print("[ternary CE weights CLIP]", ter_w_clip.tolist())
    print("[ternary CE weights FINAL]", ter_w_final.tolist())

    # ---- immune-aware sampling (TRAIN ONLY) ----
    req_cols = ["px_lymphocyte", "px_neutrophil", "px_macrophage"]
    missing = [c for c in req_cols if c not in train_ds.df.columns]
    if missing:
        raise ValueError(f"Train manifest missing columns: {missing}. Use export_manifest_immune_counts.csv")

    px_lym = train_ds.df["px_lymphocyte"].to_numpy(dtype=np.int64)
    px_neu = train_ds.df["px_neutrophil"].to_numpy(dtype=np.int64)
    px_mac = train_ds.df["px_macrophage"].to_numpy(dtype=np.int64)

    T_LYM = 25
    T_NEU = 10
    T_MAC = 10

    has_lym = px_lym >= T_LYM
    has_neu = px_neu >= T_NEU
    has_mac = px_mac >= T_MAC

    weights = np.ones(len(train_ds), dtype=np.float64)
    weights[has_lym] *= 2.0
    weights[has_neu] *= 8.0
    weights[has_mac] *= 8.0
    weights[has_neu & has_mac] *= 1.5

    weights = np.nan_to_num(weights, nan=1.0, posinf=1.0, neginf=1.0)
    weights = np.clip(weights, 1e-8, None)

    print(
        f"[class-sampler] has_lym={has_lym.mean():.2%} "
        f"has_neu={has_neu.mean():.2%} has_mac={has_mac.mean():.2%} "
        f"(T_LYM={T_LYM}, T_NEU={T_NEU}, T_MAC={T_MAC})"
    )

    train_sampler = WeightedRandomSampler(
        weights.tolist(),
        num_samples=len(weights),
        replacement=True,
    )

    # Probe val batch for sanity + preview monitor
    val_probe_dl = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    b0 = next(iter(val_probe_dl))
    x0, sem0, ter0, _ = unpack_batch(b0)

    print("[SEM] unique class ids in GT:", torch.unique(sem0).cpu().tolist())
    print("[TER] unique class ids in GT:", torch.unique(ter0).cpu().tolist())

    monitor_x = x0.to(device)
    monitor_sem = sem0.to(device)
    monitor_ter = ter0.to(device)

    # Model (dual-head)
    model = SharedUnetTwoHead(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        in_channels=3,
        decoder_channels=(256, 128, 64, 32, 16),
        sem_classes=5,
        ter_classes=3,
        activation=None,
    ).to(device)

    # -----------------------------
    # NEW: freeze semantic head weights (keep encoder+decoder trainable)
    # -----------------------------
    if FREEZE_SEMANTIC_HEAD:
        for p in model.sem_head.parameters():
            p.requires_grad = False
        print("[FREEZE] semantic head frozen (requires_grad=False)")

    # Config logging
    config = {
        "run_name": run_name,
        "seed": int(BASE_SEED),
        "device": str(device),
        "max_epochs": int(max_epochs),
        "batch_size": int(batch_size),
        "lr_init": float(lr),
        "weight_decay": float(weight_decay),
        "encoder": ENCODER,
        "encoder_weights": ENCODER_WEIGHTS,
        "freeze_semantic_head": bool(FREEZE_SEMANTIC_HEAD),
        "sem_loss_weight": float(SEM_LOSS_WEIGHT),
        "ter_loss_weight": float(TER_LOSS_WEIGHT),
        "ter_boundary_mult": float(TER_BOUNDARY_MULT),
        "focal_gamma": float(FOCAL_GAMMA),
        "ter_dice_weight": float(TER_DICE_WEIGHT),
        "monitor_key": MONITOR_KEY,
        "monitor_mode": MONITOR_MODE,
        "selection_key": SELECTION_KEY,
        "selection_mode": SELECTION_MODE,
        "train_manifest": str(train_manifest_path),
        "val_manifest": str(val_manifest_path),
        "base_dir": str(base_dir),
        "sem_weights_json": str(SEM_WEIGHTS_JSON),
        "out_root": str(out_root.resolve()),
        "preview_every_steps": int(PREVIEW_EVERY_STEPS),
        "num_workers": int(num_workers),
        "pin_memory": bool(pin_memory),
        "ternary_encoding": {"0": "background", "1": "inside", "2": "boundary(inner)"},
        "ternary_ce_weights": ter_ce_w.detach().cpu().numpy().tolist(),
        "ternary_pixel_counts": ter_counts.tolist(),
        "dropout_p": 0.2,
        "augmentation": True,
        "imagenet_normalize": True,
        "augmentation_details": "HFlip+VFlip+Rot90+ColorJitter+GaussBlur+ShiftScaleRotate+ImageNetNorm",
    }
    config_json.write_text(json.dumps(config, indent=2))

    # -----------------------------
    # Optimizer: exclude frozen semantic head params
    # -----------------------------
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable parameters found (did you freeze too much?).")

    opt = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt,
        T_max=max_epochs,
        eta_min=1e-6,
    )

    stopper = PlateauStopper(
        patience=3 * plateau_patience,
        min_delta=plateau_min_delta,
        min_epochs=plateau_min_epochs,
        mode=MONITOR_MODE,
        ema_alpha=0.3,
    )

    # Metrics schema
    metric_fields = [
        "epoch", "global_step",
        "train_loss_sem", "train_loss_ter", "train_loss_total",
        "val_loss_sem", "val_loss_ter", "val_loss_total",
        "miou_all_macro", "miou_all_micro",
        "ter_miou_macro", "ter_miou_micro",
        "monitor_value",
        "selection_value",
        "plateau_raw", "plateau_ema", "plateau_best", "plateau_bad_epochs", "plateau_patience",
        "should_stop", "lr",
    ]

    if "combo_inside_boundary" not in metric_fields:
        metric_fields.append("combo_inside_boundary")

    for c in SEM_CLASSES:
        nm = SEMANTIC_CLASS_NAMES[c]
        metric_fields += [
            f"iou_{nm}",
            f"dice_macro_{nm}", f"dice_micro_{nm}",
            f"gt_pixels_{nm}", f"pred_pixels_{nm}",
        ]

    for c in TER_CLASSES:
        nm = TER_CLASS_NAMES[c]
        metric_fields += [
            f"ter_iou_{nm}",
            f"ter_dice_macro_{nm}", f"ter_dice_micro_{nm}",
            f"ter_gt_pixels_{nm}", f"ter_pred_pixels_{nm}",
        ]

    history: dict[str, list] = {k: [] for k in metric_fields}

    best_selection = float("-inf") if SELECTION_MODE == "max" else float("inf")
    global_step = 0

    try:
        for epoch in range(1, max_epochs + 1):
            train_dl = DataLoader(
                train_ds,
                batch_size=batch_size,
                sampler=train_sampler,
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=persistent_workers,
                worker_init_fn=seed_worker,
            )

            val_dl = DataLoader(
                val_ds,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=persistent_workers,
            )

            # ---- Train ----
            model.train()
            sem_sum = 0.0
            ter_sum = 0.0
            tot_sum = 0.0
            n_seen = 0

            for step, batch in enumerate(train_dl, start=1):
                x, sem, ter, _ = unpack_batch(batch)
                x = x.to(device, non_blocking=True)
                sem = sem.to(device, non_blocking=True)
                ter = ter.to(device, non_blocking=True)

                sem_logits, ter_logits = model(x)

                # semantic loss (computed but typically not used when SEM_LOSS_WEIGHT=0.0)
                sem_loss, _ = semantic_loss_dice_only(
                    sem_logits,
                    sem,
                    dice_w=None,  # set to sem_w if you want weighted Dice
                    ignore_index=None,
                )

                # ternary combined loss: focal CE + Dice
                ter_loss, _ter_fce, _ter_dice = ternary_combined_loss(
                    ter_logits,
                    ter,
                    focal_ce_w=ter_ce_w,
                    dice_w=ter_ce_w,
                    gamma=FOCAL_GAMMA,
                    dice_weight=TER_DICE_WEIGHT,
                )

                loss = float(SEM_LOSS_WEIGHT) * sem_loss + float(TER_LOSS_WEIGHT) * ter_loss

                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

                bs = x.size(0)
                sem_sum += float(sem_loss.item()) * bs
                ter_sum += float(ter_loss.item()) * bs
                tot_sum += float(loss.item()) * bs
                n_seen += bs
                global_step += 1

                if (PREVIEW_EVERY_STEPS > 0) and (global_step % PREVIEW_EVERY_STEPS == 0):
                    model.eval()
                    with torch.no_grad():
                        sem_logits_m, ter_logits_m = model(monitor_x)
                    out_path = preview_dir / f"epoch{epoch:03d}_step{global_step:06d}.png"
                    save_preview_panel_dual(out_path, monitor_x, monitor_sem, sem_logits_m, monitor_ter, ter_logits_m, b=0)
                    print(f"[PREVIEW] {out_path}")
                    model.train()

            train_loss_sem = sem_sum / max(1, n_seen)
            train_loss_ter = ter_sum / max(1, n_seen)
            train_loss_total = tot_sum / max(1, n_seen)

            # ---- Eval ----
            val_metrics = evaluate(
                model=model,
                dl=val_dl,
                device=device,
                sem_dice_w=None,
                ter_ce_w=ter_ce_w,
                ter_loss_weight=TER_LOSS_WEIGHT,
                ignore_index_sem=None,
                ignore_index_ter=None,
                dice_fg_only=False,
                focal_gamma=FOCAL_GAMMA,
                ter_dice_weight=TER_DICE_WEIGHT,
            )

            # ----- derive combined monitor/select metric (no KeyError) -----
            combo_inside_boundary = (
                0.7 * float(val_metrics["ter_dice_micro_inside"])
                + 0.3 * float(val_metrics["ter_dice_micro_boundary"])
            )

            def resolve_metric(key: str) -> float:
                if key == "combo_inside_boundary":
                    return combo_inside_boundary
                return float(val_metrics[key])

            monitor_value = resolve_metric(MONITOR_KEY)
            selection_value = resolve_metric(SELECTION_KEY)

            prev_lr = opt.param_groups[0]["lr"]
            scheduler.step()
            lr_now = opt.param_groups[0]["lr"]
            if abs(lr_now - prev_lr) > 1e-10:
                print(f"[LR] CosineAnnealing: {prev_lr:.3e} -> {lr_now:.3e}")

            should_stop, stop_info = stopper.step(monitor_value, epoch)

            row = {
                "epoch": epoch,
                "global_step": global_step,

                "train_loss_sem": train_loss_sem,
                "train_loss_ter": train_loss_ter,
                "train_loss_total": train_loss_total,

                "val_loss_sem": float(val_metrics["loss_sem"]),
                "val_loss_ter": float(val_metrics["loss_ter"]),
                "val_loss_total": float(val_metrics["loss_total"]),

                "miou_all_macro": float(val_metrics["miou_all_macro"]),
                "miou_all_micro": float(val_metrics["miou_all_micro"]),

                "ter_miou_macro": float(val_metrics["ter_miou_macro"]),
                "ter_miou_micro": float(val_metrics["ter_miou_micro"]),

                "monitor_value": monitor_value,
                "selection_value": selection_value,
                "combo_inside_boundary": combo_inside_boundary,

                "plateau_raw": stop_info["raw"],
                "plateau_ema": stop_info["smoothed"],
                "plateau_best": stop_info["best"],
                "plateau_bad_epochs": stop_info["bad_epochs"],
                "plateau_patience": stop_info["patience"],
                "should_stop": int(should_stop),
                "lr": lr_now,
            }
            
        
            for c in SEM_CLASSES:
                nm = SEMANTIC_CLASS_NAMES[c]
                row[f"iou_{nm}"] = float(val_metrics[f"iou_{nm}"])
                row[f"dice_macro_{nm}"] = float(val_metrics[f"dice_macro_{nm}"])
                row[f"dice_micro_{nm}"] = float(val_metrics[f"dice_micro_{nm}"])
                row[f"gt_pixels_{nm}"] = int(val_metrics[f"gt_pixels_{nm}"])
                row[f"pred_pixels_{nm}"] = int(val_metrics[f"pred_pixels_{nm}"])

            for c in TER_CLASSES:
                nm = TER_CLASS_NAMES[c]
                row[f"ter_iou_{nm}"] = float(val_metrics[f"ter_iou_{nm}"])
                row[f"ter_dice_macro_{nm}"] = float(val_metrics[f"ter_dice_macro_{nm}"])
                row[f"ter_dice_micro_{nm}"] = float(val_metrics[f"ter_dice_micro_{nm}"])
                row[f"ter_gt_pixels_{nm}"] = int(val_metrics[f"ter_gt_pixels_{nm}"])
                row[f"ter_pred_pixels_{nm}"] = int(val_metrics[f"ter_pred_pixels_{nm}"])

            append_metrics_row(metrics_csv, metric_fields, row)
            _history_append(history, row, metric_fields)

            print(f"\n===== EPOCH {epoch} SUMMARY =====")
            print("\n-- TRAIN --")
            print(f"train_loss_sem:   {train_loss_sem:.4f}  (NOTE: logged even if SEM_LOSS_WEIGHT={SEM_LOSS_WEIGHT})")
            print(f"train_loss_ter:   {train_loss_ter:.4f}")
            print(f"train_loss_total: {train_loss_total:.4f}")

            print("\n-- VAL (LOSS) --")
            print(f"val_loss_sem:     {val_metrics['loss_sem']:.4f}")
            print(f"val_loss_ter:     {val_metrics['loss_ter']:.4f}")
            print(f"val_loss_total:   {val_metrics['loss_total']:.4f}")

            print("\n-- VAL (SEMANTIC) --")
            print(f"miou_all_macro:   {val_metrics['miou_all_macro']:.4f}")
            print(f"miou_all_micro:   {val_metrics['miou_all_micro']:.4f}")
            print("dice_macro:", _fmt_per_class(val_metrics["dice_macro_by_class"], SEMANTIC_CLASS_NAMES))
            print("dice_micro:", _fmt_per_class(val_metrics["dice_micro_by_class"], SEMANTIC_CLASS_NAMES))

            print("\n-- VAL (TERNARY) --")
            print(f"ter_miou_macro:   {val_metrics['ter_miou_macro']:.4f}")
            print(f"ter_miou_micro:   {val_metrics['ter_miou_micro']:.4f}")
            print("ter_dice_macro:", _fmt_per_class(val_metrics["ter_dice_macro_by_class"], TER_CLASS_NAMES))
            print("ter_dice_micro:", _fmt_per_class(val_metrics["ter_dice_micro_by_class"], TER_CLASS_NAMES))

            print(
                f"\n[MONITOR] {MONITOR_KEY}={monitor_value:.6f} (mode={MONITOR_MODE}) | "
                f"[SELECT] {SELECTION_KEY}={selection_value:.6f} (mode={SELECTION_MODE})"
            )
            print(
                f"[PLATEAU] raw={stop_info['raw']:.6f} ema={stop_info['smoothed']:.6f} "
                f"best={stop_info['best']:.6f} bad_epochs={stop_info['bad_epochs']}/{stop_info['patience']} "
                f"lr={lr_now:.3e}"
            )

            improved = False
            if SELECTION_MODE == "max":
                improved = np.isfinite(selection_value) and (selection_value > best_selection)
            else:
                improved = np.isfinite(selection_value) and (selection_value < best_selection)

            if improved:
                best_selection = float(selection_value)
                save_checkpoint(
                    ckpt_best_path,
                    model=model,
                    optimizer=opt,
                    scheduler=scheduler,
                    epoch=epoch,
                    global_step=global_step,
                    config={
                        "run_name": run_name,
                        "max_epochs": max_epochs,
                        "batch_size": batch_size,
                        "lr_init": lr,
                        "monitor_key": MONITOR_KEY,
                        "monitor_mode": MONITOR_MODE,
                        "selection_key": SELECTION_KEY,
                        "selection_mode": SELECTION_MODE,
                        "sem_loss_weight": SEM_LOSS_WEIGHT,
                        "ter_loss_weight": TER_LOSS_WEIGHT,
                        "freeze_semantic_head": FREEZE_SEMANTIC_HEAD,
                        "ter_boundary_mult": TER_BOUNDARY_MULT,
                        "focal_gamma": FOCAL_GAMMA,
                        "ter_dice_weight": TER_DICE_WEIGHT,
                        "ternary_ce_weights": ter_ce_w.detach().cpu().numpy().tolist(),
                    },
                    extra_pt={"best_selection": float(best_selection)},
                    extra_json={
                        "best_selection": float(best_selection),
                        "selection_key": SELECTION_KEY,
                        "selection_mode": SELECTION_MODE,
                        "monitor_key": MONITOR_KEY,
                        "monitor_mode": MONITOR_MODE,
                        "val_metrics": val_metrics,
                    },
                )
                update_alias(ckpt_dir / "LATEST__best.pt", ckpt_best_path)
                update_alias(ckpt_dir / "LATEST__best.json", ckpt_best_path.with_suffix(".json"))
                print(f"[CKPT] Saved BEST: {ckpt_best_path} (best_selection={best_selection:.6f})")

            save_checkpoint(
                ckpt_last_path,
                model=model,
                optimizer=opt,
                scheduler=scheduler,
                epoch=epoch,
                global_step=global_step,
                config={
                    "run_name": run_name,
                    "max_epochs": max_epochs,
                    "batch_size": batch_size,
                    "lr_init": lr,
                    "monitor_key": MONITOR_KEY,
                    "monitor_mode": MONITOR_MODE,
                    "selection_key": SELECTION_KEY,
                    "selection_mode": SELECTION_MODE,
                    "sem_loss_weight": SEM_LOSS_WEIGHT,
                    "ter_loss_weight": TER_LOSS_WEIGHT,
                    "freeze_semantic_head": FREEZE_SEMANTIC_HEAD,
                    "ter_boundary_mult": TER_BOUNDARY_MULT,
                    "focal_gamma": FOCAL_GAMMA,
                    "ter_dice_weight": TER_DICE_WEIGHT,
                    "ternary_ce_weights": ter_ce_w.detach().cpu().numpy().tolist(),
                },
                extra_pt={"best_selection": float(best_selection)},
                extra_json={
                    "best_selection": float(best_selection),
                    "selection_key": SELECTION_KEY,
                    "selection_mode": SELECTION_MODE,
                    "monitor_key": MONITOR_KEY,
                    "monitor_mode": MONITOR_MODE,
                    "val_metrics": val_metrics,
                },
            )
            update_alias(ckpt_dir / "LATEST__last.pt", ckpt_last_path)

            if should_stop:
                print(f"\n[STOP] Plateau detected on monitor ({MONITOR_KEY}) at epoch {epoch}.")
                break

    finally:
        plots_dir = runs_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        _plot_dashboard(plots_dir, history)
        write_json(plots_dir / "history.json", history)
        print(f"[HISTORY] Wrote: {plots_dir / 'history.json'}")


if __name__ == "__main__":
    main()