"""
utils.py

Shared utilities for the dual-head segmentation training pipeline.

Sections:
  1. Constants & path resolution
  2. Device & seeding
  3. JSON / checkpoint helpers
  4. Metrics (Dice, IoU)
  5. Loss functions
  6. Logging & plotting
  7. PlateauStopper
  8. AugmentedDataset
"""

import json
import csv
import os
import sys
import random
from pathlib import Path
from typing import Any, Optional
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

import albumentations as A
from albumentations.pytorch import ToTensorV2


# ================================================================
# 1. CONSTANTS & PATH RESOLUTION
# ================================================================

# ImageNet normalization constants (for pretrained ResNet encoder)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)
IMAGENET_MEAN_NP = np.array(IMAGENET_MEAN)
IMAGENET_STD_NP  = np.array(IMAGENET_STD)

TRAINING_DIR = Path(__file__).resolve().parent   # training/
PROJECT_ROOT = TRAINING_DIR.parent               # repo root (dl_seg_class_project/)

if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))


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


# globals / constants
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

# base for repo-relative asset resolution
_ASSET_BASE = PROJECT_ROOT

SEM_WEIGHTS_JSON = resolve_path("assets/class_weights.json", base=_ASSET_BASE)
with open(SEM_WEIGHTS_JSON, "r", encoding="utf-8") as f:
    wobj = json.load(f)


# ================================================================
# 2. DEVICE & SEEDING
# ================================================================

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


# ================================================================
# 3. JSON / CHECKPOINT HELPERS
# ================================================================

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


# ================================================================
# 4. METRICS
# ================================================================

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


# ================================================================
# 5. LOSS FUNCTIONS
# ================================================================

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


# ================================================================
# 6. LOGGING & PLOTTING
# ================================================================

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


# ================================================================
# 7. PLATEAU STOPPER
# ================================================================

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


# ================================================================
# 8. AUGMENTED DATASET WRAPPER
# ================================================================

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
