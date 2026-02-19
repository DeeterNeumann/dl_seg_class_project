"""
dh_train_immune_boost.py

Shared SMP encoder + shared Unet decoder + 2 heads:
  - semantic head: 5 classes (MoNuSAC nucleus typing: 0..4)
  - ternary head:  3 classes (0=bg, 1=inside, 2=boundary)

This script:
  - trains BOTH heads (semantic + ternary)
  - supports systematic experiment runs (lambda_ter tuning, seed sweeps, fixed-budget mode)
  - prints epoch summaries with BOTH semantic + ternary metrics
  - writes metrics.csv + history.json
  - saves a single dashboard.png at the end
  - uses immune-aware WeightedRandomSampler logic

Lightning-ready:
  - Robust REPO_ROOT anchoring so assets/scripts resolve regardless of CWD
  - DATA_ROOT / TRAIN_MANIFEST / VAL_MANIFEST / BASE_DIR / OUT_ROOT / NUM_WORKERS via env vars
  - Default OUT_ROOT = "<repo>/outputs"

NEW:
  - Optional grid runner mode:
      --grid
      --grid_seeds 1337,2021,7
      --grid_lambdas 0.25,0.5,1.0,2.0

CRITICAL PATH FIXES:
  - Auto-detect DATA_ROOT (prefers /team if present; falls back to /teamspace/...).
  - Auto-resolve BASE_DIR by testing candidate roots against manifest paths.
  - Hard preflight checks BEFORE spawning DataLoader workers (so you don't waste grid runs).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
import random
import traceback

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

import pandas as pd  # <-- needed for fast manifest preflight

import segmentation_models_pytorch as smp
from segmentation_models_pytorch.base import SegmentationHead


# -----------------------------
# Repo root + robust imports
# -----------------------------
THIS_FILE = Path(__file__).resolve()
if THIS_FILE.parent.name == "scripts":
    REPO_ROOT = THIS_FILE.parent.parent
else:
    REPO_ROOT = THIS_FILE.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.export_manifest_dataset import ExportManifestDataset, ExportManifestConfig  # noqa: E402


# -----------------------------
# CLI args
# -----------------------------
def _parse_csv_ints(s: str) -> list[int]:
    s = (s or "").strip()
    if not s:
        return []
    out: list[int] = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.append(int(tok))
    return out


def _parse_csv_floats(s: str) -> list[float]:
    s = (s or "").strip()
    if not s:
        return []
    out: list[float] = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.append(float(tok))
    return out


def parse_args():
    p = argparse.ArgumentParser()

    # ---- Grid runner knobs ----
    p.add_argument("--grid", action="store_true", help="Enable grid-runner (loops seeds x lambdas inside Python).")
    p.add_argument("--grid_seeds", type=str, default="", help="Comma-separated seeds, e.g. 1337,2021,7")
    p.add_argument("--grid_lambdas", type=str, default="", help="Comma-separated lambdas, e.g. 0.25,0.5,1.0,2.0")
    p.add_argument("--grid_stop_on_fail", action="store_true", help="Stop grid immediately if a run fails.")
    p.add_argument("--grid_continue_on_fail", action="store_true", help="Continue grid if a run fails (default).")

    # Core experiment knobs
    p.add_argument("--lambda_ter", type=float, default=1.0, help="Weight for ternary CE loss in total loss.")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--run_prefix", type=str, default="DUALHEAD")
    p.add_argument("--notes", type=str, default="")

    # Fixed-budget mode for clean comparisons
    p.add_argument(
        "--fixed_epochs",
        type=int,
        default=0,
        help="If >0, train exactly this many epochs (disables early-stop).",
    )

    # Common training knobs
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--max_epochs", type=int, default=1200)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--preview_every_steps", type=int, default=200)

    # Monitor/selection controls
    p.add_argument("--monitor_key", type=str, default="loss_total")
    p.add_argument("--monitor_mode", type=str, default="min", choices=["min", "max"])
    p.add_argument("--selection_key", type=str, default="ter_dice_macro_boundary")
    p.add_argument("--selection_mode", type=str, default="max", choices=["min", "max"])

    # Optional: override encoder
    p.add_argument("--encoder", type=str, default="resnet34")
    p.add_argument("--encoder_weights", type=str, default="imagenet")

    # Debug/preflight knobs
    p.add_argument("--preflight_rows", type=int, default=200, help="How many manifest rows to sample for path checks.")
    p.add_argument("--preflight_fail_fast", action="store_true", help="If any missing file in sampled rows => fail fast.")

    return p.parse_args()


def maybe_get_git_commit() -> str | None:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT), stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return None


def build_run_name(prefix: str, encoder: str, lam: float, seed: int) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    lam_s = f"{lam:.3g}".replace(".", "p")
    return f"{prefix}__{encoder}__lamTer{lam_s}__seed{seed}__{ts}"


# -----------------------------
# Env + path helpers
# -----------------------------
def env_str(key: str, default: str) -> str:
    v = os.environ.get(key, "").strip()
    return v if v else default


def _candidate_data_roots() -> list[Path]:
    """
    Prefer the location that historically worked for you (/team),
    but still support Lightning Studio layout (/teamspace/...).
    """
    cands: list[Path] = []
    # explicit env wins (but still validated)
    v = os.environ.get("DATA_ROOT", "").strip()
    if v:
        cands.append(Path(v).resolve())

    # common known roots
    cands.extend(
        [
            Path("/team").resolve(),
            Path("/teamspace/studios/this_studio/data/monusac_clean").resolve(),
            Path("/teamspace/studios/this_studio/data/monusac").resolve(),
            (REPO_ROOT / "data" / "monusac").resolve(),
        ]
    )

    # de-dupe preserving order
    seen = set()
    out: list[Path] = []
    for p in cands:
        if str(p) in seen:
            continue
        seen.add(str(p))
        out.append(p)
    return out


def _default_data_root() -> Path:
    """
    Pick the first existing candidate that *looks like* it contains MoNuSAC_outputs.
    """
    for root in _candidate_data_roots():
        if root.exists() and (root / "MoNuSAC_outputs").exists():
            return root
    # fallback: first existing root
    for root in _candidate_data_roots():
        if root.exists():
            return root
    # last resort: repo-relative
    return (REPO_ROOT / "data" / "monusac").resolve()


def _resolve_manifest_path(env_key: str, default_path: Path) -> Path:
    v = os.environ.get(env_key, "").strip()
    if v:
        p = Path(v)
        if not p.is_absolute():
            p = (REPO_ROOT / p).resolve()
        else:
            p = p.resolve()
        return p
    return default_path.resolve()


def _join_base(base_dir: Path, rel_or_abs: str) -> Path:
    p = Path(str(rel_or_abs))
    if p.is_absolute():
        return p
    return (base_dir / p).resolve()


def _manifest_required_cols(df: pd.DataFrame) -> list[str]:
    # what ExportManifestDataset uses (your traceback shows rgb_path specifically)
    req = ["rgb_path", "sem_gt_path", "ter_gt_path"]
    return [c for c in req if c not in df.columns]


def _preflight_manifest_paths(
    manifest_path: Path,
    base_candidates: list[Path],
    sample_rows: int = 200,
    fail_fast: bool = False,
) -> tuple[Path, dict[str, Any]]:
    """
    Try candidate base dirs and pick the one that yields the highest existence rate
    for rgb/sem/ter paths in a sampled subset of rows.
    """
    df = pd.read_csv(manifest_path)
    missing_cols = _manifest_required_cols(df)
    if missing_cols:
        raise ValueError(f"Manifest missing columns {missing_cols}: {manifest_path}")

    n = min(len(df), max(1, int(sample_rows)))
    probe = df.head(n)

    scored: list[tuple[float, Path, dict[str, Any]]] = []

    for base in base_candidates:
        if not base.exists():
            continue

        ok_rgb = 0
        ok_sem = 0
        ok_ter = 0
        first_missing: dict[str, str] = {}

        for i in range(n):
            rgb = _join_base(base, probe.loc[i, "rgb_path"])
            sem = _join_base(base, probe.loc[i, "sem_gt_path"])
            ter = _join_base(base, probe.loc[i, "ter_gt_path"])

            r_ok = rgb.exists()
            s_ok = sem.exists()
            t_ok = ter.exists()

            ok_rgb += int(r_ok)
            ok_sem += int(s_ok)
            ok_ter += int(t_ok)

            if fail_fast and (not (r_ok and s_ok and t_ok)):
                if not r_ok and "rgb" not in first_missing:
                    first_missing["rgb"] = str(rgb)
                if not s_ok and "sem" not in first_missing:
                    first_missing["sem"] = str(sem)
                if not t_ok and "ter" not in first_missing:
                    first_missing["ter"] = str(ter)
                break

            if not r_ok and "rgb" not in first_missing:
                first_missing["rgb"] = str(rgb)
            if not s_ok and "sem" not in first_missing:
                first_missing["sem"] = str(sem)
            if not t_ok and "ter" not in first_missing:
                first_missing["ter"] = str(ter)

        # score = mean existence across the three required files
        score = (ok_rgb + ok_sem + ok_ter) / (3.0 * n)
        details = {
            "base_dir": str(base),
            "sample_rows": int(n),
            "rgb_ok": int(ok_rgb),
            "sem_ok": int(ok_sem),
            "ter_ok": int(ok_ter),
            "score": float(score),
            "first_missing": first_missing,
        }
        scored.append((score, base, details))

    if not scored:
        raise RuntimeError(
            f"Preflight could not test any base_dir candidates (none exist). "
            f"manifest={manifest_path}"
        )

    scored.sort(key=lambda x: x[0], reverse=True)
    best_score, best_base, best_details = scored[0]

    # If best score is poor, raise with strong diagnostics
    if best_score < 0.95:
        top = scored[:5]
        msg_lines = [
            f"[PATH PREFLIGHT] FAILED to find a base_dir with >=95% path existence.",
            f"manifest: {manifest_path}",
            f"sample_rows: {n}",
            "Top candidates:",
        ]
        for s, b, det in top:
            msg_lines.append(
                f"  - score={s:.3f} base_dir={b} "
                f"(rgb_ok={det['rgb_ok']}/{n}, sem_ok={det['sem_ok']}/{n}, ter_ok={det['ter_ok']}/{n})"
            )
            fm = det.get("first_missing") or {}
            if fm:
                msg_lines.append(f"      first_missing: {fm}")
        msg_lines.append(
            "Fix: set DATA_ROOT/BASE_DIR/TRAIN_MANIFEST/VAL_MANIFEST env vars to the correct location, "
            "or ensure the export_patches files exist under the chosen root."
        )
        raise FileNotFoundError("\n".join(msg_lines))

    return best_base, best_details


# -----------------------------
# Globals / constants
# -----------------------------
SEMANTIC_CLASS_NAMES = {
    0: "background",
    1: "epithelial",
    2: "lymphocyte",
    3: "neutrophil",
    4: "macrophage",
}
SEM_CLASSES = (0, 1, 2, 3, 4)

TER_CLASS_NAMES = {0: "bg", 1: "inside", 2: "boundary"}
TER_CLASSES = (0, 1, 2)

SEM_WEIGHTS_JSON = (REPO_ROOT / "assets" / "class_weights.json").resolve()
if not SEM_WEIGHTS_JSON.exists():
    raise FileNotFoundError(f"Missing assets file: {SEM_WEIGHTS_JSON}")

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
        alias_path.symlink_to(target_path.name)
    except OSError:
        import shutil
        shutil.copy2(target_path, alias_path)


def safe_float(x, default=None):
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


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


def _dice_micro_guarded(inter: torch.Tensor, ps: torch.Tensor, gs: torch.Tensor, eps: float = 1e-6) -> float:
    gs_i = float(gs.item())
    ps_i = float(ps.item())
    if gs_i == 0.0 and ps_i == 0.0:
        return float("nan")
    if gs_i == 0.0 and ps_i > 0.0:
        return 0.0
    return float(_dice_from_stats(inter, ps, gs, eps).item())


def _iou_micro_guarded(inter: torch.Tensor, union: torch.Tensor, eps: float = 1e-6) -> float:
    if float(union.item()) == 0.0:
        return float("nan")
    return float(((inter + eps) / (union + eps)).item())


def unpack_batch(batch):
    if isinstance(batch, (list, tuple)):
        if len(batch) < 3:
            raise ValueError(f"Expected at least 3 items (x, sem, ter), got {len(batch)}")
        return batch[0], batch[1], batch[2]
    raise TypeError(f"Batch must be tuple/list, got {type(batch)}")


@torch.no_grad()
def semantic_dice_macro_per_class(
    sem_logits: torch.Tensor,
    sem_gt: torch.Tensor,
    classes=SEM_CLASSES,
    fg_only: bool = False,
    eps: float = 1e-6,
) -> dict[int, float]:
    pred = torch.argmax(sem_logits, dim=1)
    region = (sem_gt > 0) if fg_only else torch.ones_like(sem_gt, dtype=torch.bool)

    dice_macro: dict[int, float] = {}
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
    return dice_macro


@torch.no_grad()
def semantic_iou_all(sem_logits: torch.Tensor, sem_gt: torch.Tensor, classes=SEM_CLASSES, eps: float = 1e-6):
    pred = torch.argmax(sem_logits, dim=1)
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
        per_img_macro.append(torch.stack(ious).mean() if len(ious) > 0 else torch.tensor(0.0, device=pred.device))

        for c in classes:
            gt_c = (sem_gt[b] == c)
            pred_c = (pred[b] == c)
            inter_c[c] += (pred_c & gt_c).sum().float()
            union_c[c] += (pred_c | gt_c).sum().float()

    miou_macro_per_img = torch.stack(per_img_macro)

    total_inter = torch.zeros((), device=pred.device)
    total_union = torch.zeros((), device=pred.device)
    iou_by_class_micro: dict[int, float] = {}
    for c in classes:
        total_inter += inter_c[c]
        total_union += union_c[c]
        iou_by_class_micro[c] = _iou_micro_guarded(inter_c[c], union_c[c], eps=eps)

    miou_micro = float(((total_inter + eps) / (total_union + eps)).item()) if float(total_union.item()) > 0 else float("nan")
    return miou_macro_per_img, miou_micro, iou_by_class_micro


@torch.no_grad()
def ternary_dice_macro_per_class(ter_logits: torch.Tensor, ter_gt: torch.Tensor, classes=TER_CLASSES, eps: float = 1e-6):
    pred = torch.argmax(ter_logits, dim=1)
    dice_macro: dict[int, float] = {}
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
    return dice_macro


@torch.no_grad()
def ternary_iou_all(ter_logits: torch.Tensor, ter_gt: torch.Tensor, classes=TER_CLASSES, eps: float = 1e-6):
    pred = torch.argmax(ter_logits, dim=1)
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
        per_img_macro.append(torch.stack(ious).mean() if len(ious) > 0 else torch.tensor(0.0, device=pred.device))

        for c in classes:
            gt_c = (ter_gt[b] == c)
            pred_c = (pred[b] == c)
            inter_c[c] += (pred_c & gt_c).sum().float()
            union_c[c] += (pred_c | gt_c).sum().float()

    miou_macro_per_img = torch.stack(per_img_macro)

    total_inter = torch.zeros((), device=pred.device)
    total_union = torch.zeros((), device=pred.device)
    iou_by_class_micro: dict[int, float] = {}
    for c in classes:
        total_inter += inter_c[c]
        total_union += union_c[c]
        iou_by_class_micro[c] = _iou_micro_guarded(inter_c[c], union_c[c], eps=eps)

    miou_micro = float(((total_inter + eps) / (total_union + eps)).item()) if float(total_union.item()) > 0 else float("nan")
    return miou_macro_per_img, miou_micro, iou_by_class_micro


# -----------------------------
# Losses
# -----------------------------
def multiclass_soft_dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    ignore_index: int | None = None,
    class_weights: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    probs = F.softmax(logits, dim=1)

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
    dice = (2.0 * inter + eps) / (den + eps)
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
        v = d[c]
        if isinstance(v, float) and np.isnan(v):
            parts.append(f"{names.get(c, str(c))}:nan")
        else:
            parts.append(f"{names.get(c, str(c))}:{float(v):.4f}")
    return " ".join(parts)


def _history_append(history: dict[str, list], row: dict, fieldnames: list[str]) -> None:
    for k in fieldnames:
        history.setdefault(k, [])
        history[k].append(row.get(k, None))


def _plot_dashboard(plots_dir: Path, history: dict[str, list]):
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
                fv = float(v)
                if np.isnan(fv):
                    continue
                xs.append(int(e))
                ys.append(fv)
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
        ["train_loss_total", "val_loss_total", "train_loss_sem", "val_loss_sem", "train_loss_ter", "val_loss_ter"],
        "Loss (total + per-head)",
        "loss",
    )
    _plot_lines(axs[0, 1], ["miou_all_macro", "miou_all_micro"], "Semantic mIoU", "IoU")

    ax = axs[1, 0]
    any_plotted = False
    for c in SEM_CLASSES:
        nm = SEMANTIC_CLASS_NAMES[c]
        xs, ys = _series(f"dice_macro_{nm}")
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
        xs, ys = _series(f"iou_{nm}")
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

    plt.tight_layout()
    out_path = plots_dir / "dashboard.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[PLOT] {out_path}")


# -----------------------------
# Plateau stopper
# -----------------------------
@dataclass
class PlateauStopper:
    patience: int = 80
    min_delta: float = 5e-4
    min_epochs: int = 120
    mode: str = "min"
    ema_alpha: float = 0.3

    best: float = float("inf")
    bad_epochs: int = 0
    ema: float | None = None

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
        info = {"raw": float(value), "smoothed": float(v), "best": float(self.best), "bad_epochs": int(self.bad_epochs), "patience": int(self.patience)}
        return should_stop, info


# -----------------------------
# Model: shared encoder+decoder, two heads
# -----------------------------
class SharedUnetTwoHead(nn.Module):
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
            classes=1,
            activation=None,
            decoder_channels=self.decoder_channels,
        )
        dec_out_ch = self.decoder_channels[-1]
        self.sem_head = SegmentationHead(in_channels=dec_out_ch, out_channels=sem_classes, activation=activation, kernel_size=3)
        self.ter_head = SegmentationHead(in_channels=dec_out_ch, out_channels=ter_classes, activation=activation, kernel_size=3)

    def forward(self, x: torch.Tensor):
        feats = self.base.encoder(x)
        try:
            dec = self.base.decoder(*feats)
        except TypeError:
            dec = self.base.decoder(feats)
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
):
    model.eval()
    eps = 1e-6

    sem_loss_sum = 0.0
    ter_loss_sum = 0.0
    total_loss_sum = 0.0
    n = 0

    sem_dice_macro_sum = {c: 0.0 for c in SEM_CLASSES}
    ter_dice_macro_sum = {c: 0.0 for c in TER_CLASSES}
    n_batches = 0

    sem_inter_c = {c: torch.zeros((), device=device) for c in SEM_CLASSES}
    sem_ps_c = {c: torch.zeros((), device=device) for c in SEM_CLASSES}
    sem_gs_c = {c: torch.zeros((), device=device) for c in SEM_CLASSES}
    sem_union_c = {c: torch.zeros((), device=device) for c in SEM_CLASSES}

    ter_inter_c = {c: torch.zeros((), device=device) for c in TER_CLASSES}
    ter_ps_c = {c: torch.zeros((), device=device) for c in TER_CLASSES}
    ter_gs_c = {c: torch.zeros((), device=device) for c in TER_CLASSES}
    ter_union_c = {c: torch.zeros((), device=device) for c in TER_CLASSES}

    total_inter_all = torch.zeros((), device=device)
    total_union_all = torch.zeros((), device=device)

    miou_all_macro_img_sum = 0.0
    n_all_imgs = 0
    ter_miou_macro_img_sum = 0.0
    n_ter_imgs = 0

    for batch in dl:
        x, sem, ter = unpack_batch(batch)
        x = x.to(device)
        sem = sem.to(device)
        ter = ter.to(device)

        sem_logits, ter_logits = model(x)

        sem_loss, _ = semantic_loss_dice_only(sem_logits, sem, dice_w=sem_dice_w, ignore_index=ignore_index_sem)
        if ignore_index_ter is None:
            ter_loss = F.cross_entropy(ter_logits, ter, weight=ter_ce_w, reduction="mean")
        else:
            ter_loss = F.cross_entropy(ter_logits, ter, weight=ter_ce_w, reduction="mean", ignore_index=ignore_index_ter)

        total = sem_loss + float(ter_loss_weight) * ter_loss

        bs = x.size(0)
        sem_loss_sum += float(sem_loss.item()) * bs
        ter_loss_sum += float(ter_loss.item()) * bs
        total_loss_sum += float(total.item()) * bs
        n += bs

        miou_per_img, _miou_micro_unused, _ = semantic_iou_all(sem_logits, sem, classes=SEM_CLASSES, eps=eps)
        miou_all_macro_img_sum += float(miou_per_img.sum().item())
        n_all_imgs += int(miou_per_img.numel())

        ter_miou_per_img, _ter_miou_micro_unused, _ = ternary_iou_all(ter_logits, ter, classes=TER_CLASSES, eps=eps)
        ter_miou_macro_img_sum += float(ter_miou_per_img.sum().item())
        n_ter_imgs += int(ter_miou_per_img.numel())

        dm = semantic_dice_macro_per_class(sem_logits, sem, classes=SEM_CLASSES, fg_only=dice_fg_only, eps=eps)
        tdm = ternary_dice_macro_per_class(ter_logits, ter, classes=TER_CLASSES, eps=eps)
        for c in SEM_CLASSES:
            sem_dice_macro_sum[c] += float(dm[c])
        for c in TER_CLASSES:
            ter_dice_macro_sum[c] += float(tdm[c])
        n_batches += 1

        sem_pred = torch.argmax(sem_logits, dim=1)
        for c in SEM_CLASSES:
            gt_c = (sem == c)
            pr_c = (sem_pred == c)
            inter = (gt_c & pr_c).sum().float()
            ps = pr_c.sum().float()
            gs = gt_c.sum().float()
            union = (gt_c | pr_c).sum().float()
            sem_inter_c[c] += inter
            sem_ps_c[c] += ps
            sem_gs_c[c] += gs
            sem_union_c[c] += union
            total_inter_all += inter
            total_union_all += union

        ter_pred = torch.argmax(ter_logits, dim=1)
        for c in TER_CLASSES:
            gt_c = (ter == c)
            pr_c = (ter_pred == c)
            inter = (gt_c & pr_c).sum().float()
            ps = pr_c.sum().float()
            gs = gt_c.sum().float()
            union = (gt_c | pr_c).sum().float()
            ter_inter_c[c] += inter
            ter_ps_c[c] += ps
            ter_gs_c[c] += gs
            ter_union_c[c] += union

    out: dict[str, Any] = {}
    out["loss_sem"] = sem_loss_sum / max(1, n)
    out["loss_ter"] = ter_loss_sum / max(1, n)
    out["loss_total"] = total_loss_sum / max(1, n)

    out["miou_all_macro"] = miou_all_macro_img_sum / max(1, n_all_imgs)
    out["miou_all_micro"] = float(((total_inter_all + eps) / (total_union_all + eps)).item()) if float(total_union_all.item()) > 0 else float("nan")

    for c in SEM_CLASSES:
        nm = SEMANTIC_CLASS_NAMES[c]
        out[f"iou_{nm}"] = _iou_micro_guarded(sem_inter_c[c], sem_union_c[c], eps=eps)
        out[f"dice_macro_{nm}"] = sem_dice_macro_sum[c] / max(1, n_batches)
        out[f"dice_micro_{nm}"] = _dice_micro_guarded(sem_inter_c[c], sem_ps_c[c], sem_gs_c[c], eps=eps)
        out[f"gt_pixels_{nm}"] = int(sem_gs_c[c].item())
        out[f"pred_pixels_{nm}"] = int(sem_ps_c[c].item())

    out["dice_macro_by_class"] = {c: sem_dice_macro_sum[c] / max(1, n_batches) for c in SEM_CLASSES}
    out["dice_micro_by_class"] = {c: _dice_micro_guarded(sem_inter_c[c], sem_ps_c[c], sem_gs_c[c], eps=eps) for c in SEM_CLASSES}

    ter_total_inter = torch.zeros((), device=device)
    ter_total_union = torch.zeros((), device=device)
    for c in TER_CLASSES:
        ter_total_inter += ter_inter_c[c]
        ter_total_union += ter_union_c[c]

    out["ter_miou_macro"] = ter_miou_macro_img_sum / max(1, n_ter_imgs)
    out["ter_miou_micro"] = float(((ter_total_inter + eps) / (ter_total_union + eps)).item()) if float(ter_total_union.item()) > 0 else float("nan")

    for c in TER_CLASSES:
        nm = TER_CLASS_NAMES[c]
        out[f"ter_iou_{nm}"] = _iou_micro_guarded(ter_inter_c[c], ter_union_c[c], eps=eps)
        out[f"ter_dice_macro_{nm}"] = ter_dice_macro_sum[c] / max(1, n_batches)
        out[f"ter_dice_micro_{nm}"] = _dice_micro_guarded(ter_inter_c[c], ter_ps_c[c], ter_gs_c[c], eps=eps)
        out[f"ter_gt_pixels_{nm}"] = int(ter_gs_c[c].item())
        out[f"ter_pred_pixels_{nm}"] = int(ter_ps_c[c].item())

    out["ter_dice_macro_by_class"] = {c: ter_dice_macro_sum[c] / max(1, n_batches) for c in TER_CLASSES}
    out["ter_dice_micro_by_class"] = {c: _dice_micro_guarded(ter_inter_c[c], ter_ps_c[c], ter_gs_c[c], eps=eps) for c in TER_CLASSES}

    return out


# -----------------------------
# Single-run main logic
# -----------------------------
def run_single(args: argparse.Namespace) -> dict[str, Any]:
    seed_everything(args.seed)

    batch_size = int(args.batch_size)
    lr = float(args.lr)
    max_epochs = int(args.max_epochs)
    weight_decay = float(args.weight_decay)
    TER_LOSS_WEIGHT = float(args.lambda_ter)

    MONITOR_KEY = str(args.monitor_key)
    MONITOR_MODE = str(args.monitor_mode)
    SELECTION_KEY = str(args.selection_key)
    SELECTION_MODE = str(args.selection_mode)

    PREVIEW_EVERY_STEPS = int(args.preview_every_steps)

    ENCODER = str(args.encoder)
    ENCODER_WEIGHTS = str(args.encoder_weights) if args.encoder_weights else None

    # ---- ROOTS & MANIFESTS (robust) ----
    data_root_default = _default_data_root()
    DATA_ROOT = Path(env_str("DATA_ROOT", str(data_root_default))).resolve()

    train_manifest_default = DATA_ROOT / "MoNuSAC_outputs/export_patches/train_P256_S128_fg0.01/export_manifest_immune_counts.csv"
    val_manifest_default = DATA_ROOT / "MoNuSAC_outputs/export_patches/val_P256_S128_fg0.01/export_manifest_immune_counts.csv"
    train_manifest_path = _resolve_manifest_path("TRAIN_MANIFEST", train_manifest_default)
    val_manifest_path = _resolve_manifest_path("VAL_MANIFEST", val_manifest_default)

    # Base candidates include: BASE_DIR env if set, DATA_ROOT, parent of manifest, /team, /teamspace/... and REPO_ROOT
    base_candidates: list[Path] = []
    base_env = os.environ.get("BASE_DIR", "").strip()
    if base_env:
        base_candidates.append(Path(base_env).resolve())
    base_candidates.extend([DATA_ROOT, train_manifest_path.parent.parent.parent.parent, val_manifest_path.parent.parent.parent.parent])
    base_candidates.extend(_candidate_data_roots())
    base_candidates.append(REPO_ROOT)

    # De-dupe
    seen = set()
    deduped: list[Path] = []
    for p in base_candidates:
        pr = p.resolve()
        if str(pr) in seen:
            continue
        seen.add(str(pr))
        deduped.append(pr)
    base_candidates = deduped

    # Preflight to find a working BASE_DIR that matches manifest paths
    base_dir_train, train_pf = _preflight_manifest_paths(
        train_manifest_path,
        base_candidates,
        sample_rows=int(args.preflight_rows),
        fail_fast=bool(args.preflight_fail_fast),
    )
    base_dir_val, val_pf = _preflight_manifest_paths(
        val_manifest_path,
        base_candidates,
        sample_rows=int(args.preflight_rows),
        fail_fast=bool(args.preflight_fail_fast),
    )

    # If they disagree, choose the better score (they should both be ~1.0 if correct)
    base_dir = base_dir_train
    if val_pf["score"] > train_pf["score"]:
        base_dir = base_dir_val

    device = get_device()

    out_root = Path(env_str("OUT_ROOT", str(REPO_ROOT / "outputs")))
    out_root = out_root if out_root.is_absolute() else (REPO_ROOT / out_root)
    out_root = out_root.resolve()

    run_name = build_run_name(args.run_prefix, ENCODER, TER_LOSS_WEIGHT, args.seed)

    ckpt_dir = out_root / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_best_path = ckpt_dir / f"{run_name}__best.pt"
    ckpt_last_path = ckpt_dir / f"{run_name}__last.pt"

    runs_dir = out_root / "runs" / run_name
    runs_dir.mkdir(parents=True, exist_ok=True)

    metrics_csv = runs_dir / "metrics.csv"
    config_json = runs_dir / "config.json"

    if metrics_csv.exists():
        metrics_csv.unlink()

    print(f"\n[RUN] {run_name}")
    print(f"[RUN] device={device} seed={args.seed} lambda_ter={TER_LOSS_WEIGHT} fixed_epochs={int(args.fixed_epochs)}")
    print(f"[PATH] DATA_ROOT={DATA_ROOT}")
    print(f"[PATH] TRAIN_MANIFEST={train_manifest_path}")
    print(f"[PATH] VAL_MANIFEST={val_manifest_path}")
    print(f"[PATH] BASE_DIR(resolved)={base_dir}")
    print(f"[PREFLIGHT] train score={train_pf['score']:.3f} val score={val_pf['score']:.3f}")
    print(f"[OUT] out_root={out_root}")

    torch.use_deterministic_algorithms(True, warn_only=True)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    num_workers = int(env_str("NUM_WORKERS", "4"))
    pin_memory = (device.type == "cuda")
    persistent_workers = (num_workers > 0)

    train_data_config = ExportManifestConfig(csv_path=train_manifest_path, base_dir=base_dir)
    val_data_config = ExportManifestConfig(csv_path=val_manifest_path, base_dir=base_dir)
    train_ds = ExportManifestDataset(train_data_config)
    val_ds = ExportManifestDataset(val_data_config)

    # ---- immune-aware sampler ----
    req_cols = ["px_lymphocyte", "px_neutrophil", "px_macrophage"]
    missing = [c for c in req_cols if c not in train_ds.df.columns]
    if missing:
        raise ValueError(f"Train manifest missing columns: {missing}. Use export_manifest_immune_counts.csv")

    px_lym = train_ds.df["px_lymphocyte"].to_numpy(dtype=np.int64)
    px_neu = train_ds.df["px_neutrophil"].to_numpy(dtype=np.int64)
    px_mac = train_ds.df["px_macrophage"].to_numpy(dtype=np.int64)

    T_LYM, T_NEU, T_MAC = 25, 10, 10
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
        f"[SAMPLER] has_lym={has_lym.mean():.2%} has_neu={has_neu.mean():.2%} has_mac={has_mac.mean():.2%} "
        f"(T_LYM={T_LYM}, T_NEU={T_NEU}, T_MAC={T_MAC})"
    )

    train_sampler = WeightedRandomSampler(weights=weights.tolist(), num_samples=len(weights), replacement=True)

    model = SharedUnetTwoHead(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        in_channels=3,
        decoder_channels=(256, 128, 64, 32, 16),
        sem_classes=5,
        ter_classes=3,
        activation=None,
    ).to(device)

    config = {
        "run_name": run_name,
        "run_prefix": str(args.run_prefix),
        "notes": str(args.notes),
        "git_commit": maybe_get_git_commit(),
        "seed": int(args.seed),
        "lambda_ter": float(TER_LOSS_WEIGHT),
        "fixed_epochs": int(args.fixed_epochs),
        "device": str(device),
        "batch_size": int(batch_size),
        "lr_init": float(lr),
        "weight_decay": float(weight_decay),
        "max_epochs": int(max_epochs),
        "preview_every_steps": int(PREVIEW_EVERY_STEPS),
        "encoder": ENCODER,
        "encoder_weights": ENCODER_WEIGHTS,
        "monitor_key": MONITOR_KEY,
        "monitor_mode": MONITOR_MODE,
        "selection_key": SELECTION_KEY,
        "selection_mode": SELECTION_MODE,
        "data_root": str(DATA_ROOT),
        "train_manifest": str(train_manifest_path),
        "val_manifest": str(val_manifest_path),
        "base_dir": str(base_dir),
        "out_root": str(out_root),
        "num_workers": int(num_workers),
        "pin_memory": bool(pin_memory),
        "preflight": {"train": train_pf, "val": val_pf},
    }
    config_json.write_text(json.dumps(config, indent=2))

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt,
        mode="min" if MONITOR_MODE == "min" else "max",
        factor=0.5,
        patience=20,
        threshold=5e-4,
    )

    stopper = PlateauStopper(
        patience=240,
        min_delta=5e-4,
        min_epochs=120,
        mode=MONITOR_MODE,
        ema_alpha=0.3,
    )

    metric_fields = [
        "epoch", "global_step",
        "train_loss_sem", "train_loss_ter", "train_loss_total",
        "val_loss_sem", "val_loss_ter", "val_loss_total",
        "miou_all_macro", "miou_all_micro",
        "ter_miou_macro", "ter_miou_micro",
        "monitor_value", "selection_value",
        "should_stop", "lr",
        "lambda_ter", "seed",
    ]
    for c in SEM_CLASSES:
        nm = SEMANTIC_CLASS_NAMES[c]
        metric_fields += [f"iou_{nm}", f"dice_macro_{nm}", f"dice_micro_{nm}", f"gt_pixels_{nm}", f"pred_pixels_{nm}"]
    for c in TER_CLASSES:
        nm = TER_CLASS_NAMES[c]
        metric_fields += [f"ter_iou_{nm}", f"ter_dice_macro_{nm}", f"ter_dice_micro_{nm}", f"ter_gt_pixels_{nm}", f"ter_pred_pixels_{nm}"]

    history: dict[str, list] = {k: [] for k in metric_fields}

    best_selection = float("-inf") if SELECTION_MODE == "max" else float("inf")
    global_step = 0
    last_row: dict[str, Any] | None = None
    status = "running"
    error_msg: str | None = None
    error_tb: str | None = None

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

            model.train()
            sem_sum = 0.0
            ter_sum = 0.0
            tot_sum = 0.0
            n_seen = 0

            for _, batch in enumerate(train_dl, start=1):
                x, sem, ter = unpack_batch(batch)
                x = x.to(device, non_blocking=True)
                sem = sem.to(device, non_blocking=True)
                ter = ter.to(device, non_blocking=True)

                sem_logits, ter_logits = model(x)

                sem_loss, _ = semantic_loss_dice_only(sem_logits, sem, dice_w=None, ignore_index=None)
                ter_loss = F.cross_entropy(ter_logits, ter, weight=None, reduction="mean")

                loss = sem_loss + float(TER_LOSS_WEIGHT) * ter_loss

                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

                bs = x.size(0)
                sem_sum += float(sem_loss.item()) * bs
                ter_sum += float(ter_loss.item()) * bs
                tot_sum += float(loss.item()) * bs
                n_seen += bs
                global_step += 1

            train_loss_sem = sem_sum / max(1, n_seen)
            train_loss_ter = ter_sum / max(1, n_seen)
            train_loss_total = tot_sum / max(1, n_seen)

            val_metrics = evaluate(
                model=model,
                dl=val_dl,
                device=device,
                sem_dice_w=None,
                ter_ce_w=None,
                ter_loss_weight=TER_LOSS_WEIGHT,
                ignore_index_sem=None,
                ignore_index_ter=None,
                dice_fg_only=False,
            )

            if MONITOR_KEY not in val_metrics:
                raise KeyError(f"MONITOR_KEY '{MONITOR_KEY}' not found in val_metrics.")
            if SELECTION_KEY not in val_metrics:
                raise KeyError(f"SELECTION_KEY '{SELECTION_KEY}' not found in val_metrics.")

            monitor_value = float(val_metrics[MONITOR_KEY]) if not np.isnan(float(val_metrics[MONITOR_KEY])) else float("inf")
            selection_value = float(val_metrics[SELECTION_KEY])

            prev_lr = opt.param_groups[0]["lr"]
            scheduler.step(monitor_value)
            lr_now = opt.param_groups[0]["lr"]
            if lr_now != prev_lr:
                print(f"[LR] {prev_lr:.3e} -> {lr_now:.3e}")

            if args.fixed_epochs and int(args.fixed_epochs) > 0:
                should_stop = False
            else:
                should_stop, _ = stopper.step(monitor_value, epoch)

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
                "should_stop": int(should_stop),
                "lr": lr_now,
                "lambda_ter": float(TER_LOSS_WEIGHT),
                "seed": int(args.seed),
            }
            for c in SEM_CLASSES:
                nm = SEMANTIC_CLASS_NAMES[c]
                row[f"iou_{nm}"] = float(val_metrics[f"iou_{nm}"]) if not np.isnan(float(val_metrics[f"iou_{nm}"])) else float("nan")
                row[f"dice_macro_{nm}"] = float(val_metrics[f"dice_macro_{nm}"])
                row[f"dice_micro_{nm}"] = float(val_metrics[f"dice_micro_{nm}"]) if not np.isnan(float(val_metrics[f"dice_micro_{nm}"])) else float("nan")
                row[f"gt_pixels_{nm}"] = int(val_metrics[f"gt_pixels_{nm}"])
                row[f"pred_pixels_{nm}"] = int(val_metrics[f"pred_pixels_{nm}"])
            for c in TER_CLASSES:
                nm = TER_CLASS_NAMES[c]
                row[f"ter_iou_{nm}"] = float(val_metrics[f"ter_iou_{nm}"]) if not np.isnan(float(val_metrics[f"ter_iou_{nm}"])) else float("nan")
                row[f"ter_dice_macro_{nm}"] = float(val_metrics[f"ter_dice_macro_{nm}"])
                row[f"ter_dice_micro_{nm}"] = float(val_metrics[f"ter_dice_micro_{nm}"]) if not np.isnan(float(val_metrics[f"ter_dice_micro_{nm}"])) else float("nan")
                row[f"ter_gt_pixels_{nm}"] = int(val_metrics[f"ter_gt_pixels_{nm}"])
                row[f"ter_pred_pixels_{nm}"] = int(val_metrics[f"ter_pred_pixels_{nm}"])

            last_row = dict(row)
            append_metrics_row(metrics_csv, metric_fields, row)
            _history_append(history, row, metric_fields)

            print(f"\n===== EPOCH {epoch} SUMMARY =====")
            print(f"train_total={train_loss_total:.4f} (sem={train_loss_sem:.4f}, ter={train_loss_ter:.4f})")
            print(f"val_total  ={val_metrics['loss_total']:.4f} (sem={val_metrics['loss_sem']:.4f}, ter={val_metrics['loss_ter']:.4f})")
            print(f"SEM: miou_macro={val_metrics['miou_all_macro']:.4f} miou_micro={val_metrics['miou_all_micro']:.4f}")
            print(f"TER: miou_macro={val_metrics['ter_miou_macro']:.4f} miou_micro={val_metrics['ter_miou_micro']:.4f}")
            print(f"TER Dice macro: {_fmt_per_class(val_metrics['ter_dice_macro_by_class'], TER_CLASS_NAMES)}")
            print(f"[MONITOR] {MONITOR_KEY}={monitor_value:.6f} | [SELECT] {SELECTION_KEY}={selection_value:.6f} | lr={lr_now:.3e}")

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
                    config=config,
                    extra_pt={"best_selection": float(best_selection)},
                    extra_json={"best_selection": float(best_selection), "val_metrics": val_metrics},
                )
                update_alias(ckpt_dir / "LATEST__best.pt", ckpt_best_path)
                update_alias(ckpt_dir / "LATEST__best.json", ckpt_best_path.with_suffix(".json"))
                print(f"[CKPT] BEST saved (selection={best_selection:.6f})")

            save_checkpoint(
                ckpt_last_path,
                model=model,
                optimizer=opt,
                scheduler=scheduler,
                epoch=epoch,
                global_step=global_step,
                config=config,
                extra_pt={"best_selection": float(best_selection)},
                extra_json={"best_selection": float(best_selection), "val_metrics": val_metrics},
            )
            update_alias(ckpt_dir / "LATEST__last.pt", ckpt_last_path)

            if args.fixed_epochs and int(args.fixed_epochs) > 0 and epoch >= int(args.fixed_epochs):
                print(f"\n[STOP] Fixed-budget reached epoch {int(args.fixed_epochs)}.")
                break
            if should_stop:
                print(f"\n[STOP] Plateau on {MONITOR_KEY} at epoch {epoch}.")
                break

        status = "ok"

    except Exception as e:
        status = "error"
        error_msg = str(e)
        error_tb = traceback.format_exc()

    finally:
        plots_dir = runs_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        _plot_dashboard(plots_dir, history)
        write_json(plots_dir / "history.json", history)

        run_result = {
            "status": status,
            "error": error_msg,
            "traceback": error_tb,
            "run_name": run_name,
            "runs_dir": str(runs_dir),
            "metrics_csv": str(metrics_csv),
            "config_json": str(config_json),
            "seed": int(args.seed),
            "lambda_ter": float(TER_LOSS_WEIGHT),
            "best_selection": safe_float(best_selection),
            "selection_key": str(SELECTION_KEY),
            "selection_mode": str(SELECTION_MODE),
            "monitor_key": str(MONITOR_KEY),
            "monitor_mode": str(MONITOR_MODE),
            "ckpt_best_path": str(ckpt_best_path),
            "ckpt_last_path": str(ckpt_last_path),
            "last_epoch": int(last_row["epoch"]) if last_row else None,
            "last_monitor_value": safe_float(last_row.get("monitor_value")) if last_row else None,
            "last_selection_value": safe_float(last_row.get("selection_value")) if last_row else None,
            "last_val_loss_total": safe_float(last_row.get("val_loss_total")) if last_row else None,
            "last_ter_dice_macro_boundary": safe_float(last_row.get("ter_dice_macro_boundary")) if last_row else None,
            "resolved_base_dir": str(base_dir),
            "preflight": {"train": train_pf, "val": val_pf},
        }
        write_json(runs_dir / "run_result.json", run_result)

        # concise end-of-run print
        print("\n" + "=" * 110)
        print(f"[RUN COMPLETE] status={status} run={run_name}")
        if status != "ok":
            print(f"[ERROR] {error_msg}")
        else:
            print(f"[BEST] selection={best_selection:.6f} key={SELECTION_KEY}")
            if last_row:
                print(f"[LAST] epoch={last_row['epoch']} val_loss_total={last_row['val_loss_total']:.4f} ter_boundary={last_row.get('ter_dice_macro_boundary')}")
        print(f"[ARTIFACTS] {runs_dir}")
        print("=" * 110 + "\n")

    return run_result


# -----------------------------
# Grid runner
# -----------------------------
def run_grid(args: argparse.Namespace) -> int:
    seeds = _parse_csv_ints(args.grid_seeds) if args.grid_seeds else []
    lams = _parse_csv_floats(args.grid_lambdas) if args.grid_lambdas else []

    if not seeds:
        seeds = [int(args.seed)]
    if not lams:
        lams = [float(args.lambda_ter)]

    stop_on_fail = bool(args.grid_stop_on_fail)
    if args.grid_continue_on_fail:
        stop_on_fail = False

    total = len(seeds) * len(lams)
    idx = 0
    failures: list[dict[str, Any]] = []
    grid_results: list[dict[str, Any]] = []
    grid_started_at = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"\n[GRID] {len(seeds)} seeds x {len(lams)} lambdas = {total} runs")
    print(f"[GRID] seeds={seeds}")
    print(f"[GRID] lambdas={lams}")
    print(f"[GRID] stop_on_fail={stop_on_fail}\n")

    aborted = False

    for seed in seeds:
        if aborted:
            break
        for lam in lams:
            idx += 1
            print("=" * 110)
            print(f"[GRID] RUN {idx}/{total} seed={seed} lambda_ter={lam}")
            print("=" * 110)

            child = argparse.Namespace(**vars(args))
            child.grid = False
            child.seed = int(seed)
            child.lambda_ter = float(lam)

            try:
                run_result = run_single(child)
                grid_results.append(run_result)

                if run_result.get("status") != "ok":
                    failures.append(
                        {
                            "seed": seed,
                            "lambda_ter": lam,
                            "error": run_result.get("error"),
                            "run_name": run_result.get("run_name"),
                            "runs_dir": run_result.get("runs_dir"),
                        }
                    )
                    print(f"[GRID] FAIL seed={seed} lambda_ter={lam} error={run_result.get('error')}")
                    if stop_on_fail:
                        print("[GRID] stopping due to --grid_stop_on_fail")
                        aborted = True

            except Exception as e:
                tb = traceback.format_exc()
                failures.append({"seed": seed, "lambda_ter": lam, "error": str(e), "traceback": tb})
                print(f"[GRID] EXCEPTION seed={seed} lambda_ter={lam}: {e}")
                if stop_on_fail:
                    print("[GRID] stopping due to --grid_stop_on_fail")
                    aborted = True

            finally:
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            if aborted:
                break

    out_root = Path(env_str("OUT_ROOT", str(REPO_ROOT / "outputs")))
    out_root = out_root if out_root.is_absolute() else (REPO_ROOT / out_root)
    out_root = out_root.resolve()

    grid_summary = {
        "grid_started_at": grid_started_at,
        "grid_finished_at": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "aborted_early": bool(aborted),
        "run_prefix": str(args.run_prefix),
        "encoder": str(args.encoder),
        "encoder_weights": str(args.encoder_weights),
        "fixed_epochs": int(args.fixed_epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "monitor_key": str(args.monitor_key),
        "monitor_mode": str(args.monitor_mode),
        "selection_key": str(args.selection_key),
        "selection_mode": str(args.selection_mode),
        "seeds": seeds,
        "lambdas": lams,
        "total_runs_planned": int(total),
        "total_runs_executed": int(len(grid_results)),
        "failures": failures,
        "results": grid_results,
    }

    summary_path = out_root / "grid_summary.json"
    write_json(summary_path, grid_summary)

    print("\n" + "=" * 110)
    print("[GRID] COMPLETE")
    if failures:
        print(f"[GRID] failures={len(failures)}/{total}")
        for f in failures:
            print(f"  - seed={f.get('seed')} lambda_ter={f.get('lambda_ter')} error={f.get('error')}")
        print(f"[GRID_SUMMARY] {summary_path}")
        print("=" * 110 + "\n")
        return 1

    print("[GRID] all runs succeeded")
    print(f"[GRID_SUMMARY] {summary_path}")
    print("=" * 110 + "\n")
    return 0


def main():
    args = parse_args()
    if args.grid:
        rc = run_grid(args)
        raise SystemExit(rc)
    else:
        run_result = run_single(args)
        raise SystemExit(0 if run_result.get("status") == "ok" else 1)


if __name__ == "__main__":
    main()