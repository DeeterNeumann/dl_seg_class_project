"""
train.py

Main training loop for the dual-head U-Net (semantic + ternary).

Usage:
    cd training/
    python train.py
"""

import json
from pathlib import Path
from typing import Any
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

import albumentations as A
from albumentations.pytorch import ToTensorV2

# Project imports
from utils import (
    # Constants
    IMAGENET_MEAN, IMAGENET_STD,
    PROJECT_ROOT, TRAINING_DIR, BASE_SEED,
    SEMANTIC_CLASS_NAMES, SEM_CLASSES,
    TER_CLASS_NAMES, TER_CLASSES,
    SEM_WEIGHTS_JSON, wobj,
    # Path helpers
    env_str,
    # Device & seeding
    get_device, seed_everything, seed_worker,
    # JSON / checkpoint
    write_json, save_checkpoint, update_alias,
    # Metrics
    semantic_dice_per_class, semantic_iou_all,
    ternary_dice_per_class, ternary_iou_all,
    unpack_batch,
    # Losses
    semantic_loss_dice_only, ternary_combined_loss,
    compute_ternary_ce_weights_from_manifest,
    # Logging
    append_metrics_row, _fmt_per_class, _history_append,
    save_preview_panel_dual, _plot_dashboard,
    # Early stopping & augmentation
    PlateauStopper, AugmentedDataset,
)
from model import SharedUnetTwoHead
from scripts.export_manifest_dataset import ExportManifestDataset, ExportManifestConfig


# ================================================================
# EVALUATION
# ================================================================

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


# ================================================================
# MAIN
# ================================================================

def main():
    seed_everything(BASE_SEED)

    # Hyperparams
    batch_size = 8
    lr = 3e-4
    max_epochs = 100
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

    # data
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

    # outputs
    default_out = Path(env_str("OUT_ROOT", str(TRAINING_DIR / "outputs")))
    out_root = default_out if default_out.is_absolute() else (TRAINING_DIR / default_out)
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

    # device + determinism
    device = get_device()
    print("device:", device)
    print("[PROJECT_ROOT]", PROJECT_ROOT)
    print("[OUT_ROOT]", out_root.resolve())
    print("[SEM_WEIGHTS_JSON]", SEM_WEIGHTS_JSON, "exists=", SEM_WEIGHTS_JSON.exists())

    torch.use_deterministic_algorithms(True, warn_only=True)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    # dataLoader performance knobs
    num_workers = int(env_str("NUM_WORKERS", "4"))
    pin_memory = (device.type == "cuda")
    persistent_workers = (num_workers > 0)

    # data (raw datasets — no augmentation yet)
    train_data_config = ExportManifestConfig(csv_path=train_manifest_path, base_dir=base_dir)
    val_data_config   = ExportManifestConfig(csv_path=val_manifest_path,   base_dir=base_dir)
    train_ds_raw = ExportManifestDataset(train_data_config)
    val_ds_raw   = ExportManifestDataset(val_data_config)

    # joint augmentation pipelines (albumentations)
    # train: geometric + color + mild affine + ImageNet normalize
    train_aug = A.Compose([
        # geometric
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        # color (image-only; does not affect masks)
        A.ColorJitter(
            brightness=0.15, contrast=0.15,
            saturation=0.1, hue=0.04, p=0.5,
        ),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        # mild affine (conservative, preserve thin 3px boundaries)
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

    # validation: normalize only, no augmentation
    val_aug = A.Compose([
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])

    # wrap with augmentation
    train_ds = AugmentedDataset(train_ds_raw, transform=train_aug)
    val_ds   = AugmentedDataset(val_ds_raw,   transform=val_aug)

    # semantic weights (unused for optimization when SEM_LOSS_WEIGHT=0.0)
    sem_w = torch.tensor(wobj["weights"], dtype=torch.float32, device=device).clone()
    sem_w[3] = min(sem_w[3].item(), 6.0)  # neutrophil cap
    sem_w[4] = min(sem_w[4].item(), 2.0)  # macrophage cap
    print("[sem weights (capped)]", sem_w.detach().cpu().numpy().tolist())

    # ternary CE weights from TRAIN ternary masks + boundary multiplier
    # compute raw inverse-frequency weights
    ter_counts, ter_w_raw = compute_ternary_ce_weights_from_manifest(
        train_ds.df, base_dir=base_dir
    )
    ter_w_raw = ter_w_raw.astype(np.float64)

    # stage 1: initial clip (safety)
    ter_w_stage1 = np.clip(
        ter_w_raw,
        TER_WEIGHT_CAP_MIN,
        TER_WEIGHT_CAP_MAX,
    )

    # stage 2: optional boundary upweight
    ter_w_stage2 = ter_w_stage1.copy()
    ter_w_stage2[2] *= float(TER_BOUNDARY_MULT)

    # stage 3: clip + hard cap (halo control)
    ter_w_clip = np.clip(
        ter_w_stage2,
        TER_WEIGHT_CAP_MIN,
        TER_WEIGHT_CAP_MAX,
    )
    ter_w_clip[2] = min(ter_w_clip[2], float(TER_BOUNDARY_HARD_CAP))

    # stage 4: renormalize (keep mean ~ 1)
    ter_w_final = ter_w_clip / ter_w_clip.mean()

    ter_ce_w = torch.tensor(ter_w_final, dtype=torch.float32, device=device)

    # logging
    print("[ternary pixel counts bg/inside/boundary]", ter_counts.tolist())
    print("[ternary CE weights RAW]", ter_w_raw.tolist())
    print("[ternary CE weights CLIP]", ter_w_clip.tolist())
    print("[ternary CE weights FINAL]", ter_w_final.tolist())

    # immune-aware sampling (TRAIN ONLY)
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

    # probe val batch for sanity + preview monitor
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

    # model (dual-head)
    model = SharedUnetTwoHead(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        in_channels=3,
        decoder_channels=(256, 128, 64, 32, 16),
        sem_classes=5,
        ter_classes=3,
        activation=None,
    ).to(device)

    # freeze semantic head weights (keep encoder+decoder trainable)
    if FREEZE_SEMANTIC_HEAD:
        for p in model.sem_head.parameters():
            p.requires_grad = False
        print("[FREEZE] semantic head frozen (requires_grad=False)")

    # config logging
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
        "augmentation_details": "HFlip+VFlip+Rot90+ColorJitter+GaussBlur+Affine+ImageNetNorm",
        "warmup_epochs": 5,
        "lr_schedule": "LinearWarmup(5)+CosineAnnealing(95)",
    }
    config_json.write_text(json.dumps(config, indent=2))

    # optimizer: exclude frozen semantic head params (if frozen)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable parameters found (did you freeze too much?).")

    opt = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)

    warmup_epochs = 5
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        opt,
        start_factor=1e-5 / lr,   # starts at 1e-5
        end_factor=1.0,            # ramps to lr (3e-4)
        total_iters=warmup_epochs,
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt,
        T_max=max_epochs - warmup_epochs,  # cosine over remaining epochs
        eta_min=1e-6,
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        opt,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs],
    )

    stopper = PlateauStopper(
        patience=3 * plateau_patience,
        min_delta=plateau_min_delta,
        min_epochs=plateau_min_epochs,
        mode=MONITOR_MODE,
        ema_alpha=0.3,
    )

    # metrics schema
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

            # train
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
                    dice_w=None,  # set to sem_w if want weighted Dice
                    ignore_index=None,
                )

                # ternary combined loss: focal CE + dice
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

            # eval
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

            # derive combined monitor/select metric
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
                print(f"[LR] WarmupCosine: {prev_lr:.3e} -> {lr_now:.3e}")

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
