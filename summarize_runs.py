#!/usr/bin/env python3
"""
summarize_runs.py

Summarize experiment runs produced by dh_train_immune_boost.py.

Scans:
  <OUT_ROOT>/runs/<run_name>/metrics.csv
  <OUT_ROOT>/runs/<run_name>/config.json   (optional but recommended)

Outputs:
  - prints a table (one row per run) with the best epoch according to selection_key/mode
  - writes runs_summary.csv (one row per run)
  - writes grid_summary.csv (aggregated by lambda_ter across seeds: mean/std and n)

Usage:
  PYTHONPATH=. python scripts/summarize_runs.py
  PYTHONPATH=. python scripts/summarize_runs.py --runs_root /path/to/outputs/runs
  PYTHONPATH=. python scripts/summarize_runs.py --selection_key ter_dice_macro_boundary --selection_mode max

Notes:
  - If selection_key/mode are present in config.json, those are used by default.
  - If not, you can override with CLI flags.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple
import math
import statistics


# -----------------------------
# CLI
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--runs_root",
        type=str,
        default="",
        help="Path to outputs/runs. If empty, uses $OUT_ROOT/runs else ./outputs/runs.",
    )
    p.add_argument(
        "--out_csv",
        type=str,
        default="runs_summary.csv",
        help="Filename for per-run summary (written inside runs_root).",
    )
    p.add_argument(
        "--grid_csv",
        type=str,
        default="grid_summary.csv",
        help="Filename for aggregated grid summary (written inside runs_root).",
    )

    # Optional overrides (otherwise derive from config.json)
    p.add_argument("--selection_key", type=str, default="", help="Override selection key.")
    p.add_argument("--selection_mode", type=str, default="", choices=["", "min", "max"], help="Override selection mode.")
    p.add_argument("--monitor_key", type=str, default="", help="Override monitor key (reported, not used for selection).")
    p.add_argument("--monitor_mode", type=str, default="", choices=["", "min", "max"], help="Override monitor mode.")

    p.add_argument(
        "--include_pattern",
        type=str,
        default="",
        help="If provided, only include runs where run_name contains this substring.",
    )
    p.add_argument(
        "--sort_by",
        type=str,
        default="lambda_ter,seed",
        help="Comma-separated sort keys (e.g., 'lambda_ter,seed' or 'best_selection').",
    )
    p.add_argument(
        "--top_k",
        type=int,
        default=0,
        help="If >0, print only top K runs (after sorting). Still writes full CSVs.",
    )
    return p.parse_args()


# -----------------------------
# Small helpers
# -----------------------------
def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        if isinstance(x, str) and x.strip() == "":
            return None
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except Exception:
        return None


def _safe_int(x: Any) -> Optional[int]:
    try:
        if x is None:
            return None
        if isinstance(x, str) and x.strip() == "":
            return None
        return int(float(x))
    except Exception:
        return None


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _read_metrics_csv(path: Path) -> Tuple[list[str], list[dict[str, str]]]:
    with open(path, "r", newline="") as f:
        r = csv.DictReader(f)
        rows = [row for row in r]
        fields = list(r.fieldnames or [])
    return fields, rows


def _choose_best_row(
    rows: list[dict[str, str]],
    selection_key: str,
    selection_mode: str,
) -> Tuple[Optional[dict[str, str]], Optional[float]]:
    """
    Return (best_row, best_value) according to selection_key/mode.
    Falls back to None if no valid values found.
    """
    if not rows:
        return None, None
    if selection_mode not in ("min", "max"):
        selection_mode = "max"

    best_row = None
    best_val = None

    for row in rows:
        v = _safe_float(row.get(selection_key))
        if v is None:
            continue
        if best_val is None:
            best_val = v
            best_row = row
            continue
        if selection_mode == "max":
            if v > best_val:
                best_val = v
                best_row = row
        else:
            if v < best_val:
                best_val = v
                best_row = row

    return best_row, best_val


def _pretty_table(rows: list[dict[str, Any]], cols: list[str], max_width: int = 38) -> str:
    def fmt(v: Any) -> str:
        if v is None:
            return ""
        if isinstance(v, float):
            return f"{v:.4f}"
        return str(v)

    # truncate long strings
    def trunc(s: str) -> str:
        if len(s) <= max_width:
            return s
        return s[: max_width - 3] + "..."

    formatted = []
    for r in rows:
        fr = {}
        for c in cols:
            fr[c] = trunc(fmt(r.get(c)))
        formatted.append(fr)

    widths = {c: max(len(c), max((len(fr[c]) for fr in formatted), default=0)) for c in cols}

    lines = []
    header = " | ".join(c.ljust(widths[c]) for c in cols)
    sep = "-+-".join("-" * widths[c] for c in cols)
    lines.append(header)
    lines.append(sep)
    for fr in formatted:
        lines.append(" | ".join(fr[c].ljust(widths[c]) for c in cols))
    return "\n".join(lines)


def _resolve_runs_root(arg_runs_root: str) -> Path:
    if arg_runs_root.strip():
        return Path(arg_runs_root).expanduser().resolve()

    out_root = os.environ.get("OUT_ROOT", "").strip()
    if out_root:
        return (Path(out_root).expanduser().resolve() / "runs").resolve()

    # fallback: ./outputs/runs
    return (Path.cwd() / "outputs" / "runs").resolve()


# -----------------------------
# Main summarize
# -----------------------------
@dataclass
class RunSummary:
    run_name: str
    run_dir: Path
    metrics_path: Path
    config_path: Optional[Path]

    # experiment factors
    lambda_ter: Optional[float]
    seed: Optional[int]
    encoder: Optional[str]

    # selection logic
    selection_key: str
    selection_mode: str
    monitor_key: str
    monitor_mode: str

    # best epoch
    best_epoch: Optional[int]
    best_global_step: Optional[int]
    best_selection: Optional[float]

    # a few headline metrics at best epoch
    miou_all_macro: Optional[float]
    miou_all_micro: Optional[float]
    ter_miou_macro: Optional[float]
    ter_miou_micro: Optional[float]
    ter_dice_macro_boundary: Optional[float]
    ter_dice_macro_inside: Optional[float]
    ter_dice_macro_bg: Optional[float]

    val_loss_total: Optional[float]
    train_loss_total: Optional[float]
    lr: Optional[float]


def summarize_one_run(
    run_dir: Path,
    override_selection_key: str,
    override_selection_mode: str,
    override_monitor_key: str,
    override_monitor_mode: str,
) -> Optional[RunSummary]:
    metrics_path = run_dir / "metrics.csv"
    if not metrics_path.exists():
        return None

    config_path = run_dir / "config.json"
    cfg = _read_json(config_path) if config_path.exists() else {}

    selection_key = override_selection_key.strip() or str(cfg.get("selection_key", "selection_value"))
    selection_mode = override_selection_mode.strip() or str(cfg.get("selection_mode", "max"))
    monitor_key = override_monitor_key.strip() or str(cfg.get("monitor_key", "monitor_value"))
    monitor_mode = override_monitor_mode.strip() or str(cfg.get("monitor_mode", "min"))

    fields, rows = _read_metrics_csv(metrics_path)
    best_row, best_val = _choose_best_row(rows, selection_key=selection_key, selection_mode=selection_mode)

    if best_row is None:
        # no valid selection values; fall back to last row if present
        best_row = rows[-1] if rows else None
        best_val = _safe_float(best_row.get(selection_key)) if best_row else None

    # factors (prefer config.json, fallback to metrics row)
    lam = _safe_float(cfg.get("lambda_ter")) or _safe_float(best_row.get("lambda_ter") if best_row else None)
    seed = _safe_int(cfg.get("seed")) or _safe_int(best_row.get("seed") if best_row else None)
    encoder = cfg.get("encoder") if isinstance(cfg.get("encoder"), str) else None

    def gf(k: str) -> Optional[float]:
        return _safe_float(best_row.get(k)) if best_row else None

    def gi(k: str) -> Optional[int]:
        return _safe_int(best_row.get(k)) if best_row else None

    return RunSummary(
        run_name=run_dir.name,
        run_dir=run_dir,
        metrics_path=metrics_path,
        config_path=config_path if config_path.exists() else None,

        lambda_ter=lam,
        seed=seed,
        encoder=encoder,

        selection_key=selection_key,
        selection_mode=selection_mode,
        monitor_key=monitor_key,
        monitor_mode=monitor_mode,

        best_epoch=gi("epoch"),
        best_global_step=gi("global_step"),
        best_selection=best_val,

        miou_all_macro=gf("miou_all_macro"),
        miou_all_micro=gf("miou_all_micro"),
        ter_miou_macro=gf("ter_miou_macro"),
        ter_miou_micro=gf("ter_miou_micro"),
        ter_dice_macro_boundary=gf("ter_dice_macro_boundary"),
        ter_dice_macro_inside=gf("ter_dice_macro_inside"),
        ter_dice_macro_bg=gf("ter_dice_macro_bg"),

        val_loss_total=gf("val_loss_total"),
        train_loss_total=gf("train_loss_total"),
        lr=gf("lr"),
    )


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def main():
    args = parse_args()
    runs_root = _resolve_runs_root(args.runs_root)

    if not runs_root.exists():
        raise FileNotFoundError(f"runs_root does not exist: {runs_root}")

    run_dirs = [p for p in runs_root.iterdir() if p.is_dir()]
    run_dirs.sort(key=lambda p: p.name)

    summaries: list[RunSummary] = []
    for rd in run_dirs:
        if args.include_pattern and args.include_pattern not in rd.name:
            continue
        s = summarize_one_run(
            rd,
            override_selection_key=args.selection_key,
            override_selection_mode=args.selection_mode,
            override_monitor_key=args.monitor_key,
            override_monitor_mode=args.monitor_mode,
        )
        if s is not None:
            summaries.append(s)

    if not summaries:
        print(f"No runs found under: {runs_root}")
        return

    # Flatten for CSV/table
    flat_rows: list[dict[str, Any]] = []
    for s in summaries:
        flat_rows.append({
            "run_name": s.run_name,
            "lambda_ter": s.lambda_ter,
            "seed": s.seed,
            "encoder": s.encoder,

            "selection_key": s.selection_key,
            "selection_mode": s.selection_mode,
            "best_epoch": s.best_epoch,
            "best_global_step": s.best_global_step,
            "best_selection": s.best_selection,

            "ter_dice_macro_boundary": s.ter_dice_macro_boundary,
            "ter_dice_macro_inside": s.ter_dice_macro_inside,
            "ter_dice_macro_bg": s.ter_dice_macro_bg,
            "ter_miou_macro": s.ter_miou_macro,
            "ter_miou_micro": s.ter_miou_micro,

            "miou_all_macro": s.miou_all_macro,
            "miou_all_micro": s.miou_all_micro,

            "val_loss_total": s.val_loss_total,
            "train_loss_total": s.train_loss_total,
            "lr": s.lr,

            "run_dir": str(s.run_dir),
        })

    # Sorting
    sort_keys = [k.strip() for k in args.sort_by.split(",") if k.strip()]

    def sort_key_fn(r: dict[str, Any]):
        out = []
        for k in sort_keys:
            v = r.get(k)
            # None-safe sorting: None goes last
            if isinstance(v, (int, float)) and v is not None:
                out.append((0, v))
            elif v is None:
                out.append((1, 0))
            else:
                out.append((0, str(v)))
        return tuple(out)

    flat_rows_sorted = sorted(flat_rows, key=sort_key_fn)

    # Optionally reduce printed rows
    to_print = flat_rows_sorted
    if args.top_k and args.top_k > 0:
        to_print = flat_rows_sorted[: args.top_k]

    # Console table (compact, high-signal)
    table_cols = [
        "lambda_ter", "seed",
        "best_epoch", "best_selection",
        "ter_dice_macro_boundary",
        "ter_miou_macro",
        "miou_all_macro",
        "val_loss_total",
        "run_name",
    ]
    print(f"\n[runs_root] {runs_root}")
    print(_pretty_table(to_print, cols=table_cols))

    # Write per-run CSV
    out_csv_path = (runs_root / args.out_csv).resolve()
    fieldnames = [
        "run_name", "lambda_ter", "seed", "encoder",
        "selection_key", "selection_mode",
        "best_epoch", "best_global_step", "best_selection",
        "ter_dice_macro_boundary", "ter_dice_macro_inside", "ter_dice_macro_bg",
        "ter_miou_macro", "ter_miou_micro",
        "miou_all_macro", "miou_all_micro",
        "val_loss_total", "train_loss_total", "lr",
        "run_dir",
    ]
    write_csv(out_csv_path, flat_rows_sorted, fieldnames=fieldnames)
    print(f"\n[write] {out_csv_path}")

    # Aggregate grid by lambda_ter (across seeds)
    by_lam: dict[float, list[dict[str, Any]]] = {}
    for r in flat_rows_sorted:
        lam = r.get("lambda_ter")
        lamf = _safe_float(lam)
        if lamf is None:
            continue
        by_lam.setdefault(lamf, []).append(r)

    grid_rows: list[dict[str, Any]] = []
    for lam in sorted(by_lam.keys()):
        rows = by_lam[lam]

        def collect(key: str) -> list[float]:
            vals = []
            for rr in rows:
                v = _safe_float(rr.get(key))
                if v is not None:
                    vals.append(v)
            return vals

        sel_key = rows[0].get("selection_key", "")
        vals_sel = collect("best_selection")
        vals_bnd = collect("ter_dice_macro_boundary")
        vals_miou = collect("miou_all_macro")

        def mean_std(xs: list[float]) -> tuple[Optional[float], Optional[float]]:
            if not xs:
                return None, None
            if len(xs) == 1:
                return xs[0], 0.0
            return statistics.mean(xs), statistics.stdev(xs)

        m_sel, s_sel = mean_std(vals_sel)
        m_bnd, s_bnd = mean_std(vals_bnd)
        m_miou, s_miou = mean_std(vals_miou)

        grid_rows.append({
            "lambda_ter": lam,
            "n_runs": len(rows),
            "selection_key": sel_key,

            "best_selection_mean": m_sel,
            "best_selection_std": s_sel,

            "boundary_dice_mean": m_bnd,
            "boundary_dice_std": s_bnd,

            "miou_all_macro_mean": m_miou,
            "miou_all_macro_std": s_miou,
        })

    grid_csv_path = (runs_root / args.grid_csv).resolve()
    grid_fields = [
        "lambda_ter", "n_runs", "selection_key",
        "best_selection_mean", "best_selection_std",
        "boundary_dice_mean", "boundary_dice_std",
        "miou_all_macro_mean", "miou_all_macro_std",
    ]
    write_csv(grid_csv_path, grid_rows, fieldnames=grid_fields)
    print(f"[write] {grid_csv_path}")

    # Quick “winner” note: best mean boundary dice
    best_grid = None
    for gr in grid_rows:
        v = _safe_float(gr.get("boundary_dice_mean"))
        if v is None:
            continue
        if best_grid is None or v > float(best_grid.get("boundary_dice_mean", float("-inf"))):
            best_grid = gr

    if best_grid:
        print(
            "\n[grid best] "
            f"lambda_ter={best_grid['lambda_ter']} "
            f"boundary_dice_mean={best_grid['boundary_dice_mean']:.4f} "
            f"(n={best_grid['n_runs']})"
        )


if __name__ == "__main__":
    main()