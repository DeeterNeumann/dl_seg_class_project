#!/usr/bin/env bash
#
# download_from_lightning.sh
#
# Downloads training outputs from Lightning Studio to local machine via SCP.
# Prerequisites: SSH configured for ssh.lightning.ai in ~/.ssh/config
#
# Usage: bash download_from_lightning.sh

set -euo pipefail

# ---- Configuration ----
REMOTE="ssh.lightning.ai"
REMOTE_BASE="/teamspace/studios/this_studio/dl_seg_class_project/outputs"

# Local project root (directory where this script lives)
LOCAL_ROOT="$(cd "$(dirname "$0")" && pwd)"

# Local destinations
WEIGHTS_DIR="${LOCAL_ROOT}/deploy/weights"
EXAMPLES_DIR="${LOCAL_ROOT}/deploy/examples"
REPORT_DIR="${LOCAL_ROOT}/lightning_outputs"

# Run identifiers (timestamps from checkpoint filenames)
RUN8_TS="20260207_050806"
RUN9_TS="20260207_115812"

# Common checkpoint prefix
CKPT_PREFIX="FROZENSEM__BOUNDW__resnet34__terLam1.0__bmult1.0__select_combo_inside_boundary"

# ---- Create local directories ----
echo "=== Creating local directories ==="
mkdir -p "${WEIGHTS_DIR}"
mkdir -p "${EXAMPLES_DIR}"
mkdir -p "${REPORT_DIR}/run8"
mkdir -p "${REPORT_DIR}/run9"
echo "  ${WEIGHTS_DIR}"
echo "  ${EXAMPLES_DIR}"
echo "  ${REPORT_DIR}/run8"
echo "  ${REPORT_DIR}/run9"
echo ""

# ---- Download checkpoints ----
echo "=== Downloading checkpoints (this may take a few minutes) ==="
echo ""

echo "[1/4] Run 8 best checkpoint (.pt) ..."
scp "${REMOTE}:${REMOTE_BASE}/checkpoints/${CKPT_PREFIX}__${RUN8_TS}__best.pt" \
    "${WEIGHTS_DIR}/run8_best.pt"

echo "[2/4] Run 8 best config (.json) ..."
scp "${REMOTE}:${REMOTE_BASE}/checkpoints/${CKPT_PREFIX}__${RUN8_TS}__best.json" \
    "${WEIGHTS_DIR}/run8_best.json"

echo "[3/4] Run 9 best checkpoint (.pt) ..."
scp "${REMOTE}:${REMOTE_BASE}/checkpoints/${CKPT_PREFIX}__${RUN9_TS}__best.pt" \
    "${WEIGHTS_DIR}/run9_best.pt"

echo "[4/4] Run 9 best config (.json) ..."
scp "${REMOTE}:${REMOTE_BASE}/checkpoints/${CKPT_PREFIX}__${RUN9_TS}__best.json" \
    "${WEIGHTS_DIR}/run9_best.json"

echo ""

# ---- Download dashboards and history ----
echo "=== Downloading dashboards & history ==="
echo ""

RUNS_BASE="${REMOTE_BASE}/runs/${CKPT_PREFIX}"

echo "[Run 8] dashboard.png ..."
scp "${REMOTE}:${RUNS_BASE}__${RUN8_TS}/plots/dashboard.png" \
    "${REPORT_DIR}/run8/dashboard.png"

echo "[Run 8] history.json ..."
scp "${REMOTE}:${RUNS_BASE}__${RUN8_TS}/plots/history.json" \
    "${REPORT_DIR}/run8/history.json"

echo "[Run 9] dashboard.png ..."
scp "${REMOTE}:${RUNS_BASE}__${RUN9_TS}/plots/dashboard.png" \
    "${REPORT_DIR}/run9/dashboard.png"

echo "[Run 9] history.json ..."
scp "${REMOTE}:${RUNS_BASE}__${RUN9_TS}/plots/history.json" \
    "${REPORT_DIR}/run9/history.json"

echo ""

# ---- Download preview images for Gradio demo ----
echo "=== Downloading preview images (for Gradio demo examples) ==="
echo ""

PREVIEW_BASE="${REMOTE_BASE}/preview/${CKPT_PREFIX}__${RUN8_TS}"

# Download the last 3 preview images (most representative of final model quality)
# First, list available previews and grab the last 3
echo "Fetching preview file list..."
PREVIEW_FILES=$(ssh "${REMOTE}" "ls -1 ${PREVIEW_BASE}/*.png 2>/dev/null | tail -3" || true)

if [ -n "${PREVIEW_FILES}" ]; then
    for PFILE in ${PREVIEW_FILES}; do
        FNAME=$(basename "${PFILE}")
        echo "  Downloading ${FNAME} ..."
        scp "${REMOTE}:${PFILE}" "${EXAMPLES_DIR}/${FNAME}"
    done
else
    echo "  No preview images found at ${PREVIEW_BASE}/"
    echo "  You can manually add example images to ${EXAMPLES_DIR}/"
fi

echo ""

# ---- Summary ----
echo "========================================="
echo "  Download complete!"
echo "========================================="
echo ""
echo "Checkpoints:"
ls -lh "${WEIGHTS_DIR}"/*.pt 2>/dev/null || echo "  (none found)"
echo ""
echo "Configs:"
ls -lh "${WEIGHTS_DIR}"/*.json 2>/dev/null || echo "  (none found)"
echo ""
echo "Report files:"
ls -lh "${REPORT_DIR}"/run8/* 2>/dev/null || echo "  (none found for run8)"
ls -lh "${REPORT_DIR}"/run9/* 2>/dev/null || echo "  (none found for run9)"
echo ""
echo "Example images:"
ls -lh "${EXAMPLES_DIR}"/*.png 2>/dev/null || echo "  (none found)"
echo ""
echo "Next steps:"
echo "  1. Copy your preferred checkpoint to deploy/weights/best_model.pt"
echo "     cp ${WEIGHTS_DIR}/run8_best.pt ${WEIGHTS_DIR}/best_model.pt"
echo "  2. Continue building deploy/model.py"
