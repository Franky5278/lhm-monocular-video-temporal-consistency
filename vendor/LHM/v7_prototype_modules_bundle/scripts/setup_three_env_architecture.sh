#!/usr/bin/env bash
set -euo pipefail

# Three-environment layout for the poster/LHM experiment.
# 1) code env: current runnable LHM code under /root/autodl-tmp/LHM-run
# 2) asset/input env: persistent datasets, pretrained weights, source videos/frames under FS disk
# 3) result env: persistent rendered outputs, metrics, logs under FS disk

RUN_CODE_DIR="${1:-/root/autodl-tmp/LHM-run}"
SHARED_ROOT="${2:-/autodl-fs/data/lhm_shared}"

CODE_BACKUP_DIR="$SHARED_ROOT/code/LHM-run-poster-v7"
ASSET_DIR="$SHARED_ROOT/shared_assets"
INPUT_DIR="$SHARED_ROOT/poster_inputs"
RESULT_DIR="$SHARED_ROOT/poster_results"
DATASET_DIR="$SHARED_ROOT/datasets"
LOG_DIR="$RESULT_DIR/logs"

mkdir -p "$SHARED_ROOT/code" "$CODE_BACKUP_DIR" "$ASSET_DIR" "$INPUT_DIR" "$RESULT_DIR" "$DATASET_DIR" "$LOG_DIR"
mkdir -p "$ASSET_DIR/diffusion_models" "$ASSET_DIR/gfpgan_weights"

# Keep backward-compatible links if old names exist.
if [ -d "$SHARED_ROOT/ureca_inputs" ] && [ ! -e "$INPUT_DIR/zju_313_wk2" ]; then
  ln -sfn "$SHARED_ROOT/ureca_inputs/zju_313_wk2" "$INPUT_DIR/zju_313_wk2"
fi

cat > "$SHARED_ROOT/poster_env_paths.json" <<EOF
{
  "code_env_current_run": "$RUN_CODE_DIR",
  "code_env_fs_backup": "$CODE_BACKUP_DIR",
  "asset_env": "$ASSET_DIR",
  "input_env": "$INPUT_DIR",
  "dataset_env": "$DATASET_DIR",
  "result_env": "$RESULT_DIR",
  "log_env": "$LOG_DIR",
  "diffusion_models": "$ASSET_DIR/diffusion_models"
}
EOF

# Snapshot only lightweight code files, not huge outputs/checkpoints.
if [ -d "$RUN_CODE_DIR" ]; then
  rsync -a --delete \
    --exclude 'pretrained_models/' \
    --exclude 'train_data/' \
    --exclude 'train_data.bak/' \
    --exclude 'gfpgan/**/weights/' \
    --exclude '**/__pycache__/' \
    --exclude '*.pth' --exclude '*.pt' --exclude '*.ckpt' --exclude '*.safetensors' \
    "$RUN_CODE_DIR/" "$CODE_BACKUP_DIR/" || true
fi

echo "[OK] Three-environment layout prepared."
echo "Config: $SHARED_ROOT/poster_env_paths.json"
echo "Code env:   $RUN_CODE_DIR"
echo "Asset env:  $ASSET_DIR"
echo "Input env:  $INPUT_DIR"
echo "Result env: $RESULT_DIR"
