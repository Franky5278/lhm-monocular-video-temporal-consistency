#!/bin/bash
# Fetch the pretrained weights this repository does not ship.
#
# Layout after this script:
#   $ROOT/LHM/exps/releases/video_human_benchmark/human-lrm-500M/step_060000/
#   $ROOT/sd15_model/                     (Stable Diffusion 1.5, diffusers layout)
#   $ROOT/LHM/pretrained_models/...       (pose weights, from the prior model tar)
set -Eeuo pipefail

ROOT=${ROOT:-/workspace/lhm}
mkdir -p "$ROOT"/{downloads,logs}
cd "$ROOT/downloads"

PRIOR_URL=https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/aigc3d/data/for_lingteng/LHM/LHM_prior_model.tar
MODEL_URL=https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/aigc3d/data/for_lingteng/LHM/LHM-0.5B.tar

echo "== LHM-0.5B (8.1 GB) $(date -Is)"
[ -s LHM-0.5B.tar ] || curl -fL --retry 5 --continue-at - -o LHM-0.5B.tar "$MODEL_URL"
tar -xf LHM-0.5B.tar -C "$ROOT/LHM"
touch "$ROOT/MODEL_READY"

echo "== LHM prior model (18.8 GB) $(date -Is)"
[ -s LHM_prior_model.tar ] || curl -fL --retry 5 --continue-at - -o LHM_prior_model.tar "$PRIOR_URL"
tar -xf LHM_prior_model.tar -C "$ROOT/LHM"
touch "$ROOT/PRIOR_READY"

echo "== Stable Diffusion 1.5 (ModelScope) $(date -Is)"
if [ ! -f "$ROOT/sd15_model/model_index.json" ]; then
  "$ROOT/env_system/bin/python" - <<PY
from modelscope import snapshot_download
path = snapshot_download('AI-ModelScope/stable-diffusion-v1-5')
print('downloaded to', path)
import os, shutil
target = os.path.join('$ROOT', 'sd15_model')
if os.path.islink(target) or os.path.exists(target):
    shutil.rmtree(target, ignore_errors=True)
os.symlink(path, target)
PY
fi
touch "$ROOT/SD_READY"

# GFPGAN / facexlib weights are optional: LHM downloads them from GitHub on first
# use, which is slow from mainland China. CompShare images ship a mirror.
for src in /model/HuggingFace/nlightcho/gfpgan-v1.3/GFPGANv1.3.pth \
           /model/HuggingFace/andzhang01/Codeformer/facelib/detection_Resnet50_Final.pth; do
  [ -f "$src" ] || continue
  case "$src" in
    *GFPGANv1.3.pth) dst="$ROOT/env_system/lib/python3.10/site-packages/gfpgan/weights/GFPGANv1.3.pth" ;;
    *detection_Resnet50_Final.pth) dst="$ROOT/env_system/lib/python3.10/site-packages/facexlib/weights/detection_Resnet50_Final.pth" ;;
  esac
  [ -f "$dst" ] || { mkdir -p "$(dirname "$dst")"; cp "$src" "$dst"; echo "seeded $(basename "$dst")"; }
done

echo "== done $(date -Is). DINOv2 ViT-L weights download automatically on the first model build."
