#!/bin/bash
# Build the Python environment used by this repository.
#
# Target: Ubuntu 22.04 container with CUDA 12.1 driver, python 3.10 available.
# Tested on CompShare "pytorch_2.3.0_Py3.12" image with system python 3.10.
# The build compiles four CUDA extensions and takes ~40-60 min on 8 cores.
set -Eeuo pipefail

ROOT=${ROOT:-/workspace/lhm}
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PY="$ROOT/env_system/bin/python"
LOG="$ROOT/logs/setup_env.log"

mkdir -p "$ROOT"/{logs,results,downloads}
exec > >(tee -a "$LOG") 2>&1
echo "== setup_env $(date -Is) repo=$REPO root=$ROOT"

export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq build-essential git curl ffmpeg python3.10-venv \
  libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 >/dev/null

if [ ! -x "$PY" ]; then
  python3.10 -m venv "$ROOT/env_system"
  "$PY" -m pip install -U pip setuptools==74.0.0 wheel ninja packaging
fi
export PATH="$ROOT/env_system/bin:/usr/local/cuda/bin:$PATH"
export CUDA_HOME=/usr/local/cuda
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-8.9}   # Ada: 4090 / 4090_48G
export FORCE_CUDA=1 MAX_JOBS=${MAX_JOBS:-2} OMP_NUM_THREADS=4 XFORMERS_DISABLED=1

"$PY" -m pip install -c <(printf 'torch==2.3.0\ntorchvision==0.18.0\nnumpy==1.26.4\n') \
  -r "$REPO/requirements.txt"

# BasicSR 1.4.2 still imports the removed torchvision.functional_tensor alias.
"$PY" - <<'PYCODE'
from pathlib import Path
import site
for base in site.getsitepackages():
    p = Path(base) / 'basicsr/data/degradations.py'
    if p.exists():
        p.write_text(p.read_text().replace('torchvision.transforms.functional_tensor',
                                           'torchvision.transforms.functional'))
PYCODE

# Compiled extensions (this is the slow part).
"$PY" -m pip install 'git+https://github.com/facebookresearch/pytorch3d.git@v0.7.7' --no-build-isolation
"$PY" -m pip install 'git+https://github.com/ashawkey/diff-gaussian-rasterization.git' --no-build-isolation
"$PY" -m pip install 'git+https://github.com/camenduru/simple-knn.git' --no-build-isolation
MMCV_WITH_OPS=1 "$PY" -m pip install mmcv-full==1.7.2 --no-build-isolation
"$PY" -m pip install -e "$ROOT/LHM/engine/pose_estimation/third-party/ViTPose" --no-deps
SAM2_BUILD_CUDA=0 "$PY" -m pip install 'git+https://github.com/hitsz-zuoqi/sam2.git' --no-build-isolation --no-deps

"$PY" -m pip freeze > "$ROOT/results/environment_freeze.txt"
touch "$ROOT/CPU_READY"
echo "== setup_env done $(date -Is)"
echo "Next: scripts/download_models.sh, then research/preflight.py"
