#!/bin/bash
# Run the experiment pipeline on a GPU host.
#
# Phases:
#   A  ZJU-313 (60 frames) + ZJU-390 kick (120 frames): baseline, stabilization,
#      ablations, keyframes, fusion, calibrated camera, diffusion, evaluation
#   B  ZJU-390 mid10 (300 frames): baseline + stabilization + evaluation
#
# Each stage is resumable: an existing stage.json with status "success" is reused.
set -uo pipefail

ROOT=${ROOT:-/workspace/lhm}
REPO=${REPO:-$ROOT/project}
PY="$ROOT/env_system/bin/python"
GPU_HOURS=${GPU_HOURS:-3.0}
SD_MODEL=${SD_MODEL:-$ROOT/sd15_model}
CKPT=${CKPT:-$ROOT/LHM/exps/releases/video_human_benchmark/human-lrm-500M/step_060000}
OUT=${OUT:-$ROOT/results/experiment_v1}

export PATH="$ROOT/env_system/bin:/usr/local/cuda/bin:$PATH"
export PYTHONPATH="$ROOT/LHM:$REPO/research"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8} PYTHONUNBUFFERED=1 XFORMERS_DISABLED=1
mkdir -p "$ROOT/logs" "$(dirname "$OUT")"

LOG="$ROOT/logs/gpu_run.log"
START=$(date +%s)
DEADLINE=$((START + $(python3 -c "print(int($GPU_HOURS*3600))")))
echo "$START" > "$ROOT/GPU_STARTED_UNIX"
echo "$DEADLINE" > "$ROOT/GPU_DEADLINE_UNIX"
rm -f "$ROOT/GPU_JOB_EXITED"

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | tee -a "$LOG"
"$PY" "$REPO/scripts/watch_state.py" >> "$ROOT/logs/watch_state.log" 2>&1 &
WATCH=$!

run_phase() {
  local tag="$1"; shift
  echo "$(date -Is) START $tag" >> "$LOG"
  "$PY" "$REPO/research/run_experiments.py" \
    --source "$ROOT/LHM" --inputs "$ROOT/inputs" --checkpoint "$CKPT" \
    --out "$OUT" --deadline "$DEADLINE" --sd-model "$SD_MODEL" "$@" >> "$LOG" 2>&1
  echo "$(date -Is) END $tag exit=$?" >> "$LOG"
}

run_phase A --sequences zju_313_wk2,zju_390_kick_wk2
if [ $((DEADLINE - $(date +%s))) -gt 1800 ]; then
  run_phase B --sequences zju_390_mid10
else
  echo "$(date -Is) SKIP phase B (deadline)" >> "$LOG"
fi

kill "$WATCH" 2>/dev/null || true
tar -C "$(dirname "$OUT")" -czf "$OUT.tar.gz" "$(basename "$OUT")" 2>/dev/null
echo "$(date -Is) all done" > "$ROOT/GPU_JOB_EXITED"
echo "results: $OUT  archive: $OUT.tar.gz"
