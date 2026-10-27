# Reproducing the results on a fresh GPU host

Everything below is what was actually executed on 2026-10-27 (RTX 4090 24 GB,
Ubuntu 22.04 container, CUDA 12.1 driver). A 24 GB card is required; 16 GB may work
for the 313 sequence but was not tested.

## 0. Host assumptions

- x86_64 Linux, CUDA 12.1 driver, `python3.10`, `git`, outbound network.
- Working root used by all scripts: `/workspace/lhm` (override with `ROOT=`).
- The compiled extensions are built for **sm_89 (Ada)**. On a different compute
  capability, re-run `scripts/setup_env.sh` with `TORCH_CUDA_ARCH_LIST` set for that
  card, otherwise the CUDA kernels will fail at runtime.

## 1. Clone and place the code

```bash
sudo mkdir -p /workspace && sudo chown "$USER" /workspace
git clone https://github.com/Franky5278/lhm-monocular-video-temporal-consistency.git \
  /workspace/lhm/project
cd /workspace/lhm/project
```

## 2. Environment (~40–60 min, compiles four CUDA extensions)

```bash
ROOT=/workspace/lhm bash scripts/setup_env.sh
```

Installs the pinned stack from `requirements.txt`, builds `pytorch3d`,
`diff-gaussian-rasterization`, `simple-knn`, `mmcv-full`, ViTPose and SAM2.
The full frozen environment of the reference run is
`evidence/environment_cpu_freeze.txt`.

## 3. Weights (~27 GB download, not in git)

```bash
ROOT=/workspace/lhm bash scripts/download_models.sh
```

Fetches `LHM-0.5B.tar` (8.1 GB) and `LHM_prior_model.tar` (18.8 GB) from the
upstream OSS buckets, plus Stable Diffusion 1.5 through ModelScope. DINOv2 ViT-L is
downloaded automatically by LHM on the first model build (~1.2 GB).

## 4. Inputs (not redistributed — see `docs/INPUTS.md`)

```bash
ls /workspace/lhm/inputs
# cameras_313.json  cameras_390.json
# zju_313_wk2/  zju_390_kick_wk2/  zju_390_mid10/
```

Each sequence needs `source_view_A.mp4`, `source_view_A_frames/*.jpg`,
`heldout_view_B.mp4` and (for the two masked sequences) `heldout_view_B_masks/*.png`.
`evidence/input_manifest.json` lists the SHA-256 of all 506 files of the reference
inputs, so a prepared copy can be verified file by file.

## 5. Preflight (CPU, no GPU cost)

```bash
LHM_ROOT=/workspace/lhm /workspace/lhm/env_system/bin/python research/preflight.py
/workspace/lhm/env_system/bin/python research/test_invariants.py
/workspace/lhm/env_system/bin/python research/gpu_smoke.py /workspace/lhm/results/gpu_smoke.json
```

`preflight.py` checks 12 imports and 3 pose assets; `test_invariants.py` runs 8
mathematical checks; `gpu_smoke.py` verifies that the compiled CUDA kernels actually
execute (simple-knn vs pairwise distances, PyTorch3D KNN, MMCV NMS).

## 6. Run

```bash
ROOT=/workspace/lhm REPO=/workspace/lhm/project GPU_HOURS=3 \
  bash scripts/run_gpu.sh
```

Phase A runs ZJU-313 and ZJU-390 kick end to end (baseline, stabilization, seven
ablations, four extra keyframes, K=2/3/5 fusion with uniform and agreement weights,
graph refinement, calibrated camera, four diffusion settings, evaluation).
Phase B adds ZJU-390 mid10 if more than 30 minutes remain in the deadline.

Every stage writes `stage.json` (status, seconds, peak VRAM, config). Re-running
skips stages that already succeeded, so an interrupted run resumes cheaply.

## 7. Analyse

```bash
python research/analyze_results.py \
  --root /workspace/lhm/results/experiment_v1 \
  --out  /workspace/lhm/results/experiment_v1/analysis
python research/make_comparison.py --out figures/compare_313.gif --panels \
  source=/workspace/lhm/inputs/zju_313_wk2/source_view_A.mp4 \
  baseline=/workspace/lhm/results/experiment_v1/zju_313_wk2/core/baseline.mp4 \
  stabilized=/workspace/lhm/results/experiment_v1/zju_313_wk2/core/stabilized.mp4
```

`analyze_results.py` produces `metrics_table.csv`, the figures, and
`results_tables.md` directly from stage records; it does not need a GPU.

## Reference run

| item | value |
|---|---|
| stages | 67, all success |
| GPU compute | 776 s |
| wall clock | 20 min |
| GPU | RTX 4090 24 GB, 16 vCPU, 64 GB RAM |
| torch | 2.3.0+cu121, Python 3.10.12 |
