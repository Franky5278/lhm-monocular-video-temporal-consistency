# What is in this repository, and what is deliberately not

The goal of the split: `git clone` must give a **complete, runnable code base** that
can rebuild the environment, fetch the weights, accept the data, run the GPU
pipeline and regenerate every table and figure. Large binaries and licensed data are
fetched or prepared, never committed.

## Included (needed to run or to verify a claim)

| Path | Why it must be in git |
|---|---|
| `research/run_experiments.py` | headless pipeline (phases, ablations, fusion, calibrated camera, diffusion) |
| `research/evaluate_video.py` | strict evaluator (frame count/fps rejection, masked PSNR, LPIPS, temporal delta) |
| `research/cameras.py`, `research/rectify_inputs.py` | ZJU relative camera algebra and undistortion |
| `research/analyze_results.py` | regenerates `metrics_table.csv`, figures and tables from stage records, no GPU |
| `research/make_comparison.py` | builds the side-by-side GIF/MP4 clips |
| `research/motion_metrics_from_dirs.py` | recomputes jitter/deviation from saved SMPL-X folders |
| `research/test_invariants.py`, `research/preflight.py`, `research/gpu_smoke.py` | the checks that gate a run |
| `research/plot_historical.py` | the historical figure is reproducible from the raw log |
| `vendor/LHM/` | patched upstream source; contains the 60→120 frame fix and the renderer/camera patches |
| `scripts/setup_env.sh` | rebuilds the exact Python/CUDA environment |
| `scripts/download_models.sh` | fetches the 27 GB of weights this repo cannot ship |
| `scripts/run_gpu.sh`, `scripts/watch_state.py` | the run wrapper and its progress writer |
| `requirements.txt` | direct pins; the full freeze is `evidence/environment_cpu_freeze.txt` |
| `docs/REPRODUCE.md`, `docs/INPUTS.md` | the end-to-end procedure and the data contract |
| `results/gpu_20261027/stages/` | 67 `stage.json` + `summary.json` + `motion_metrics.json` records — the primary evidence |
| `results/gpu_20261027/metrics_table.csv`, `motion_reproduction.csv`, `run_config.json` | derived tables and run identity |
| `results/gpu_20261027/analysis/`, `figures/` | figures and comparison clips referenced by the README |
| `evidence/` | historical log, hash manifests, preflight, evaluator and camera validation |
| `reports/` | recovery audit, protocol, GPU results |

## Deliberately excluded

| Excluded | Size / reason | How it is obtained |
|---|---|---|
| Model weights (`LHM-0.5B.tar`, prior model, SD 1.5, DINOv2, GFPGAN) | 27 GB + 1.2 GB + 0.3 GB; binaries do not belong in git | `scripts/download_models.sh` |
| ZJU-MoCap derived inputs (frames, videos, masks, camera JSON) | research licence forbids redistribution | `docs/INPUTS.md`; SHA-256 of all 506 reference files is in `evidence/input_manifest.json` |
| `env_system/` virtualenv | 5.2 GB, host-specific | `scripts/setup_env.sh` |
| Run logs (`logs/gpu_run.log`, per-stage `stdout.log`) | tens of MB of mostly-tqdm noise | regenerate by running |
| Per-frame CSVs (`per_frame.csv`) | 67 × ~100 KB; `summary.json` already carries the bootstrap intervals | regenerate with `research/evaluate_video.py` |
| `train_data/users/motion_processing/` raw SMPL-X folders | 11 MB of intermediates | regenerate; the derived table is `motion_reproduction.csv` |
| One-off deployment scripts (patches, wheel prefetch, instance control) | environment-specific glue, not part of the method | not needed to reproduce |
| Credentials (`API keys`, SSH keys, instance passwords) | security | never committed; `.gitignore` blocks the local copies |

## Health checks before every push

```bash
python -m compileall -q research scripts          # syntax
python research/test_invariants.py                # 8 mathematical checks (CPU)
python research/analyze_results.py --root results/gpu_20261027/stages --out /tmp/a
python - <<'PY'                                   # README links resolve
import re, pathlib
text = pathlib.Path('README.md').read_text()
for m in re.findall(r'!\[[^\]]*\]\(([^)]+)\)', text):
    assert pathlib.Path(m).exists(), m
print('README assets OK')
PY
```
