# V7 prototype modules for LHM poster pipeline

## What this package adds

This package makes the previously missing poster-direction items operational at a **prototype-active** level:

1. Three-environment filesystem layout
2. Local Stable Diffusion v1.5 model downloader/checker for Level-1.5 img2img restoration
3. 2D foreground-aligned held-out evaluation prototype
4. Multi-keyframe render/score-level fusion prototype
5. Status probe script
6. Optional Gradio v7 file with status wording updated to prototype-active where appropriate

## Important honesty boundary

These modules do **not** magically turn LHM into PF-LHM or true multi-image canonical 3DGS fusion. They provide a practical prototype path:

- `Canonical Gaussian fusion`: prototype-active through multi-keyframe render/score-level fusion. True GS tensor fusion still requires LHM internals to expose canonical Gaussian tensors from each keyframe.
- `Diffusion-based restoration`: prototype-active once the local Diffusers model exists and Gradio uses `diffusers_img2img`.
- `Camera-aligned rendering`: prototype-active through 2D foreground-aligned evaluation. True ZJU K/R/T renderer integration is still future core-renderer work.

## Install locations on AutoDL

Copy files into your running LHM folder:

```bash
cd /root/autodl-tmp/LHM-run
unzip -o v7_prototype_modules_bundle.zip
```

Expected locations:

```text
/root/autodl-tmp/LHM-run/app_motion_ms_v7_prototype_active.py
/root/autodl-tmp/LHM-run/scripts/setup_three_env_architecture.sh
/root/autodl-tmp/LHM-run/scripts/download_sd15_modelscope.py
/root/autodl-tmp/LHM-run/scripts/check_sd15_diffusers_model.py
/root/autodl-tmp/LHM-run/scripts/v7_camera_aligned_eval_2d.py
/root/autodl-tmp/LHM-run/scripts/v7_multikeyframe_render_fusion.py
/root/autodl-tmp/LHM-run/scripts/v7_status_probe.py
```

## Step 1: create the three-environment layout

```bash
cd /root/autodl-tmp/LHM-run
bash scripts/setup_three_env_architecture.sh
cat /autodl-fs/data/lhm_shared/poster_env_paths.json
```

## Step 2: download/check local diffusion model

```bash
cd /root/autodl-tmp/LHM-run
conda activate lhm
pip install -q modelscope diffusers transformers accelerate safetensors

python scripts/download_sd15_modelscope.py
python scripts/check_sd15_diffusers_model.py \
  /autodl-fs/data/lhm_shared/shared_assets/diffusion_models/stable-diffusion-img2img-local
```

Use this path in Gradio:

```text
/autodl-fs/data/lhm_shared/shared_assets/diffusion_models/stable-diffusion-img2img-local
```

## Step 3: optional Gradio v7 replacement

```bash
cd /root/autodl-tmp/LHM-run
cp app_motion_ms.py app_motion_ms_backup_before_v7_prototype_active.py
cp app_motion_ms_v7_prototype_active.py app_motion_ms.py
python -m py_compile app_motion_ms.py
```

Start:

```bash
pkill -f app_motion_ms.py || true
export LHM_URECA_RESULTS_ROOT=/autodl-fs/data/lhm_shared/poster_results
python -u app_motion_ms.py --model_name LHM-500M-HF \
  2>&1 | tee /autodl-fs/data/lhm_shared/poster_results/logs/poster_gradio_v7.log
```

## Step 4: run status probe

```bash
cd /root/autodl-tmp/LHM-run
python scripts/v7_status_probe.py
```

## Standalone camera-aligned 2D eval prototype

After a render finishes and you know the rendered mp4 path:

```bash
python scripts/v7_camera_aligned_eval_2d.py \
  --rendered /path/to/rendered.mp4 \
  --gt /autodl-fs/data/lhm_shared/ureca_inputs/zju_313_wk2/heldout_view_B.mp4 \
  --mask_dir /autodl-fs/data/lhm_shared/ureca_inputs/zju_313_wk2/heldout_view_B_masks \
  --out_csv /autodl-fs/data/lhm_shared/poster_results/v7_camera_aligned_eval_2d.csv
```

## Standalone multi-keyframe render fusion prototype

Put candidate videos from different source keyframes in one folder, then:

```bash
python scripts/v7_multikeyframe_render_fusion.py \
  --candidate_dir /autodl-fs/data/lhm_shared/poster_results/multikeyframe_candidates \
  --gt /autodl-fs/data/lhm_shared/ureca_inputs/zju_313_wk2/heldout_view_B.mp4 \
  --out_dir /autodl-fs/data/lhm_shared/poster_results/v7_multikeyframe_fusion \
  --mode best
```
