# GPU experiment results — 2026-10-27

All numbers below come from `results/gpu_20261027/` in this repository: 67 stages,
67 successes, 776 s of GPU compute, one RTX 4090 (24 GB, 16 vCPU, 64 GB RAM),
torch 2.3.0+cu121, seed 42. Nothing in this file is estimated or carried over from
the historical run.

## 1. Reproduction of the report's smoothing result

The paper reports pose jitter 0.128502 → 0.032545 (3.95×) and translation jitter
0.00132372 → 0.00052960 (2.50×) on ZJU-313. The historical run had a decoding bug
that duplicated the 60-frame clip into 120 frames; that bug is fixed here, so the
inputs are the true 60/120/300 frames and the absolute numbers are not expected to
match. The **reduction ratios** are what the method predicts, and they reproduce:

| sequence | frames | baseline pose jitter | stabilized | ratio | baseline trans jitter | stabilized | ratio |
|---|---|---|---|---|---|---|---|
| ZJU-313 | 60 | 0.113829 | 0.028262 | **4.03×** | 0.0010304 | 0.0004118 | **2.50×** |
| ZJU-390 kick | 120 | 0.062341 | 0.017991 | **3.47×** | 0.0011005 | 0.0003856 | **2.85×** |
| ZJU-390 mid10 | 300 | 0.085995 | 0.024966 | **3.44×** | 0.0010505 | 0.0003945 | **2.66×** |

Paper: 3.95× pose, 2.50× translation. Measured on the corrected input: 4.03× and
2.50× for ZJU-313, and 3.44–3.47× / 2.66–2.85× on the two ZJU-390 segments.

Jitter is the mean per-frame second difference of the 6D rotation vector (pose) and
of metric translation, matching `engine/pose_estimation/pose_utils/postprocess.py`.

## 2. Ablation: which component does the work

ZJU-313, alpha = 0.25 for single-component variants:

| variant | pose jitter | trans jitter | pose deviation | geodesic deviation |
|---|---|---|---|---|
| raw (no smoothing) | 0.490220 | 0.0035456 | 0.000000 | 0.001° |
| Gaussian prefilter only | 0.116800 | 0.0010304 | 0.149780 | 0.617° |
| pose EMA only | 0.028262 | 0.0010304 | 0.424334 | 2.044° |
| translation-velocity EMA only | 0.116800 | 0.0004118 | 0.145909 | 0.617° |
| full, alpha 0.10 | 0.012542 | 0.0002002 | 0.707614 | 3.553° |
| full, alpha 0.50 | 0.053130 | 0.0006855 | 0.268500 | 1.202° |
| full, alpha 0.80 | 0.078730 | 0.0009236 | 0.196800 | 0.825° |

Reading:
- The Gaussian prefilter alone removes 76% of the pose jitter but already costs
  0.15 of pose deviation; this is the *baseline* the paper compares against.
- Pose EMA is responsible for essentially all remaining pose-jitter reduction
  (0.1168 → 0.0283) and costs 2.0° of geodesic deviation.
- Translation-velocity EMA is orthogonal: it cuts translation jitter 2.5× without
  touching pose deviation (0.1459 → same as Gaussian-only).
- Alpha controls the trade-off monotonically; alpha 0.1 reaches 39× jitter
  reduction at 3.55° deviation, alpha 0.8 keeps 0.83° at 6.2× reduction.

![Smoothing trade-off](figures/fig_smoothing_tradeoff.png)

## 3. Multi-keyframe reconstruction and fusion (future work item)

Four additional keyframes were reconstructed independently (frames 0, 15, 45, 59),
then fused with K = 2, 3, 5 sources, uniform and agreement-weighted, plus a
colour-only graph refinement (0.10 and 0.25).

| variant | PSNR masked (dB) | SSIM | LPIPS | temporal delta |
|---|---|---|---|---|
| single keyframe (baseline) | 2.103 | 0.0926 | 0.8263 | 0.0036 |
| temporal stabilized | 2.102 | 0.0926 | 0.8267 | 0.0035 |
| K=2 uniform | 2.263 | 0.0947 | 0.8232 | 0.0038 |
| K=5 uniform | 2.263 | 0.0944 | 0.8231 | 0.0038 |
| K=5 agreement | 2.263 | 0.0943 | 0.8231 | 0.0038 |
| graph refinement 0.25 | 2.263 | 0.0944 | 0.8229 | 0.0037 |

Fusion runs end-to-end, serializes exactly, and its correspondence check passes
(anchor deviation recorded per stage). The measured effect is **within noise**: the
masked PSNR gain of 0.16 dB is not evidence of a better avatar, and there is no
monotone trend with K. This is reported as a null result, not as a success.

![Fusion comparison](figures/compare_fusion.gif)

## 4. Frame-wise diffusion restoration (future work item)

Stable Diffusion 1.5 img2img, paired seed 42 + frame index, all frames:

| sequence | variant | PSNR masked | LPIPS | temporal delta |
|---|---|---|---|---|
| ZJU-313 | stabilized | 2.102 | 0.8267 | 0.0035 |
| ZJU-313 | SD strength 0.10, 12 steps | 2.259 | 0.8257 | 0.0055 |
| ZJU-313 | SD strength 0.15, 20 steps | 2.330 | 0.8339 | 0.0097 |
| ZJU-390 kick | stabilized | 3.267 | 0.7777 | 0.0020 |
| ZJU-390 kick | SD strength 0.10, 12 steps | 3.420 | 0.7760 | 0.0041 |
| ZJU-390 kick | SD strength 0.15, 20 steps | 3.504 | 0.7835 | 0.0084 |

Restoration improves the full-frame/perceptual alignment slightly but **increases**
synchronized temporal delta by 1.6–2.8×, reproducing the historical observation
(0.873 → 1.101) that frame-wise restoration fights temporal consistency.

![Restoration comparison](figures/compare_313.gif)

## 5. Calibrated camera (future work item)

The ZJU-313 source and held-out views were undistorted with the real intrinsics, the
motion was fitted with the source camera, and the avatar was rendered through the
relative B camera derived from the annotations.

| render camera | PSNR masked | SSIM | LPIPS |
|---|---|---|---|
| LHM default camera | 2.103 | 0.0926 | 0.8263 |
| calibrated B camera | **9.493** | 0.0878 | 0.6865 |

The calibrated render is 4.5× better in masked PSNR and clearly better in LPIPS,
which confirms that camera modelling — not avatar detail — dominates this comparison.
9.5 dB is still far from a usable reconstruction, so this is a diagnostic result.

## 6. What the image metrics do and do not mean

The held-out view is a different camera. Comparing a source-camera render to it
pixel-wise yields 0.63–0.86 dB full-frame PSNR for every variant, which is
effectively noise; those numbers are kept in `metrics_table.csv` for transparency
but are not used as evidence. Only the calibrated-camera row and the motion /
temporal metrics are interpretable, and the report states this limit explicitly.

## 7. Cost and runtime

| item | value |
|---|---|
| stages executed | 67 (all success) |
| GPU compute time | 776 s |
| wall-clock for the full run | 20 min |
| GPU | RTX 4090 24 GB @ 2.05 CNY/h |
| estimated GPU cost | ≈ 1.1 CNY |
