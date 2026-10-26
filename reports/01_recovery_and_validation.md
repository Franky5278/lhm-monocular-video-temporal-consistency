# Source recovery and validation

This report separates recovered historical evidence, current implementation changes, and new experiments. At this checkpoint, no new CompShare GPU reconstruction result has been certified.

## Historical evidence

The recovered `LHM-run-poster-v7` log contains the report's exact rounded motion results. Its baseline already applies Gaussian preprocessing; the improved variant additionally applies group-specific pose EMA and translation-velocity EMA. Pose jitter changes from 0.1285019219 to 0.0325448886, and translation jitter from 0.0013237192 to 0.0005296041. Pose deviation increases from 0.1611207873 to 0.4766274393; translation deviation increases from 0.0014066116 to 0.0082089286. These values describe smoothness versus fidelity to the estimated raw motion, not accuracy against motion ground truth.

The same log confirms pretrained img2img restoration of the first 30 of 120 output frames. It also records held-out temporal delta 0.9257648 → 0.8733716 after stabilization, rising to 1.1012279 after restoration. The historical metric is in 0–255 pixel units and uses uncalibrated image resizing/cropping. New normalized or calibrated metrics must not be compared numerically with it as though their protocols were identical.

The supplied paper explicitly leaves trained 3D diffusion, reliable per-Gaussian fusion, a reloadable fused canonical avatar, and full calibrated camera evaluation for future work. Related papers are not reproduced baselines. The README has been corrected accordingly.

## A confirmed implementation defect

`engine/pose_estimation/video2motion.py::load_video` originally read the complete input twice. On the retained ZJU-313 input, its 120 frames consist of two pixel-identical 60-frame halves. The corrected function returns exactly the original 60-frame first pass. The regression verifies decoded pixels, not only container duration.

This explains the historical 60-input/120-output discrepancy. It does not by itself recover the complete historical optimizer state or certify every old metric. The new evaluator rejects unequal frame counts or frame rates rather than silently truncating a comparison.

## Current checks

| Check | Evidence | Interpretation |
|---|---|---|
| Single-pass decoding | `evidence/frame_duplication_regression.json` | Real retained input, exact pixel preservation |
| Evaluator identity and mismatch rejection | `evidence/evaluator_validation.json` | Sanity tests, not reconstruction quality |
| ZJU relative-camera algebra | `evidence/camera_algebra_checks.json` | Calibration arithmetic, not estimated-pose accuracy |
| Fusion and serialization invariants | `evidence/invariant_tests.log` | Mathematical implementation checks |
| End-to-end GPU reconstruction | Pending | No new quantitative claim |

The invariant suite covers quaternion antipodes, mismatched body shape and anchors, the requirement for multiple independent sources, log-scale averaging, high-confidence preservation during graph refinement, serialization, and equivalence of native attention to an explicit attention calculation. A passing synthetic test is not counted as a dataset experiment.

## Recovery boundaries

The original local backup remains unchanged. Its file hashes are retained in `evidence/recovered_source/manifest.json`. New modifications live in the vendored working copy and Git history. The historical log is preserved separately from new runs. Temporary baseline/stabilized videos referenced by the old server log were not recovered at their original paths; a retained four-way comparison is not a replacement for original-resolution videos when recomputing exact old metrics.

The deployment record also preserves environment failures, including a memory-limited dependency solver, slow wheel downloads, and an incompatible optional attention wheel. These are deployment failures, not negative reconstruction results. The final environment freeze and CPU/GPU preflight records will identify the actual successful configuration if one is obtained.

CPU deployment update (2026-10-27): all 12 import checks and three pose-weight presence checks pass on Python3.10/Torch2.3+cu121. PyTorch3D, Gaussian rasterizer, simple-knn and MMCV extensions compiled in cardless mode. These checks do not verify CUDA kernel execution or full reconstruction. See ../evidence/cpu_preflight.json and environment_cpu_freeze.txt. GPU startup was rejected by the provider for insufficient capacity; no new GPU result is claimed.
