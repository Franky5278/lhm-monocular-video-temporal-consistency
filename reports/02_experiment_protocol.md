# Experimental protocol and interpretation

The experiment plan is fixed before inspecting new held-out scores. It tests a frozen LHM backbone with explicit input selection, motion filtering, canonical assets, and restoration. Individual `stage.json` records are the authority for whether a stage actually ran and succeeded.

## Data and isolation

The retained inputs contain three segments from two subjects: ZJU-313, ZJU-390 kick, and ZJU-390 mid-sequence. The two ZJU-390 segments are not independent subjects. Source and held-out camera IDs, extraction ranges, frame counts, and source keyframes are retained in manifests. Frame-count equality and 30-fps correspondence are checked before evaluation.

The held-out video is used for evaluation only. It is not used to rank source frames, choose a Gaussian fusion weight, tune a parameter, or select the published best result. Camera-B calibration is used only for its rendering transform and distortion correction. The old standalone candidate-video script that ranks candidates against held-out GT is not part of the new selection pipeline.

## Planned variants

| Stage | Controlled comparison |
|---|---|
| Motion preprocessing | Estimated raw motion, Gaussian-only, pose EMA only, velocity EMA only, combined filtering |
| EMA coefficient | 0.10, 0.25, 0.50, 0.80, keeping the canonical avatar fixed |
| Source reconstruction | Five retained source keyframes, shared identity shape |
| Canonical fusion | K=2/3/5, equal weights versus an offset-agreement confidence proxy |
| Graph refinement | Color residual strengths 0.10 and 0.25 on low-confidence canonical anchors |
| Image restoration | Strengths 0.10/0.15, 12/20 configured steps, paired per-frame seeds, full restored sequences |
| Camera integration | Separate rectified source-K fitting and relative Camera-B K/R/T rendering attempt |

K=2 samples the earliest and latest available retained keyframes; K=3 samples earliest, middle, latest. K=5 uses all five. All sources share the reference-estimated body shape. Fusion is rejected if anchor coordinates, point counts, or body shapes disagree. This is an explicit correspondence check, not a proof of reliable surface visibility.

Quaternions are aligned to a common hemisphere before averaging. Positive scales are averaged in log space. Confidence comes from cross-source offset agreement, with stable softmax normalization. It is not calibrated visibility, segmentation confidence, or a learned reliability score. Graph refinement changes appearance attributes only and may connect nearby but semantically different surfaces; it is a heuristic ablation, not trained 3D diffusion.

## Measurements

Every successful video evaluation exports per-frame CSVs, summary JSON, and sampled GT/prediction/error strips. Metrics use an explicitly stated 512×512 evaluation size. Full-image PSNR and SSIM, true mask-pixel PSNR, GT-bounding-box PSNR, and independently normalized bounding-box PSNR are separately named. Independent box normalization is not a calibrated new-view metric.

LPIPS evaluates all compared frames. Temporal error compares signed RGB frame differences in 0–1 units, with both full-image and foreground-union variants. The first frame has no temporal-difference measurement. Descriptive confidence intervals use frame blocks; they must not be interpreted as evidence of generalization across independent people.

Motion results retain the historical 6D rotation-space second-difference/deviation definitions for traceability. Additional physical measurements include geodesic pose deviation in degrees, translation speed in meters/second, and translation acceleration in meters/second². Deviation is relative to fitted raw motion, not motion ground truth. `endpoint_displacement_m` replaces the misleading historical `drift_trans` name. Endpoint error versus raw is reported separately because velocity integration includes endpoint correction.

## Restoration and reproducibility

Restoration is frame-wise pretrained img2img; no new model is trained. Every configuration uses the same `42 + frame_index` noise seed for a given frame, so parameter comparisons are paired. Configured steps and effective denoising steps are distinct: low strength truncates the img2img schedule, and 12 steps at strengths 0.10 and 0.15 can both yield only one effective denoising step. Twenty-step variants are included to expose that difference.

A restored sequence is not reported as a video-diffusion model. Identity changes, temporal flicker, and worsened metrics remain valid negative results. Full-sequence and per-frame measurements are retained rather than selecting only attractive frames.

## Execution and stopping

Stages record configuration, elapsed seconds, peak GPU memory, artifact paths, and a success/failure/skip status. Successful stages can be reused after interruption. A budget deadline reserves time for backup. Source weights and input hashes, environment versions, failure logs, and execution commands accompany the final successful run records.

These are planned comparisons. A row in this plan is not a completed result. Missing dependencies, rejected correspondence, failed camera rendering, or budget-limited stages must remain visible in the final status table.
