# Input data contract

The datasets are **not** redistributed in this repository. ZJU-MoCap is a
research-licensed dataset and the frames here are derived from it. What this
repository does ship is the exact contract the code expects plus the SHA-256 of every
file of the reference inputs (`evidence/input_manifest.json`, 506 entries).

## Required layout

```
inputs/
  cameras_313.json          # ZJU camera annotations, 313 subject
  cameras_390.json          # ZJU camera annotations, 390 subject
  zju_313_wk2/
    source_view_A.mp4       # 60 frames, 30 fps, H.264
    source_view_A_frames/   # keyframe_00_frame_000000.jpg ... (5 files)
    heldout_view_B.mp4      # 60 frames, same frame count and fps
    heldout_view_B_masks/   # mask_000000.png ... one per frame
    manifest.json
  zju_390_kick_wk2/         # 120 frames, same structure
  zju_390_mid10/            # 300 frames, no mask directory
```

Hard requirements enforced by the code:

- `source_view_A.mp4` and `heldout_view_B.mp4` must decode to the **same frame count
  and the same fps**; the evaluator raises rather than truncating.
- `heldout_view_B_masks/` must contain exactly one PNG per held-out frame, or be
  absent (then only full-frame metrics are produced).
- `source_view_A_frames/` holds the reference images used for reconstruction. For
  ZJU-313 these are five frames spread over the clip (indices 0, 15, 30, 44, 59);
  the runner picks the middle one as the primary reference and reconstructs the
  others for fusion.
- Keyframe images must be the same resolution as the motion video, because the
  camera intrinsics in `cameras_*.json` are expressed for that resolution.

## How the reference inputs were produced

1. Decode the ZJU-MoCap source video at 30 fps. The historical pipeline had a bug
   that decoded each clip twice and produced 120 frames from a 60-frame file; the
   fixed loader (`vendor/LHM/engine/pose_estimation/video2motion.py::load_video`)
   decodes one pass, verified pixel-exactly in
   `evidence/frame_duplication_regression.json`.
2. Export the five reference frames as JPEG (quality 95) at the video resolution.
3. Export the held-out camera view as an MP4 with the same frame count and fps, and
   its foreground mask per frame from the ZJU annotations.
4. Convert the ZJU annotation arrays to `cameras_*.json` (intrinsics, distortion,
   relative camera A→B, in metres). `research/cameras.py` and
   `evidence/camera_algebra_checks.json` document and verify the conversion.
5. The calibrated-camera stage additionally undistorts the source, the keyframes,
   the GT and the masks, and writes `source_camera.json` per sequence; that file is
   produced at run time by `research/rectify_inputs.py`, not by hand.

## Verifying a copy

```bash
python - <<'PY'
import hashlib, json, pathlib
manifest = json.load(open('evidence/input_manifest.json'))
root = pathlib.Path('/workspace/lhm/inputs')
bad = []
for item in manifest:
    p = root / item['path']
    if not p.exists():
        bad.append((item['path'], 'missing')); continue
    if hashlib.sha256(p.read_bytes()).hexdigest() != item['sha256']:
        bad.append((item['path'], 'hash mismatch'))
print('files', len(manifest), 'problems', len(bad))
print(*bad[:10], sep='\n')
PY
```

## Using your own video

Any monocular clip works if you can supply a second, synchronised view for
evaluation. Without a second view, drop `heldout_view_B.*` and the evaluator's
image metrics are skipped while motion and temporal metrics still run. `cameras_*.json`
is only needed for the calibrated-camera stage.
