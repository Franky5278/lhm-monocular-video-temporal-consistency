#!/usr/bin/env python3
"""
Prepare ZJU-MoCap sequence data for the URECA/LHM WK2 protocol.

It converts one ZJU-MoCap sequence into:
  1) source_view_A_frames/       selected source keyframes for LHM input
  2) source_view_A.mp4           source-view motion video
  3) heldout_view_B.mp4          held-out ground-truth video
  4) heldout_view_B_frames/      copied GT frames
  5) heldout_view_B_masks/       copied foreground masks if available
  6) manifest.json               metadata for reproducibility

Expected dataset shape:
  <zju_root>/CoreView_313/annots.npy

This script reads image paths from annots.npy instead of guessing camera folders.
"""

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


def _as_list(x: Any) -> List[Any]:
    if isinstance(x, np.ndarray):
        return x.tolist()
    return list(x)


def resolve_seq_dir(zju_root: Path, seq: str) -> Path:
    candidates = [
        zju_root / f"CoreView_{seq}",
        zju_root / seq,
    ]
    for c in candidates:
        if (c / "annots.npy").exists():
            return c
    raise FileNotFoundError(
        f"Cannot find annots.npy for seq={seq}. Tried:\n"
        + "\n".join(str(c / "annots.npy") for c in candidates)
    )


def read_bgr(path: Path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Cannot read image: {path}")
    return img


def write_video(image_paths: Sequence[Path], out_path: Path, fps: int, max_side: int = 0) -> Tuple[int, int]:
    if not image_paths:
        raise ValueError("No image paths were provided to write_video().")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    first = read_bgr(image_paths[0])
    h0, w0 = first.shape[:2]

    scale = 1.0
    if max_side and max(h0, w0) > max_side:
        scale = max_side / float(max(h0, w0))
    w = int(round(w0 * scale))
    h = int(round(h0 * scale))

    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open video writer for {out_path}")

    for p in image_paths:
        img = read_bgr(p)
        if img.shape[:2] != (h0, w0):
            img = cv2.resize(img, (w0, h0), interpolation=cv2.INTER_AREA)
        if scale != 1.0:
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        writer.write(img)

    writer.release()
    return h, w


def find_mask(seq_dir: Path, rel_img_path: str) -> Optional[Path]:
    rel = Path(rel_img_path)
    candidates = [
        seq_dir / "mask_cihp" / rel,
        seq_dir / "mask" / rel,
        seq_dir / "masks" / rel,
    ]
    candidates += [p.with_suffix(".png") for p in candidates]
    for p in candidates:
        if p.exists():
            return p
    return None


def copy_file(src: Path, dst: Path) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return str(dst)


def evenly_spaced_indices(n_items: int, n_select: int) -> List[int]:
    if n_items <= 0:
        return []
    if n_select <= 0:
        return []
    if n_items <= n_select:
        return list(range(n_items))
    return np.linspace(0, n_items - 1, n_select).round().astype(int).tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zju_root", required=True, help="Path to downloaded zju_mocap folder")
    parser.add_argument("--seq", default="313", help="Sequence id, e.g. 313")
    parser.add_argument("--source_cam", type=int, default=0, help="Camera index for source frames/motion")
    parser.add_argument("--heldout_cam", type=int, default=1, help="Camera index for held-out GT video")
    parser.add_argument("--start", type=int, default=0, help="First frame index in annots['ims']")
    parser.add_argument("--end", type=int, default=59, help="Last frame index in annots['ims']; use -1 for last available")
    parser.add_argument("--step", type=int, default=1, help="Frame stride")
    parser.add_argument("--num_keyframes", type=int, default=5, help="Number of source keyframes to copy")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--max_side", type=int, default=0, help="Optional resize for mp4 videos; 0 keeps original")
    parser.add_argument("--out_root", required=True, help="Output folder for URECA prepared files")
    args = parser.parse_args()

    if args.step <= 0:
        raise ValueError("--step must be positive")
    if args.source_cam == args.heldout_cam:
        print("[WARN] source_cam and heldout_cam are identical. This is not a real held-out view protocol.")

    zju_root = Path(args.zju_root).expanduser().resolve()
    seq_dir = resolve_seq_dir(zju_root, args.seq)
    ann_path = seq_dir / "annots.npy"
    out_root = Path(args.out_root).expanduser().resolve()

    ann: Dict[str, Any] = np.load(str(ann_path), allow_pickle=True).item()
    ims_records = _as_list(ann["ims"])

    if not ims_records:
        raise RuntimeError(f"No frame records found in {ann_path}")

    first_ims = _as_list(ims_records[0]["ims"])
    num_cams = len(first_ims)

    if args.source_cam < 0 or args.source_cam >= num_cams:
        raise ValueError(f"--source_cam out of range. num_cams={num_cams}, got {args.source_cam}")
    if args.heldout_cam < 0 or args.heldout_cam >= num_cams:
        raise ValueError(f"--heldout_cam out of range. num_cams={num_cams}, got {args.heldout_cam}")

    last_frame = len(ims_records) - 1 if args.end < 0 else min(args.end, len(ims_records) - 1)
    frame_ids = list(range(args.start, last_frame + 1, args.step))
    if not frame_ids:
        raise RuntimeError("No frames selected. Check --start/--end/--step.")

    source_paths: List[Path] = []
    heldout_paths: List[Path] = []
    heldout_masks: List[Optional[Path]] = []

    for fid in frame_ids:
        rels = _as_list(ims_records[fid]["ims"])
        if args.source_cam >= len(rels) or args.heldout_cam >= len(rels):
            raise RuntimeError(f"Frame {fid} has only {len(rels)} cameras.")
        source_rel = rels[args.source_cam]
        heldout_rel = rels[args.heldout_cam]

        source_img = seq_dir / source_rel
        heldout_img = seq_dir / heldout_rel

        if not source_img.exists():
            raise FileNotFoundError(f"Missing source image at frame {fid}: {source_img}")
        if not heldout_img.exists():
            raise FileNotFoundError(f"Missing held-out image at frame {fid}: {heldout_img}")

        source_paths.append(source_img)
        heldout_paths.append(heldout_img)
        heldout_masks.append(find_mask(seq_dir, heldout_rel))

    source_frames_dir = out_root / "source_view_A_frames"
    heldout_frames_dir = out_root / "heldout_view_B_frames"
    heldout_masks_dir = out_root / "heldout_view_B_masks"
    source_video = out_root / "source_view_A.mp4"
    heldout_video = out_root / "heldout_view_B.mp4"

    for p in [source_frames_dir, heldout_frames_dir, heldout_masks_dir]:
        p.mkdir(parents=True, exist_ok=True)

    # source keyframes for LHM identity input
    key_local_indices = evenly_spaced_indices(len(source_paths), args.num_keyframes)
    copied_keyframes = []
    for k, local_idx in enumerate(key_local_indices):
        fid = frame_ids[local_idx]
        src = source_paths[local_idx]
        dst = source_frames_dir / f"keyframe_{k:02d}_frame_{fid:06d}{src.suffix.lower()}"
        copied_keyframes.append(copy_file(src, dst))

    # held-out frames and masks
    copied_heldout_frames = []
    copied_heldout_masks = []
    for local_idx, src in enumerate(heldout_paths):
        fid = frame_ids[local_idx]
        dst = heldout_frames_dir / f"gt_frame_{fid:06d}{src.suffix.lower()}"
        copied_heldout_frames.append(copy_file(src, dst))

        mask = heldout_masks[local_idx]
        if mask and mask.exists():
            mdst = heldout_masks_dir / f"mask_{fid:06d}.png"
            copied_heldout_masks.append(copy_file(mask, mdst))

    source_video_shape = write_video(source_paths, source_video, fps=args.fps, max_side=args.max_side)
    heldout_video_shape = write_video(heldout_paths, heldout_video, fps=args.fps, max_side=args.max_side)

    manifest = {
        "seq": args.seq,
        "seq_dir": str(seq_dir),
        "annots_path": str(ann_path),
        "num_cameras": num_cams,
        "source_cam": args.source_cam,
        "heldout_cam": args.heldout_cam,
        "start": args.start,
        "end": last_frame,
        "step": args.step,
        "num_selected_frames": len(frame_ids),
        "fps": args.fps,
        "max_side": args.max_side,
        "source_frames_dir": str(source_frames_dir),
        "source_video": str(source_video),
        "heldout_gt_video": str(heldout_video),
        "heldout_frames_dir": str(heldout_frames_dir),
        "heldout_masks_dir": str(heldout_masks_dir) if copied_heldout_masks else None,
        "num_heldout_masks_found": len(copied_heldout_masks),
        "source_video_hw": source_video_shape,
        "heldout_video_hw": heldout_video_shape,
        "keyframes": copied_keyframes,
    }

    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"[OK] Prepared URECA WK2 files under: {out_root}")


if __name__ == "__main__":
    main()
