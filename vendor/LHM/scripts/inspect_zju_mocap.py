#!/usr/bin/env python3
"""
Inspect a downloaded ZJU-MoCap sequence before using it for URECA/LHM WK2.

Expected dataset shape after download:
  <zju_root>/CoreView_313/annots.npy
  <zju_root>/CoreView_313/<image paths recorded in annots.npy>

This script does NOT require the old NeuralBody environment. It only needs numpy/opencv.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

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


def read_image_shape(path: Path) -> Optional[List[int]]:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        return None
    h, w = img.shape[:2]
    return [h, w, 3]


def find_mask(seq_dir: Path, rel_img_path: str) -> Optional[Path]:
    rel = Path(rel_img_path)
    candidates = [
        seq_dir / "mask_cihp" / rel,
        seq_dir / "mask" / rel,
        seq_dir / "masks" / rel,
    ]
    # masks are often png regardless of image suffix
    candidates += [p.with_suffix(".png") for p in candidates]
    for p in candidates:
        if p.exists():
            return p
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zju_root", required=True, help="Path to downloaded zju_mocap folder")
    parser.add_argument("--seq", default="313", help="Sequence id, e.g. 313")
    parser.add_argument("--max_print", type=int, default=8, help="How many image paths to print")
    parser.add_argument("--json_out", default="", help="Optional path to save inspection json")
    args = parser.parse_args()

    zju_root = Path(args.zju_root).expanduser().resolve()
    seq_dir = resolve_seq_dir(zju_root, args.seq)
    ann_path = seq_dir / "annots.npy"

    ann: Dict[str, Any] = np.load(str(ann_path), allow_pickle=True).item()
    if "ims" not in ann:
        raise KeyError(f"{ann_path} does not contain key 'ims'. Keys: {list(ann.keys())}")
    if "cams" not in ann:
        raise KeyError(f"{ann_path} does not contain key 'cams'. Keys: {list(ann.keys())}")

    ims_records = _as_list(ann["ims"])
    cams = ann["cams"]

    cam_count = None
    if isinstance(cams, dict):
        if "K" in cams:
            cam_count = len(cams["K"])
        elif "R" in cams:
            cam_count = len(cams["R"])
        elif "T" in cams:
            cam_count = len(cams["T"])

    first_record = ims_records[0]
    first_ims = _as_list(first_record["ims"])
    if cam_count is None:
        cam_count = len(first_ims)

    samples = []
    for cam_idx, rel in enumerate(first_ims[: args.max_print]):
        img_path = seq_dir / rel
        mask_path = find_mask(seq_dir, rel)
        samples.append({
            "camera_index": cam_idx,
            "relative_image_path": rel,
            "image_exists": img_path.exists(),
            "image_shape": read_image_shape(img_path) if img_path.exists() else None,
            "mask_path": str(mask_path) if mask_path else None,
            "mask_exists": bool(mask_path and mask_path.exists()),
        })

    report = {
        "zju_root": str(zju_root),
        "sequence_dir": str(seq_dir),
        "annots_path": str(ann_path),
        "annots_keys": list(ann.keys()),
        "num_frames": len(ims_records),
        "num_cameras_estimated": cam_count,
        "first_frame_num_image_paths": len(first_ims),
        "sample_entries_from_first_frame": samples,
        "has_new_params_dir": (seq_dir / "new_params").exists(),
        "has_params_dir": (seq_dir / "params").exists(),
        "has_mask_cihp_dir": (seq_dir / "mask_cihp").exists(),
        "has_mask_dir": (seq_dir / "mask").exists(),
    }

    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[OK] saved inspection report to {out}")


if __name__ == "__main__":
    main()
