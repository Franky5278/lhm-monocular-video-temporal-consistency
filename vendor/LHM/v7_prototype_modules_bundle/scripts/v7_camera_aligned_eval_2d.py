#!/usr/bin/env python3
"""Prototype camera-aligned held-out evaluation.

This is NOT true 3D camera K/R/T rendering. It performs a 2D foreground/mask-aligned
comparison between rendered and held-out videos to make the evaluation less dominated by
scale/position mismatch. It is useful as a prototype until the LHM renderer is wired to ZJU
held-out camera intrinsics/extrinsics.
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np


def read_video(path: str, max_frames: int = 0) -> List[np.ndarray]:
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
        if max_frames and len(frames) >= max_frames:
            break
    cap.release()
    return frames


def sorted_masks(mask_dir: Optional[str]) -> List[Path]:
    if not mask_dir:
        return []
    root = Path(mask_dir)
    if not root.exists():
        return []
    out = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp"):
        out.extend(root.glob(ext))
    return sorted(out)


def foreground_bbox_from_mask(mask: np.ndarray, pad: int = 10) -> Optional[Tuple[int, int, int, int]]:
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    ys, xs = np.where(mask > 20)
    if len(xs) < 20 or len(ys) < 20:
        return None
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    h, w = mask.shape[:2]
    return max(0, x1 - pad), max(0, y1 - pad), min(w, x2 + pad + 1), min(h, y2 + pad + 1)


def foreground_bbox_from_image(frame: np.ndarray, pad: int = 10) -> Optional[Tuple[int, int, int, int]]:
    # rendered LHM videos often use near-white background; find non-white foreground
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mask = gray < 245
    ys, xs = np.where(mask)
    if len(xs) < 20 or len(ys) < 20:
        return None
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    h, w = gray.shape[:2]
    return max(0, x1 - pad), max(0, y1 - pad), min(w, x2 + pad + 1), min(h, y2 + pad + 1)


def crop(frame: np.ndarray, box: Optional[Tuple[int, int, int, int]]) -> np.ndarray:
    if box is None:
        return frame
    x1, y1, x2, y2 = box
    return frame[y1:y2, x1:x2]


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    mse = np.mean((a - b) ** 2)
    if mse <= 1e-12:
        return 99.0
    return 20.0 * math.log10(255.0 / math.sqrt(float(mse)))


def ssim_simple(a: np.ndarray, b: np.ndarray) -> float:
    # global SSIM fallback, no skimage dependency
    a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype(np.float32)
    b = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY).astype(np.float32)
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    mu_a, mu_b = a.mean(), b.mean()
    var_a, var_b = a.var(), b.var()
    cov = ((a - mu_a) * (b - mu_b)).mean()
    return float(((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a**2 + mu_b**2 + c1) * (var_a + var_b + c2)))


def temporal_delta_error(frames_a: List[np.ndarray], frames_b: List[np.ndarray]) -> float:
    vals = []
    for i in range(1, min(len(frames_a), len(frames_b))):
        da = cv2.absdiff(frames_a[i], frames_a[i - 1]).astype(np.float32) / 255.0
        db = cv2.absdiff(frames_b[i], frames_b[i - 1]).astype(np.float32) / 255.0
        vals.append(float(np.mean(np.abs(da - db))))
    return float(np.mean(vals)) if vals else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rendered", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--mask_dir", default="")
    ap.add_argument("--out_csv", default="/autodl-fs/data/lhm_shared/poster_results/v7_camera_aligned_eval_2d.csv")
    ap.add_argument("--max_frames", type=int, default=120)
    ap.add_argument("--crop_size", type=int, default=512)
    args = ap.parse_args()

    rendered = read_video(args.rendered, args.max_frames)
    gt = read_video(args.gt, args.max_frames)
    masks = sorted_masks(args.mask_dir)
    n = min(len(rendered), len(gt))
    if n == 0:
        raise SystemExit("No comparable frames.")

    aligned_rendered = []
    aligned_gt = []
    psnrs, ssims = [], []
    for i in range(n):
        r = rendered[i]
        g = gt[i]
        r_box = foreground_bbox_from_image(r)
        if i < len(masks):
            m = cv2.imread(str(masks[i]), cv2.IMREAD_GRAYSCALE)
            g_box = foreground_bbox_from_mask(m) if m is not None else foreground_bbox_from_image(g)
        else:
            g_box = foreground_bbox_from_image(g)
        rc = crop(r, r_box)
        gc = crop(g, g_box)
        rc = cv2.resize(rc, (args.crop_size, args.crop_size), interpolation=cv2.INTER_AREA)
        gc = cv2.resize(gc, (args.crop_size, args.crop_size), interpolation=cv2.INTER_AREA)
        aligned_rendered.append(rc)
        aligned_gt.append(gc)
        psnrs.append(psnr(rc, gc))
        ssims.append(ssim_simple(rc, gc))

    row = {
        "variant": "camera_aligned_2d_crop_prototype",
        "frames": n,
        "PSNR_2d_aligned_crop": float(np.mean(psnrs)),
        "SSIM_2d_aligned_crop": float(np.mean(ssims)),
        "temporal_delta_error_2d_aligned": temporal_delta_error(aligned_rendered, aligned_gt),
        "rendered": args.rendered,
        "gt": args.gt,
        "mask_dir": args.mask_dir,
        "note": "2D foreground-aligned prototype; true ZJU K/R/T renderer integration pending",
    }

    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)
    print(row)
    print(f"[OK] wrote {out}")


if __name__ == "__main__":
    main()
