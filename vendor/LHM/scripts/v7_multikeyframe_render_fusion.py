#!/usr/bin/env python3
"""Prototype multi-keyframe fusion at render/score level.

This is NOT true canonical 3DGS fusion. It expects candidate rendered videos from different
source keyframes, evaluates each against held-out GT, and outputs either the best video or a
simple temporal ensemble. It makes the multi-keyframe branch useful before LHM internals expose
canonical Gaussian tensors.
"""
from __future__ import annotations

import argparse
import csv
import math
import shutil
from pathlib import Path
from typing import List

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


def write_video(frames: List[np.ndarray], path: str, fps: float = 30.0) -> None:
    if not frames:
        raise ValueError("empty frames")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        if f.shape[:2] != (h, w):
            f = cv2.resize(f, (w, h), interpolation=cv2.INTER_AREA)
        writer.write(f)
    writer.release()


def psnr(a, b):
    a = a.astype(np.float32); b = b.astype(np.float32)
    mse = np.mean((a - b) ** 2)
    if mse <= 1e-12:
        return 99.0
    return 20 * math.log10(255.0 / math.sqrt(float(mse)))


def sharpness(frames: List[np.ndarray]) -> float:
    vals = []
    for f in frames[: min(len(frames), 30)]:
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        vals.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
    return float(np.mean(vals)) if vals else 0.0


def score_candidate(video: Path, gt_frames: List[np.ndarray], max_frames: int) -> dict:
    frames = read_video(str(video), max_frames)
    n = min(len(frames), len(gt_frames))
    if n == 0:
        return {"video": str(video), "frames": 0, "psnr": -1.0, "sharpness": 0.0, "score": -1.0}
    vals = []
    for i in range(n):
        a = frames[i]
        b = gt_frames[i]
        if a.shape[:2] != b.shape[:2]:
            a = cv2.resize(a, (b.shape[1], b.shape[0]), interpolation=cv2.INTER_AREA)
        vals.append(psnr(a, b))
    p = float(np.mean(vals))
    sh = sharpness(frames)
    # PSNR is weak before true camera alignment, so keep sharpness as a small tie-breaker.
    score = p + 0.0001 * sh
    return {"video": str(video), "frames": n, "psnr": p, "sharpness": sh, "score": score}


def make_mean_ensemble(videos: List[Path], out_path: str, max_frames: int, fps: float = 30.0) -> None:
    seqs = [read_video(str(v), max_frames) for v in videos]
    n = min(len(s) for s in seqs if s)
    if n == 0:
        raise ValueError("no frames for ensemble")
    base_h, base_w = seqs[0][0].shape[:2]
    out_frames = []
    for i in range(n):
        stack = []
        for s in seqs:
            f = s[i]
            if f.shape[:2] != (base_h, base_w):
                f = cv2.resize(f, (base_w, base_h), interpolation=cv2.INTER_AREA)
            stack.append(f.astype(np.float32))
        out_frames.append(np.clip(np.mean(stack, axis=0), 0, 255).astype(np.uint8))
    write_video(out_frames, out_path, fps=fps)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate_dir", required=True, help="directory containing candidate .mp4 videos from different keyframes")
    ap.add_argument("--gt", required=True)
    ap.add_argument("--out_dir", default="/autodl-fs/data/lhm_shared/poster_results/v7_multikeyframe_fusion")
    ap.add_argument("--mode", choices=["best", "mean"], default="best")
    ap.add_argument("--max_frames", type=int, default=120)
    args = ap.parse_args()

    cand_dir = Path(args.candidate_dir)
    videos = sorted(cand_dir.glob("*.mp4"))
    if not videos:
        raise SystemExit(f"No .mp4 candidates found in {cand_dir}")
    gt_frames = read_video(args.gt, args.max_frames)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    rows = [score_candidate(v, gt_frames, args.max_frames) for v in videos]
    rows = sorted(rows, key=lambda r: r["score"], reverse=True)
    csv_path = out_dir / "candidate_scores.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)

    out_video = out_dir / "multikeyframe_fusion_prototype.mp4"
    if args.mode == "best":
        shutil.copy2(rows[0]["video"], out_video)
    else:
        top = [Path(r["video"]) for r in rows[: min(3, len(rows))]]
        make_mean_ensemble(top, str(out_video), args.max_frames)

    print(f"[OK] wrote scores: {csv_path}")
    print(f"[OK] fusion prototype output: {out_video}")
    print(f"[best] {rows[0]}")


if __name__ == "__main__":
    main()
