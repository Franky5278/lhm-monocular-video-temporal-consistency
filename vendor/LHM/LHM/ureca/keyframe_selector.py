from __future__ import annotations

import glob
import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

IMAGE_EXTS = ("*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp")


def _read_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _to_np(x, default=None) -> np.ndarray:
    if x is None:
        if default is None:
            return np.zeros((0,), dtype=np.float32)
        return np.asarray(default, dtype=np.float32)
    return np.asarray(x, dtype=np.float32)


def load_smplx_motion_np(smplx_dir: str) -> Tuple[List[str], np.ndarray, np.ndarray]:
    files = sorted(glob.glob(os.path.join(smplx_dir, "*.json")))
    if len(files) == 0:
        raise FileNotFoundError(f"No json files found in smplx_dir: {smplx_dir}")

    pose_keys = [
        "root_pose", "global_orient", "body_pose", "jaw_pose", "leye_pose", "reye_pose",
        "lhand_pose", "rhand_pose", "expr",
    ]
    poses, trans = [], []
    for fp in files:
        d = _read_json(fp)
        parts = []
        for k in pose_keys:
            val = d.get(k, None)
            arr = _to_np(val).reshape(-1)
            if arr.size > 0:
                parts.append(arr)
        poses.append(np.concatenate(parts, axis=0) if parts else np.zeros((1,), dtype=np.float32))
        tr = d.get("trans", d.get("transl", [0.0, 0.0, 0.0]))
        trans.append(_to_np(tr, default=[0, 0, 0]).reshape(-1)[:3])

    max_d = max(p.shape[0] for p in poses)
    poses_pad = np.zeros((len(poses), max_d), dtype=np.float32)
    for i, p in enumerate(poses):
        poses_pad[i, : p.shape[0]] = p
    return files, poses_pad, np.asarray(trans, dtype=np.float32)


def select_keyframes_from_smplx(
    smplx_dir: str,
    num_keyframes: int = 5,
    min_gap: int = 8,
    include_first: bool = True,
    include_last: bool = True,
    save_json: Optional[str] = None,
) -> Dict[str, object]:
    files, poses, trans = load_smplx_motion_np(smplx_dir)
    T = len(files)
    if T <= max(1, num_keyframes):
        idxs = list(range(T))
    else:
        pose_std = float(np.std(poses, axis=0).mean() + 1e-6)
        trans_std = float(np.std(trans, axis=0).mean() + 1e-6)
        feat = np.concatenate([poses / pose_std, (trans / trans_std) * 2.0], axis=1)
        selected: List[int] = []
        if include_first:
            selected.append(0)
        if include_last and T - 1 not in selected:
            selected.append(T - 1)
        while len(selected) < min(num_keyframes, T):
            best_i, best_score = None, -1.0
            for i in range(T):
                if i in selected:
                    continue
                if any(abs(i - j) < min_gap for j in selected):
                    continue
                dist = min(float(np.linalg.norm(feat[i] - feat[j])) for j in selected) if selected else 0.0
                center_bonus = 1.0 - abs(i - (T - 1) / 2.0) / max((T - 1) / 2.0, 1.0)
                score = dist + 0.05 * center_bonus
                if score > best_score:
                    best_i, best_score = i, score
            if best_i is None:
                remaining = [i for i in range(T) if i not in selected]
                if not remaining:
                    break
                best_i = remaining[len(remaining) // 2]
            selected.append(int(best_i))
        idxs = sorted(selected[:num_keyframes])
    result = {"smplx_dir": smplx_dir, "num_frames": T, "indices": idxs, "files": [files[i] for i in idxs]}
    if save_json:
        Path(save_json).parent.mkdir(parents=True, exist_ok=True)
        with open(save_json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
    return result


def list_image_files(folder: str) -> List[str]:
    files: List[str] = []
    for ext in IMAGE_EXTS:
        files.extend(glob.glob(os.path.join(folder, ext)))
    return sorted(files)


def blur_score_rgb(img_rgb: np.ndarray) -> float:
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _read_rgb(path: str) -> Optional[np.ndarray]:
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def extract_frames_from_video(video_path: str, out_dir: str, max_frames: int = 120, stride: int = 5) -> List[str]:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open source video: {video_path}")
    paths: List[str] = []
    i = 0
    kept = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % max(1, int(stride)) == 0:
            p = os.path.join(out_dir, f"frame_{i:06d}.png")
            cv2.imwrite(p, frame)
            paths.append(p)
            kept += 1
            if kept >= max_frames:
                break
        i += 1
    cap.release()
    return paths


def select_source_keyframes(
    source_path: str,
    out_dir: str,
    num_keyframes: int = 3,
    min_gap: int = 3,
    video_stride: int = 5,
    save_json: Optional[str] = None,
) -> Dict[str, object]:
    """Select source image keyframes for multi-frame LHM inference.

    source_path can be:
    - a folder of PNG/JPG images
    - a single image
    - an MP4/MOV video

    Selected images are copied into out_dir and returned as file paths.
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    if not source_path or not os.path.exists(source_path):
        raise FileNotFoundError(f"source_path does not exist: {source_path}")

    if os.path.isdir(source_path):
        candidates = list_image_files(source_path)
    else:
        ext = Path(source_path).suffix.lower()
        if ext in {".mp4", ".mov", ".avi", ".mkv", ".webm"}:
            candidates = extract_frames_from_video(source_path, os.path.join(out_dir, "extracted_frames"), stride=video_stride)
        else:
            candidates = [source_path]

    if len(candidates) == 0:
        raise FileNotFoundError(f"No source images found from: {source_path}")

    scores = []
    for idx, p in enumerate(candidates):
        img = _read_rgb(p)
        if img is None:
            continue
        blur = blur_score_rgb(img)
        # prefer sharp frames and a little bit of temporal spread
        center_bonus = 1.0 - abs(idx - (len(candidates) - 1) / 2.0) / max((len(candidates) - 1) / 2.0, 1.0)
        scores.append((idx, p, blur + 0.01 * center_bonus, blur))

    if not scores:
        raise RuntimeError(f"All source candidates failed to load from {source_path}")

    selected: List[Tuple[int, str, float, float]] = []
    for item in sorted(scores, key=lambda x: x[2], reverse=True):
        idx = item[0]
        if any(abs(idx - s[0]) < min_gap for s in selected):
            continue
        selected.append(item)
        if len(selected) >= min(num_keyframes, len(scores)):
            break
    if len(selected) < min(num_keyframes, len(scores)):
        for item in sorted(scores, key=lambda x: x[2], reverse=True):
            if item not in selected:
                selected.append(item)
            if len(selected) >= min(num_keyframes, len(scores)):
                break
    selected = sorted(selected, key=lambda x: x[0])

    copied = []
    frame_rows = []
    for k, (idx, src, score, blur) in enumerate(selected):
        dst = os.path.join(out_dir, f"source_keyframe_{k:02d}{Path(src).suffix.lower() or '.png'}")
        shutil.copy2(src, dst)
        copied.append(dst)
        frame_rows.append({"rank": k, "source_index": int(idx), "src": src, "path": dst, "score": float(score), "blur": float(blur)})

    result = {
        "source_path": source_path,
        "num_candidates": len(candidates),
        "num_selected": len(copied),
        "selected_paths": copied,
        "frames": frame_rows,
    }
    if save_json:
        Path(save_json).parent.mkdir(parents=True, exist_ok=True)
        with open(save_json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
    return result
