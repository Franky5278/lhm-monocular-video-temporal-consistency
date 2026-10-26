from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import torch


def read_video_frames(path: str, max_frames: Optional[int] = None, resize_to: Optional[Tuple[int, int]] = None) -> np.ndarray:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if resize_to is not None:
            frame = cv2.resize(frame, resize_to, interpolation=cv2.INTER_AREA)
        frames.append(frame)
        if max_frames is not None and len(frames) >= max_frames:
            break
    cap.release()
    if len(frames) == 0:
        raise ValueError(f"No frames in video: {path}")
    return np.stack(frames, axis=0)


def ssim_gray(x: np.ndarray, y: np.ndarray) -> float:
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    mu_x = cv2.GaussianBlur(x, (11, 11), 1.5)
    mu_y = cv2.GaussianBlur(y, (11, 11), 1.5)
    sigma_x = cv2.GaussianBlur(x * x, (11, 11), 1.5) - mu_x * mu_x
    sigma_y = cv2.GaussianBlur(y * y, (11, 11), 1.5) - mu_y * mu_y
    sigma_xy = cv2.GaussianBlur(x * y, (11, 11), 1.5) - mu_x * mu_y
    m = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / ((mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2) + 1e-8)
    return float(np.mean(m))


def compute_render_video_metrics(
    pred_video: str,
    gt_video: Optional[str],
    max_frames: int = 120,
    try_lpips: bool = True,
) -> Dict[str, object]:
    nan = float("nan")
    base = {
        "pred_video": pred_video,
        "gt_video": gt_video,
        "valid_eval": False,
        "psnr": nan,
        "ssim": nan,
        "lpips": nan,
        "num_eval_frames": 0,
        "note": "held-out GT missing or invalid",
    }
    if not pred_video or not os.path.exists(pred_video):
        base["note"] = "pred_video missing"
        return base
    if not gt_video or not os.path.exists(gt_video):
        return base

    pred = read_video_frames(pred_video, max_frames=max_frames)
    h, w = pred.shape[1:3]
    gt = read_video_frames(gt_video, max_frames=max_frames, resize_to=(w, h))
    n = min(len(pred), len(gt))
    pred = pred[:n]
    gt = gt[:n]

    mse = np.mean((pred.astype(np.float32) - gt.astype(np.float32)) ** 2, axis=(1, 2, 3))
    psnr = float(np.mean([20 * math.log10(255.0 / math.sqrt(max(float(m), 1e-8))) for m in mse]))
    ssims = []
    for i in range(n):
        p_gray = cv2.cvtColor(pred[i], cv2.COLOR_RGB2GRAY)
        g_gray = cv2.cvtColor(gt[i], cv2.COLOR_RGB2GRAY)
        ssims.append(ssim_gray(p_gray, g_gray))

    lpips_val = nan
    if try_lpips:
        try:
            import lpips  # type: ignore
            device = "cuda" if torch.cuda.is_available() else "cpu"
            loss_fn = lpips.LPIPS(net="alex").to(device).eval()
            vals = []
            with torch.no_grad():
                for i in range(n):
                    p = torch.from_numpy(pred[i]).float().permute(2, 0, 1).unsqueeze(0) / 127.5 - 1
                    g = torch.from_numpy(gt[i]).float().permute(2, 0, 1).unsqueeze(0) / 127.5 - 1
                    vals.append(float(loss_fn(p.to(device), g.to(device)).item()))
            lpips_val = float(np.mean(vals))
        except Exception as e:
            lpips_val = nan

    return {
        "pred_video": pred_video,
        "gt_video": gt_video,
        "valid_eval": True,
        "psnr": psnr,
        "ssim": float(np.mean(ssims)),
        "lpips": lpips_val,
        "num_eval_frames": int(n),
        "note": "render-only video compared against held-out original video",
    }
