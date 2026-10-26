#!/usr/bin/env python3
"""
Evaluate rendered video against held-out GT video.

Metrics:
  - PSNR / SSIM on full frame
  - foreground-crop PSNR / SSIM if a mask folder is provided
  - LPIPS if --use_lpips is enabled and the lpips package is installed
  - simple temporal metrics:
      temporal_delta_l1_mean: how much the rendered video changes frame to frame
      temporal_delta_error_l1_mean: difference between rendered temporal changes and GT temporal changes

This script is intended for WK2 validation, not final camera-calibrated novel-view rendering.
"""

import argparse
import json
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def read_video_rgb(path: Path, max_frames: int = 0) -> List[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    frames: List[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if max_frames and len(frames) >= max_frames:
            break
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames read from {path}")
    return frames


def list_masks(mask_dir: str) -> List[Path]:
    if not mask_dir:
        return []
    p = Path(mask_dir)
    if not p.exists():
        return []
    masks = sorted(list(p.glob("*.png")) + list(p.glob("*.jpg")) + list(p.glob("*.jpeg")))
    return masks


def resize_to_match(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    if pred.shape[:2] == gt.shape[:2]:
        return pred
    return cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_AREA)


def read_mask(mask_path: Path, target_hw: Tuple[int, int]) -> Optional[np.ndarray]:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    h, w = target_hw
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    return (mask > 0).astype(np.uint8)


def crop_by_mask(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray, pad: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    if mask.sum() == 0:
        return pred, gt
    x, y, w, h = cv2.boundingRect(mask)
    y0 = max(0, y - pad)
    x0 = max(0, x - pad)
    y1 = min(gt.shape[0], y + h + pad)
    x1 = min(gt.shape[1], x + w + pad)
    return pred[y0:y1, x0:x1], gt[y0:y1, x0:x1]


def calc_psnr(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(peak_signal_noise_ratio(gt, pred, data_range=255))


def calc_ssim(pred: np.ndarray, gt: np.ndarray) -> float:
    h, w = gt.shape[:2]
    # skimage needs window size <= min(h, w) and odd.
    if min(h, w) < 7:
        return float("nan")
    return float(structural_similarity(gt, pred, channel_axis=2, data_range=255))


def to_lpips_tensor(img: np.ndarray, device: torch.device) -> torch.Tensor:
    arr = img.astype(np.float32) / 127.5 - 1.0
    ten = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
    return ten


def try_create_lpips(device: torch.device):
    try:
        import lpips  # type: ignore
        net = lpips.LPIPS(net="alex").to(device)
        net.eval()
        return net
    except Exception as exc:
        print(f"[WARN] LPIPS disabled because it could not be initialized: {exc}")
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rendered", required=True, help="Path to rendered video")
    parser.add_argument("--gt", required=True, help="Path to held-out ground-truth video")
    parser.add_argument("--mask_dir", default="", help="Optional folder with held-out masks")
    parser.add_argument("--out_dir", required=True, help="Output directory for metrics")
    parser.add_argument("--max_frames", type=int, default=0, help="0 means compare all shared frames")
    parser.add_argument("--use_lpips", action="store_true", help="Enable LPIPS if installed")
    parser.add_argument("--device", default="cuda", help="cuda or cpu for LPIPS")
    args = parser.parse_args()

    rendered_path = Path(args.rendered)
    gt_path = Path(args.gt)
    if not rendered_path.exists():
        raise FileNotFoundError(f"Missing rendered video: {rendered_path}")
    if not gt_path.exists():
        raise FileNotFoundError(f"Missing GT video: {gt_path}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_frames = read_video_rgb(rendered_path, max_frames=args.max_frames)
    gt_frames = read_video_rgb(gt_path, max_frames=args.max_frames)
    n = min(len(pred_frames), len(gt_frames))
    if n <= 0:
        raise RuntimeError("No overlapping frames to compare.")

    masks = list_masks(args.mask_dir)

    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    lpips_net = try_create_lpips(device) if args.use_lpips else None

    rows = []
    temporal_delta_l1 = []
    temporal_delta_error_l1 = []

    prev_pred = None
    prev_gt = None

    with torch.no_grad():
        for i in range(n):
            pred = resize_to_match(pred_frames[i], gt_frames[i])
            gt = gt_frames[i]

            row = {
                "frame": i,
                "psnr_full": calc_psnr(pred, gt),
                "ssim_full": calc_ssim(pred, gt),
            }

            if lpips_net is not None:
                pred_t = to_lpips_tensor(pred, device)
                gt_t = to_lpips_tensor(gt, device)
                row["lpips_full"] = float(lpips_net(pred_t, gt_t).item())

            if masks and i < len(masks):
                mask = read_mask(masks[i], gt.shape[:2])
                if mask is not None:
                    pred_crop, gt_crop = crop_by_mask(pred, gt, mask)
                    if pred_crop.size and gt_crop.size:
                        row["psnr_fg_crop"] = calc_psnr(pred_crop, gt_crop)
                        row["ssim_fg_crop"] = calc_ssim(pred_crop, gt_crop)
                        if lpips_net is not None and min(gt_crop.shape[:2]) >= 16:
                            pred_ct = to_lpips_tensor(pred_crop, device)
                            gt_ct = to_lpips_tensor(gt_crop, device)
                            row["lpips_fg_crop"] = float(lpips_net(pred_ct, gt_ct).item())

            if prev_pred is not None and prev_gt is not None:
                pred_delta = pred.astype(np.float32) - prev_pred.astype(np.float32)
                gt_delta = gt.astype(np.float32) - prev_gt.astype(np.float32)
                temporal_delta_l1.append(float(np.mean(np.abs(pred_delta))))
                temporal_delta_error_l1.append(float(np.mean(np.abs(pred_delta - gt_delta))))

            prev_pred = pred
            prev_gt = gt
            rows.append(row)

    df = pd.DataFrame(rows)
    summary = {
        "num_frames_compared": n,
        "rendered": str(rendered_path),
        "gt": str(gt_path),
        "mask_dir": args.mask_dir,
        "psnr_full_mean": float(df["psnr_full"].mean()),
        "ssim_full_mean": float(df["ssim_full"].mean()),
        "temporal_delta_l1_mean": float(np.mean(temporal_delta_l1)) if temporal_delta_l1 else None,
        "temporal_delta_error_l1_mean": float(np.mean(temporal_delta_error_l1)) if temporal_delta_error_l1 else None,
    }

    for col in ["psnr_fg_crop", "ssim_fg_crop", "lpips_full", "lpips_fg_crop"]:
        if col in df.columns:
            summary[f"{col}_mean"] = float(df[col].mean())

    df.to_csv(out_dir / "per_frame_metrics.csv", index=False)
    (out_dir / "summary_metrics.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[OK] saved: {out_dir / 'per_frame_metrics.csv'}")
    print(f"[OK] saved: {out_dir / 'summary_metrics.json'}")


if __name__ == "__main__":
    main()
