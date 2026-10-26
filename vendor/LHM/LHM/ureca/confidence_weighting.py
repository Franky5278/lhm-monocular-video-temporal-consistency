from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


def _safe_norm(xs: Sequence[float], eps: float = 1e-8) -> List[float]:
    arr = np.asarray(xs, dtype=np.float64)
    if arr.size == 0:
        return []
    arr = arr - arr.min()
    if float(arr.max()) < eps:
        arr = np.ones_like(arr)
    else:
        arr = arr / (arr.max() + eps) + 0.1
    arr = arr / (arr.sum() + eps)
    return [float(x) for x in arr]


def frame_sharpness(path: str) -> float:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        return 0.0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def compute_source_confidence_weights(
    image_paths: Sequence[str],
    mode: str = "sharpness",
) -> Tuple[List[float], Dict[str, object]]:
    if len(image_paths) == 0:
        return [], {"mode": mode, "weights": [], "note": "empty image_paths"}
    if mode == "uniform":
        weights = [1.0 / len(image_paths)] * len(image_paths)
        return weights, {"mode": mode, "weights": weights}
    sharp = [frame_sharpness(p) for p in image_paths]
    weights = _safe_norm(sharp)
    return weights, {
        "mode": mode,
        "weights": weights,
        "sharpness": [float(x) for x in sharp],
        "note": "currently sharpness-based confidence; can be replaced by pose/mask confidence later",
    }
