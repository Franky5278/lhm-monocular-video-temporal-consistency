from __future__ import annotations

import csv
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def _default_results_root() -> str:
    return os.environ.get(
        "LHM_URECA_RESULTS_ROOT",
        "/autodl-fs/data/lhm_shared/ureca_results",
    )


def now_tag() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def make_experiment_dir(root: Optional[str] = None, prefix: str = "exp") -> str:
    root = root or _default_results_root()
    Path(root).mkdir(parents=True, exist_ok=True)
    base = Path(root) / f"{prefix}_{now_tag()}"
    path = base
    k = 1
    while path.exists():
        path = Path(f"{base}_{k:02d}")
        k += 1
    path.mkdir(parents=True, exist_ok=False)
    (path / "selected_keyframes").mkdir(exist_ok=True)
    (path / "render_only").mkdir(exist_ok=True)
    (path / "reports").mkdir(exist_ok=True)
    return str(path)


def save_json(obj: Dict[str, Any], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def save_csv(rows: List[Dict[str, Any]], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with open(path, "w", encoding="utf-8") as f:
            f.write("")
        return
    keys = []
    for r in rows:
        for k in r.keys():
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def copy_if_exists(src: Optional[str], dst: str) -> Optional[str]:
    if not src or not os.path.exists(src):
        return None
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copy2(src, dst)
        return dst
    except Exception:
        return None


def dataframe_rows_from_render_metrics(metrics: Dict[str, Dict[str, Any]]) -> List[List[Any]]:
    rows: List[List[Any]] = []
    for variant, m in metrics.items():
        rows.append([
            variant,
            bool(m.get("valid_eval", False)),
            m.get("psnr", float("nan")),
            m.get("ssim", float("nan")),
            m.get("lpips", float("nan")),
            int(m.get("num_eval_frames", 0) or 0),
        ])
    return rows
