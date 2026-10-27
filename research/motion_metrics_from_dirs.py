"""Recompute motion smoothness metrics from saved SMPL-X parameter folders.

Uses the same definitions as the project's postprocess module: per-frame second
difference of the 6D joint rotation representation and of metric translation.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def rodrigues(vec: np.ndarray) -> np.ndarray:
    theta = np.linalg.norm(vec, axis=-1, keepdims=True)
    k = vec / np.clip(theta, 1e-12, None)
    K = np.zeros(vec.shape[:-1] + (3, 3))
    K[..., 0, 1], K[..., 0, 2] = -k[..., 2], k[..., 1]
    K[..., 1, 0], K[..., 1, 2] = k[..., 2], -k[..., 0]
    K[..., 2, 0], K[..., 2, 1] = -k[..., 1], k[..., 0]
    I = np.broadcast_to(np.eye(3), K.shape)
    s = np.sin(theta)[..., None]
    c = np.cos(theta)[..., None]
    return I + s * K + (1 - c) * (K @ K)


def load(folder: Path):
    poses, trans = [], []
    for f in sorted(folder.glob("*.json")):
        d = json.loads(f.read_text())
        p = np.concatenate([
            np.asarray(d["root_pose"], dtype=np.float64),
            np.asarray(d["body_pose"], dtype=np.float64).reshape(-1),
            np.asarray(d["jaw_pose"], dtype=np.float64),
            np.asarray(d["leye_pose"], dtype=np.float64),
            np.asarray(d["reye_pose"], dtype=np.float64),
            np.asarray(d["lhand_pose"], dtype=np.float64).reshape(-1),
            np.asarray(d["rhand_pose"], dtype=np.float64).reshape(-1),
        ])
        poses.append(p.reshape(-1, 3))
        trans.append(np.asarray(d["trans"], dtype=np.float64))
    return np.stack(poses), np.stack(trans)          # [T, J, 3], [T, 3]


def sixd(poses: np.ndarray) -> np.ndarray:
    R = rodrigues(poses)
    return R[..., :, :2].reshape(len(poses), -1)      # first two columns per joint


def second_diff(x: np.ndarray) -> np.ndarray:
    return x[2:] - 2 * x[1:-1] + x[:-2]


def metrics(folder: Path, raw_folder: Path | None) -> dict:
    poses, trans = load(folder)
    p6, r6 = sixd(poses), None
    out = {
        "frames": int(len(poses)),
        "jitter_pose": float(np.linalg.norm(second_diff(p6), axis=-1).mean()) if len(poses) > 2 else 0.0,
        "jitter_trans": float(np.linalg.norm(second_diff(trans), axis=-1).mean()) if len(trans) > 2 else 0.0,
        "endpoint_displacement_m": float(np.linalg.norm(trans[-1] - trans[0])),
    }
    if raw_folder is not None:
        raw_poses, raw_trans = load(raw_folder)
        rp6 = sixd(raw_poses)
        out["deviation_pose"] = float(np.linalg.norm(p6 - rp6, axis=-1).mean())
        out["deviation_trans"] = float(np.linalg.norm(trans - raw_trans, axis=-1).mean())
    return out


def main() -> None:
    base = Path(sys.argv[1])
    mapping = json.loads(Path(sys.argv[2]).read_text())
    rows = []
    for seq, dirs in mapping.items():
        raw = Path(dirs["raw"]) if dirs.get("raw") else None
        for variant in ("baseline", "ours_full", "pose_only", "trans_only"):
            folder = Path(dirs[variant]) if dirs.get(variant) else None
            if not folder or not folder.exists():
                continue
            row = {"sequence": seq, "variant": variant, **metrics(folder, raw)}
            rows.append(row)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    print(",".join(keys))
    for r in rows:
        print(",".join(str(round(r[k], 8)) if isinstance(r[k], float) else str(r[k]) for k in keys))


if __name__ == "__main__":
    main()
