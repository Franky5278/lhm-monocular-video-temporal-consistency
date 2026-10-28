"""Fair comparison against standard temporal filters at a matched fidelity budget.

The pipeline's stabilizer is compared with two off-the-shelf alternatives that get
the same input (the raw SMPL-X motion) and the same metric definitions:

  * moving average (centred, no phase lag) on the 6D rotation and translation
  * One-Euro filter (Casiez et al. 2012), a standard low-latency smoother

For each method a strength sweep traces a (deviation, jitter) Pareto front. The
honest question is not "does smoothing reduce jitter" (any filter does) but "at the
same deviation cost, which filter removes more jitter".
"""
from __future__ import annotations

import argparse
import json
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
    return I + np.sin(theta)[..., None] * K + (1 - np.cos(theta)[..., None]) * (K @ K)


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
    return np.stack(poses), np.stack(trans)


def to6d(poses: np.ndarray) -> np.ndarray:
    R = rodrigues(poses)
    return R[..., :, :2].reshape(len(poses), -1)


def second_diff(x: np.ndarray) -> np.ndarray:
    return x[2:] - 2 * x[1:-1] + x[:-2]


def metrics(pred6: np.ndarray, pred_t: np.ndarray, raw6: np.ndarray, raw_t: np.ndarray) -> dict:
    j_pose = np.linalg.norm(second_diff(pred6), axis=-1).mean()
    j_trans = np.linalg.norm(second_diff(pred_t), axis=-1).mean()
    dev_pose = np.linalg.norm(pred6 - raw6, axis=-1).mean()
    dev_trans = np.linalg.norm(pred_t - raw_t, axis=-1).mean()
    return dict(jitter_pose=float(j_pose), jitter_trans=float(j_trans),
                deviation_pose=float(dev_pose), deviation_trans=float(dev_trans))


def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return x.copy()
    pad = window // 2
    padded = np.pad(x, ((pad, pad),) + ((0, 0),) * (x.ndim - 1), mode="edge")
    kernel = np.ones(window) / window
    out = np.apply_along_axis(lambda v: np.convolve(v, kernel, mode="valid"), 0, padded)
    return out[: len(x)]


def one_euro(x: np.ndarray, min_cutoff=1.0, beta=0.5, d_cutoff=1.0, freq=30.0) -> np.ndarray:
    def alpha(cutoff):
        tau = 1.0 / (2 * np.pi * cutoff)
        te = 1.0 / freq
        return 1.0 / (1.0 + tau / te)
    out = np.empty_like(x)
    x_prev = x[0].copy()
    dx_prev = np.zeros_like(x[0])
    a_d = alpha(d_cutoff)
    for i in range(len(x)):
        dx = (x[i] - x_prev) * freq
        dx_hat = a_d * dx + (1 - a_d) * dx_prev
        cutoff = min_cutoff + beta * np.abs(dx_hat)
        a = 1.0 / (1.0 + (1.0 / (2 * np.pi * np.maximum(cutoff, 1e-6))) * freq)
        x_hat = a * x[i] + (1 - a) * x_prev
        out[i] = x_hat
        x_prev, dx_prev = x_hat, dx_hat
    return out


def ema_forward(x: np.ndarray, a: float) -> np.ndarray:
    out = np.empty_like(x)
    acc = x[0].copy()
    for i in range(len(x)):
        acc = a * x[i] + (1 - a) * acc
        out[i] = acc
    return out


def ema_zero_phase(x: np.ndarray, a: float) -> np.ndarray:
    """Causal EMA applied forward then backward: same operator, no group delay.

    The causal EMA used by the pipeline trades jitter for lag; the backward pass
    cancels the lag, which is free in an offline render.
    """
    return ema_forward(ema_forward(x, a)[::-1], a)[::-1]


def orthonormalise_sixd(x: np.ndarray) -> np.ndarray:
    shape = x.shape
    cols = x.reshape(shape[0], -1, 3, 2)
    a, b = cols[..., 0], cols[..., 1]
    a = a / np.clip(np.linalg.norm(a, axis=-1, keepdims=True), 1e-9, None)
    b = b - (a * b).sum(-1, keepdims=True) * a
    b = b / np.clip(np.linalg.norm(b, axis=-1, keepdims=True), 1e-9, None)
    return np.stack([a, b], axis=-1).reshape(shape)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="raw smplx_params folder")
    ap.add_argument("--ours", required=True, help="our stabilized folder")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    raw_poses, raw_trans = load(Path(args.raw))
    our_poses, our_trans = load(Path(args.ours))
    raw6, our6 = to6d(raw_poses), to6d(our_poses)

    rows = [{"method": "raw (no filter)", **metrics(raw6, raw_trans, raw6, raw_trans)}]
    rows.append({"method": "ours (pose EMA + trans-velocity EMA)",
                 **metrics(our6, our_trans, raw6, raw_trans)})

    for w in (3, 5, 9, 15, 25):
        f6 = orthonormalise_sixd(moving_average(raw6, w))
        ft = moving_average(raw_trans, w)
        rows.append({"method": f"moving average w={w}", **metrics(f6, ft, raw6, raw_trans)})

    for mc in (0.5, 1.0, 2.0, 4.0):
        for beta in (0.0, 0.5, 1.0):
            f6 = orthonormalise_sixd(one_euro(raw6, min_cutoff=mc, beta=beta))
            ft = one_euro(raw_trans, min_cutoff=mc, beta=beta)
            rows.append({"method": f"one-euro mc={mc} beta={beta}",
                         **metrics(f6, ft, raw6, raw_trans)})

    for a in (0.05, 0.10, 0.20, 0.30, 0.50):
        f6 = orthonormalise_sixd(ema_zero_phase(raw6, a))
        ft = ema_zero_phase(raw_trans, a)
        rows.append({"method": f"zero-phase EMA a={a}",
                     **metrics(f6, ft, raw6, raw_trans)})

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0])
    with out.open("w", encoding="utf8", newline="") as fh:
        fh.write(",".join(keys) + "\n")
        for r in rows:
            fh.write(",".join(f"{r[k]:.8f}" if isinstance(r[k], float) else str(r[k]) for k in keys) + "\n")

    # Matched-budget: compare each method at the deviation cost of our filter.
    target = rows[1]["deviation_pose"]
    best = {}
    for r in rows[2:]:
        rel = abs(r["deviation_pose"] - target) / target
        if r["method"].split()[0] not in best or rel < best[r["method"].split()[0]][0]:
            best[r["method"].split()[0]] = (rel, r)
    print(f"our filter: deviation={target:.6f} jitter={rows[1]['jitter_pose']:.6f}")
    for name, (rel, r) in best.items():
        print(f"  {name:14s} closest point {r['method']:28s} dev={r['deviation_pose']:.6f} "
              f"({rel*100:5.1f}% off) jitter={r['jitter_pose']:.6f} "
              f"-> {r['jitter_pose']/rows[1]['jitter_pose']:.2f}x ours")
    print(f"wrote {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
