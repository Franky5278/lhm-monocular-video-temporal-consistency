#!/usr/bin/env python3
"""Make a four-way comparison video for poster/debug visualization."""
import argparse
from pathlib import Path
import cv2
import numpy as np


def read_frames(path):
    cap = cv2.VideoCapture(str(path))
    frames = []
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames, fps


def fit(frame, w, h):
    if frame is None:
        return np.full((h, w, 3), 255, dtype=np.uint8)
    ih, iw = frame.shape[:2]
    scale = min(w / iw, h / ih)
    nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.full((h, w, 3), 255, dtype=np.uint8)
    x0, y0 = (w - nw) // 2, (h - nh) // 2
    canvas[y0:y0+nh, x0:x0+nw] = resized
    return canvas


def label(frame, text):
    out = frame.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 34), (255, 255, 255), -1)
    cv2.putText(out, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2, cv2.LINE_AA)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', required=True)
    ap.add_argument('--gt', required=True)
    ap.add_argument('--baseline', required=True)
    ap.add_argument('--improved', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--cell_w', type=int, default=360)
    ap.add_argument('--cell_h', type=int, default=360)
    args = ap.parse_args()

    videos = []
    for p in [args.source, args.gt, args.baseline, args.improved]:
        frames, fps = read_frames(p)
        videos.append(frames)
    n = min(len(v) for v in videos if v)
    if n <= 0:
        raise RuntimeError('No frames available for comparison')
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(outp), cv2.VideoWriter_fourcc(*'mp4v'), 30, (args.cell_w*2, args.cell_h*2))
    names = ['Source / target motion', 'Held-out GT', 'Single-keyframe LHM', 'Temporal-stabilized / restored']
    for i in range(n):
        cells = [label(fit(videos[j][i], args.cell_w, args.cell_h), names[j]) for j in range(4)]
        top = np.concatenate([cells[0], cells[1]], axis=1)
        bottom = np.concatenate([cells[2], cells[3]], axis=1)
        grid = np.concatenate([top, bottom], axis=0)
        writer.write(grid)
    writer.release()
    print(str(outp))

if __name__ == '__main__':
    main()
