"""Compose side-by-side comparison clips for the README (no GPU needed)."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def read_video(path: Path, size: int) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA))
    cap.release()
    return frames


def label(frame: np.ndarray, text: str) -> np.ndarray:
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except Exception:
        font = ImageFont.load_default()
    draw.rectangle([0, 0, img.width, 22], fill=(0, 0, 0))
    draw.text((6, 3), text, fill=(255, 255, 255), font=font)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--panels", nargs="+", required=True,
                    help="label=path pairs, in display order")
    args = ap.parse_args()

    stacks = []
    for item in args.panels:
        name, path = item.split("=", 1)
        frames = read_video(Path(path), args.size)
        if not frames:
            raise SystemExit(f"no frames in {path}")
        stacks.append((name, frames))

    total = min(len(f) for _, f in stacks)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    gif_frames = []
    writer = None
    for i in range(total):
        row = np.hstack([label(frames[i], f"{name}  f{i:03d}") for name, frames in stacks])
        if writer is None:
            writer = cv2.VideoWriter(str(out.with_suffix(".mp4")),
                                     cv2.VideoWriter_fourcc(*"mp4v"), args.fps,
                                     (row.shape[1], row.shape[0]))
        writer.write(row)
        gif_frames.append(Image.fromarray(cv2.cvtColor(row, cv2.COLOR_BGR2RGB)))
    if writer is not None:
        writer.release()
    if gif_frames:
        gif_frames[0].save(out, save_all=True, append_images=gif_frames[1:],
                           duration=int(1000 / args.fps), loop=0, optimize=True)
    print(f"{out} panels={len(stacks)} frames={total}")


if __name__ == "__main__":
    main()
