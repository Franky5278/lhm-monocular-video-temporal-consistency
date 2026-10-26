#!/usr/bin/env python3
"""Level-1.5 diffusion/detail restoration prototype for rendered avatar videos.

This script does NOT train a diffusion model. It applies a local pretrained Diffusers
img2img model frame-by-frame to an already rendered video, then recomposes a restored
video. It also includes an unsharp_preview fallback for debugging pipeline wiring.
"""

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np


def read_bgr_video(path: str):
    cap = cv2.VideoCapture(str(path))
    frames = []
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames read from {path}")
    return frames, fps


def write_bgr_video(frames, path: str, fps: float):
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (w, h))
    for frame in frames:
        if frame.shape[:2] != (h, w):
            frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
        writer.write(frame)
    writer.release()


def unsharp_bgr(frame, amount=0.75, sigma=1.0):
    blur = cv2.GaussianBlur(frame, (0, 0), sigmaX=float(sigma), sigmaY=float(sigma))
    sharp = cv2.addWeighted(frame, 1.0 + float(amount), blur, -float(amount), 0)
    return np.clip(sharp, 0, 255).astype(np.uint8)


def restore_with_unsharp(frames, max_frames):
    restored = []
    for i, frame in enumerate(frames):
        restored.append(unsharp_bgr(frame) if i < max_frames else frame)
    return restored


def restore_with_diffusers_img2img(frames, model_path, prompt, negative_prompt, strength, guidance_scale, steps, max_frames):
    import torch
    from diffusers import StableDiffusionImg2ImgPipeline
    from PIL import Image

    if not model_path or not os.path.exists(model_path):
        raise FileNotFoundError(f"Local pretrained diffusion model path not found: {model_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        model_path,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
        local_files_only=True,
    ).to(device)
    try:
        pipe.enable_attention_slicing()
    except Exception:
        pass

    restored = []
    with torch.no_grad():
        for i, frame in enumerate(frames):
            if i >= max_frames:
                restored.append(frame)
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            out = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=pil,
                strength=float(strength),
                guidance_scale=float(guidance_scale),
                num_inference_steps=int(steps),
            ).images[0]
            out_bgr = cv2.cvtColor(np.array(out), cv2.COLOR_RGB2BGR)
            if out_bgr.shape[:2] != frame.shape[:2]:
                out_bgr = cv2.resize(out_bgr, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_AREA)
            restored.append(out_bgr)

    del pipe
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return restored


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Rendered input video")
    ap.add_argument("--out", required=True, help="Restored output mp4")
    ap.add_argument("--method", default="diffusers_img2img", choices=["diffusers_img2img", "unsharp_preview"])
    ap.add_argument("--model_path", default="", help="Local Diffusers img2img model folder")
    ap.add_argument("--max_frames", type=int, default=12)
    ap.add_argument("--strength", type=float, default=0.15)
    ap.add_argument("--guidance_scale", type=float, default=2.0)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--prompt", default="realistic detailed human avatar, natural clothing details, sharp texture, temporally stable")
    ap.add_argument("--negative_prompt", default="blur, artifacts, distorted face, distorted hands, flicker, inconsistent identity")
    ap.add_argument("--summary", default="")
    args = ap.parse_args()

    frames, fps = read_bgr_video(args.input)
    max_frames = max(1, min(int(args.max_frames), len(frames)))

    if args.method == "unsharp_preview":
        restored = restore_with_unsharp(frames, max_frames)
        active_type = "fallback_non_diffusion_preview"
    else:
        restored = restore_with_diffusers_img2img(
            frames,
            args.model_path,
            args.prompt,
            args.negative_prompt,
            args.strength,
            args.guidance_scale,
            args.steps,
            max_frames,
        )
        active_type = "level15_pretrained_diffusion_prototype"

    write_bgr_video(restored, args.out, fps)
    summary = {
        "input": args.input,
        "out": args.out,
        "method": args.method,
        "active_type": active_type,
        "model_path": args.model_path,
        "total_frames": len(frames),
        "processed_frames": max_frames,
        "strength": args.strength,
        "guidance_scale": args.guidance_scale,
        "steps": args.steps,
    }
    summary_path = args.summary or str(Path(args.out).with_suffix(".json"))
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
