#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path

REQUIRED = [
    "model_index.json",
    "unet/config.json",
    "vae/config.json",
    "tokenizer/tokenizer_config.json",
    "scheduler/scheduler_config.json",
]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("model_path", nargs="?", default="/autodl-fs/data/lhm_shared/shared_assets/diffusion_models/stable-diffusion-img2img-local")
    args = p.parse_args()
    root = Path(args.model_path)
    print(f"[check] {root}")
    if not root.exists():
        raise SystemExit(f"[FAIL] path does not exist: {root}")
    missing = []
    for rel in REQUIRED:
        ok = (root / rel).exists()
        print(("[OK]   " if ok else "[MISS] ") + rel)
        if not ok:
            missing.append(rel)
    if missing:
        raise SystemExit("[FAIL] not a complete Diffusers pipeline directory")
    print("[OK] Diffusers model directory looks usable for StableDiffusionImg2ImgPipeline.from_pretrained(..., local_files_only=True)")

if __name__ == "__main__":
    main()
