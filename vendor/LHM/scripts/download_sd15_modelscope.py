#!/usr/bin/env python3
"""Download a local Stable Diffusion v1.5 Diffusers-format model from ModelScope.

Default output target:
  /autodl-fs/data/lhm_shared/shared_assets/diffusion_models/stable-diffusion-img2img-local

The script copies/symlinks the actual snapshot directory to the target path expected by the Gradio UI.
"""
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


def has_model_index(path: Path) -> bool:
    return (path / "model_index.json").exists()


def copytree_merge(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        target = dst / item.name
        if item.is_dir():
            if target.exists() and not target.is_dir():
                target.unlink()
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            shutil.copy2(item, target)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default="AI-ModelScope/stable-diffusion-v1-5")
    parser.add_argument("--cache_dir", default="/autodl-fs/data/lhm_shared/shared_assets/diffusion_models")
    parser.add_argument("--target", default="/autodl-fs/data/lhm_shared/shared_assets/diffusion_models/stable-diffusion-img2img-local")
    parser.add_argument("--copy", action="store_true", help="copy snapshot into target instead of symlink")
    args = parser.parse_args()

    from modelscope import snapshot_download

    cache_dir = Path(args.cache_dir)
    target = Path(args.target)
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"[download] model_id={args.model_id}")
    # Diffusers loads component weights. Monolithic training checkpoints and
    # alternative framework exports consume bandwidth without supporting this run.
    snapshot = Path(snapshot_download(
        args.model_id, cache_dir=str(cache_dir),
        ignore_patterns=['*.ckpt', 'v1-5-*.safetensors', '*.msgpack', '*.onnx', '*.pb'],
    )).resolve()
    print(f"[download] snapshot={snapshot}")

    if not has_model_index(snapshot):
        candidates = list(snapshot.rglob("model_index.json"))
        if not candidates:
            raise FileNotFoundError(f"model_index.json not found under snapshot: {snapshot}")
        snapshot = candidates[0].parent.resolve()
        print(f"[download] using nested diffusers dir={snapshot}")

    if target.exists() or target.is_symlink():
        if target.is_symlink() or target.is_file():
            target.unlink()
        elif args.copy:
            shutil.rmtree(target)
        else:
            shutil.rmtree(target)

    if args.copy:
        copytree_merge(snapshot, target)
        print(f"[OK] copied to {target}")
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        os.symlink(str(snapshot), str(target))
        print(f"[OK] symlinked {target} -> {snapshot}")

    print("Use this in Gradio Local pretrained diffusion model path:")
    print(target)


if __name__ == "__main__":
    main()
