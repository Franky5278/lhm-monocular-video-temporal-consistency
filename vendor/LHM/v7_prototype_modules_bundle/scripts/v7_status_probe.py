#!/usr/bin/env python3
"""Probe whether v7 prototype modules are installed and usable."""
from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path("/autodl-fs/data/lhm_shared")
MODEL = ROOT / "shared_assets/diffusion_models/stable-diffusion-img2img-local"
INPUTS = ROOT / "poster_inputs/zju_313_wk2"
FALLBACK_INPUTS = ROOT / "ureca_inputs/zju_313_wk2"


def exists(p: Path) -> bool:
    return p.exists()


def status() -> list[dict]:
    inputs = INPUTS if INPUTS.exists() else FALLBACK_INPUTS
    return [
        {
            "module": "three_environment_architecture",
            "status": "active" if (ROOT / "poster_env_paths.json").exists() else "not active",
            "note": str(ROOT / "poster_env_paths.json"),
        },
        {
            "module": "diffusers_img2img_model",
            "status": "active" if (MODEL / "model_index.json").exists() else "missing",
            "note": str(MODEL),
        },
        {
            "module": "zju_313_inputs",
            "status": "active" if (inputs / "source_view_A.mp4").exists() and (inputs / "heldout_view_B.mp4").exists() else "missing",
            "note": str(inputs),
        },
        {
            "module": "camera_aligned_2d_eval_script",
            "status": "installed" if Path("scripts/v7_camera_aligned_eval_2d.py").exists() else "missing",
            "note": "prototype only; true K/R/T renderer integration pending",
        },
        {
            "module": "multikeyframe_render_fusion_script",
            "status": "installed" if Path("scripts/v7_multikeyframe_render_fusion.py").exists() else "missing",
            "note": "prototype only; true GS-level fusion pending",
        },
    ]


def main() -> None:
    rows = status()
    print(json.dumps(rows, indent=2))

if __name__ == "__main__":
    main()
