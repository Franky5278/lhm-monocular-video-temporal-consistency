from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from LHM.outputs.output import GaussianAppOutput


def refine_canonical_gs_with_diffusion(
    gs_model_list: Sequence[GaussianAppOutput],
    enable: bool = False,
    checkpoint_path: Optional[str] = None,
    num_steps: int = 10,
    strength: float = 0.3,
) -> Tuple[List[GaussianAppOutput], Dict[str, object]]:
    """Week-1 diffusion hook.

    This deliberately does not pretend to be trained diffusion. It preserves the
    exact location in the pipeline: after canonical GS fusion and before LBS
    animation. Week-3 will replace this with the trained Gaussian denoiser.
    """
    report = {
        "enabled": bool(enable),
        "checkpoint_path": checkpoint_path,
        "num_steps": int(num_steps),
        "strength": float(strength),
        "active": False,
        "note": "Week-1 hook only. No trained diffusion checkpoint loaded yet.",
    }
    if not enable:
        report["note"] = "Diffusion refiner disabled by checkbox."
        return list(gs_model_list), report
    if not checkpoint_path:
        report["note"] = "Diffusion checkbox enabled, but checkpoint_path is empty. Identity returned."
        return list(gs_model_list), report
    # Real model loading will be implemented after dataset cache + training.
    report["note"] = "Checkpoint path provided, but Week-1 code does not load diffusion yet. Identity returned."
    return list(gs_model_list), report
