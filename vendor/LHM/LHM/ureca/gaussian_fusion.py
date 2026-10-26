from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch

from LHM.outputs.output import GaussianAppOutput


def _normalize_weights(n: int, weights: Optional[Sequence[float]], device=None, dtype=torch.float32) -> torch.Tensor:
    if weights is None:
        w = torch.ones(n, dtype=dtype, device=device) / max(n, 1)
    else:
        w = torch.tensor(list(weights), dtype=dtype, device=device)
        if w.numel() != n:
            raise ValueError(f"weights length {w.numel()} != number of sources {n}")
        w = w / (w.sum() + 1e-8)
    return w


def _weighted_sum(tensors: Sequence[torch.Tensor], weights: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(tensors[0])
    for i, x in enumerate(tensors):
        out = out + x * weights[i].to(device=x.device, dtype=x.dtype)
    return out


def _normalize_quat(q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return q / (q.norm(dim=-1, keepdim=True) + eps)


def _field(a: GaussianAppOutput, name: str) -> torch.Tensor:
    if not hasattr(a, name):
        raise AttributeError(f"GaussianAppOutput missing field: {name}")
    return getattr(a, name)


def fuse_gaussian_attrs(
    attrs: Sequence[GaussianAppOutput],
    weights: Optional[Sequence[float]] = None,
    mode: str = "weighted_mean",
) -> GaussianAppOutput:
    if len(attrs) == 0:
        raise ValueError("attrs is empty")
    if len(attrs) == 1:
        return attrs[0]
    if mode not in {"weighted_mean", "median"}:
        raise ValueError(f"Unsupported fusion mode: {mode}")
    ref_n = _field(attrs[0], "offset_xyz").shape[0]
    for a in attrs:
        if _field(a, "offset_xyz").shape[0] != ref_n:
            raise ValueError("All Gaussian attrs must have same number/order of anchor points")

    weights_t = _normalize_weights(len(attrs), weights)
    collect = lambda name: [_field(a, name) for a in attrs]

    if mode == "median":
        offset_xyz = torch.median(torch.stack(collect("offset_xyz"), dim=0), dim=0).values
        opacity = torch.median(torch.stack(collect("opacity"), dim=0), dim=0).values
        scaling = torch.median(torch.stack(collect("scaling"), dim=0), dim=0).values
        shs = torch.median(torch.stack(collect("shs"), dim=0), dim=0).values
        rotation = _normalize_quat(_weighted_sum(collect("rotation"), weights_t))
    else:
        offset_xyz = _weighted_sum(collect("offset_xyz"), weights_t)
        opacity = _weighted_sum(collect("opacity"), weights_t).clamp(1e-4, 1.0 - 1e-4)
        scaling = _weighted_sum(collect("scaling"), weights_t).clamp_min(1e-6)
        shs = _weighted_sum(collect("shs"), weights_t)
        rotation = _normalize_quat(_weighted_sum(collect("rotation"), weights_t))

    return GaussianAppOutput(
        offset_xyz=offset_xyz,
        opacity=opacity,
        rotation=rotation,
        scaling=scaling,
        shs=shs,
        use_rgb=attrs[0].use_rgb,
    )


def fuse_gs_model_lists(
    gs_model_lists: Sequence[Sequence[GaussianAppOutput]],
    weights: Optional[Sequence[float]] = None,
    mode: str = "weighted_mean",
) -> Tuple[List[GaussianAppOutput], Dict[str, object]]:
    if len(gs_model_lists) == 0:
        raise ValueError("gs_model_lists is empty")
    batch_size = len(gs_model_lists[0])
    for gl in gs_model_lists:
        if len(gl) != batch_size:
            raise ValueError("All gs_model_lists must have same batch size")

    fused: List[GaussianAppOutput] = []
    for b in range(batch_size):
        fused.append(fuse_gaussian_attrs([gl[b] for gl in gs_model_lists], weights=weights, mode=mode))
    report = {
        "num_sources": len(gs_model_lists),
        "batch_size": batch_size,
        "fusion_mode": mode,
        "weights": list(weights) if weights is not None else None,
        "valid_fusion": len(gs_model_lists) >= 2,
        "note": "true multi-source fusion only if num_sources >= 2",
    }
    return fused, report
