"""Portable canonical-avatar snapshots and conservative, auditable fusion.

These are inference-time heuristics, not a trained multi-image model. Correspondence
is accepted only when the canonical anchor coordinates agree across all inputs.
"""
from pathlib import Path
import copy
import torch

FIELDS = ('offset_xyz', 'opacity', 'rotation', 'scaling', 'shs')


def cpu_tree(x):
    if torch.is_tensor(x):
        return x.detach().cpu()
    if isinstance(x, dict):
        return {k: cpu_tree(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [cpu_tree(v) for v in x]
    return x


def save_bundle(path, gs, query_points, transform, betas, motion, metadata):
    bundle = {
        'schema': 'lhm.canonical.v1',
        'gs': [{**{k: cpu_tree(getattr(g, k)) for k in FIELDS},
                'use_rgb': bool(g.use_rgb)} for g in gs],
        'query_points': cpu_tree(query_points),
        'transform_mat_neutral_pose': cpu_tree(transform),
        'betas': cpu_tree(betas),
        'motion': cpu_tree({k: motion[k] for k in
                            ('smplx_params', 'render_c2ws', 'render_intrs', 'render_bg_colors')}),
        'metadata': metadata,
    }
    validate(bundle)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, path)
    return bundle


def validate(bundle):
    if bundle.get('schema') != 'lhm.canonical.v1':
        raise ValueError('Unsupported asset schema')
    q = bundle['query_points']
    if not torch.is_tensor(q) or q.ndim != 3 or q.shape[-1] != 3:
        raise ValueError('Expected query points [batch, anchors, 3]')
    if len(bundle['gs']) != q.shape[0]:
        raise ValueError('Gaussian/anchor batch mismatch')
    for g in bundle['gs']:
        for k in FIELDS:
            if g[k].shape[0] != q.shape[1] or not torch.isfinite(g[k]).all():
                raise ValueError('Invalid attribute: ' + k)
        if (g['scaling'] <= 0).any():
            raise ValueError('Nonpositive Gaussian scale')
    return {'batch': q.shape[0], 'anchors': q.shape[1]}


def load_bundle(path):
    # Accept only locally produced project assets. Never load arbitrary pickle files.
    x = torch.load(path, map_location='cpu', weights_only=False)
    validate(x)
    return x


def restore_gs(bundle, device='cuda'):
    from LHM.outputs.output import GaussianAppOutput
    return [GaussianAppOutput(**{k: v.to(device) if torch.is_tensor(v) else v
                                 for k, v in g.items()}) for g in bundle['gs']]


def fuse_bundles(bundles, frame_weights=None, agreement_scale=0.02):
    if len(bundles) < 2:
        raise ValueError('Fusion needs at least two independently reconstructed sources')
    for b in bundles:
        validate(b)
    ref = bundles[0]
    for b in bundles[1:]:
        if b['query_points'].shape != ref['query_points'].shape:
            raise ValueError('Anchor topology mismatch')
        if not torch.allclose(b['query_points'], ref['query_points'], atol=1e-6, rtol=1e-6):
            raise ValueError('Anchor correspondence unverified; use shared shape and anchor order')
        if not torch.allclose(b['betas'], ref['betas'], atol=1e-6, rtol=1e-6):
            raise ValueError('Cannot fuse different body shapes without registration')
    out = copy.deepcopy(ref)
    fw = torch.as_tensor(frame_weights if frame_weights is not None else
                         [1.] * len(bundles), dtype=torch.float32)
    if fw.shape != (len(bundles),) or not torch.isfinite(fw).all() or (fw <= 0).any():
        raise ValueError('Invalid positive frame weights')
    confidence = []
    for batch in range(len(ref['gs'])):
        xyz = torch.stack([b['gs'][batch]['offset_xyz'] for b in bundles]).float()
        center = xyz.median(dim=0).values
        disagreement = (xyz - center).norm(dim=-1)
        # An agreement proxy, explicitly not a visibility/occlusion estimator.
        weights = torch.softmax(-disagreement / agreement_scale + fw.log()[:, None], dim=0)
        g = out['gs'][batch]
        for field in FIELDS:
            values = torch.stack([b['gs'][batch][field] for b in bundles]).float()
            w = weights.reshape(weights.shape + (1,) * (values.ndim - 2))
            if field == 'rotation':
                values = values / values.norm(dim=-1, keepdim=True).clamp_min(1e-8)
                sign = (values * values[0:1]).sum(dim=-1, keepdim=True).sign()
                values = values * torch.where(sign == 0, torch.ones_like(sign), sign)
            if field == 'scaling':
                values = values.clamp_min(1e-8).log()
            result = (values * w).sum(dim=0)
            if field == 'rotation':
                result = result / result.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            if field == 'scaling':
                result = result.exp()
            g[field] = result.to(ref['gs'][batch][field].dtype)
        confidence.append(torch.exp(-disagreement.mean(dim=0) / agreement_scale))
    out['fusion_confidence'] = confidence
    out['metadata'] = {**ref['metadata'], 'fusion': {
        'sources': len(bundles), 'frame_weights': fw.tolist(),
        'agreement_scale_m': agreement_scale,
        'correspondence': 'canonical query coordinates and shared shape verified',
        'confidence': 'cross-source offset agreement proxy; not visibility',
        'method': 'weighted linear attributes, hemisphere-corrected quaternion, log scale',
    }}
    validate(out)
    return out


def refine_graph(bundle, amount=0.15, neighbors=8):
    """Small color-only low-confidence graph residual; no trained diffusion claim."""
    import numpy as np
    from scipy.spatial import cKDTree
    if not 0 <= amount <= 1:
        raise ValueError('amount must be in [0,1]')
    out = copy.deepcopy(bundle)
    for batch, g in enumerate(out['gs']):
        q = out['query_points'][batch].float().numpy()
        _, indices = cKDTree(q).query(q, k=min(neighbors + 1, len(q)))
        indices = np.asarray(indices)[:, 1:]
        confidence = out.get('fusion_confidence', [torch.ones(len(q))] * len(out['gs']))[batch]
        colors = g['shs'].float()
        local = colors[torch.as_tensor(indices)].mean(dim=1)
        weight = (amount * (1 - confidence)).reshape((-1,) + (1,) * (colors.ndim - 1))
        g['shs'] = (colors + weight * (local - colors)).to(g['shs'].dtype)
    out['metadata'] = {**out['metadata'], 'refinement': {
        'method': 'canonical kNN color residual', 'amount': amount,
        'neighbors': neighbors, 'trained': False, 'geometry_changed': False,
        'limitation': 'Euclidean neighbors can cross garment/body boundaries',
    }}
    validate(out)
    return out
