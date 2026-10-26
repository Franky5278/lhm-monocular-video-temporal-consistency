"""Frame-aligned evaluation with explicit metric units and full per-frame data."""
import argparse
import csv
import json
from pathlib import Path
import cv2
import numpy as np
from skimage.metrics import structural_similarity


def psnr(a, b, mask=None):
    error = (a.astype(np.float32) / 255 - b.astype(np.float32) / 255) ** 2
    if mask is not None:
        error = error[mask]
    if not error.size:
        return None
    mse = float(error.mean())
    return 99.0 if mse < 1e-12 else float(-10 * np.log10(mse))


def bbox(mask):
    ys, xs = np.where(mask)
    if not len(xs):
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def crop(image, box):
    if box is None:
        return image
    x1, y1, x2, y2 = box
    return image[y1:y2, x1:x2]


def bootstrap_blocks(values, seed=42, block=5, samples=1000):
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)])
    if not len(v):
        return None
    chunks = [v[i:i+block] for i in range(0, len(v), block)]
    rng = np.random.default_rng(seed)
    means = [np.concatenate([chunks[i] for i in rng.integers(0, len(chunks), len(chunks))]).mean()
             for _ in range(samples)]
    return {'mean': float(v.mean()), 'median': float(np.median(v)), 'std': float(v.std()),
            'n_frames': len(v), 'block_bootstrap_95_interval': np.percentile(means, [2.5, 97.5]).tolist()}


def evaluate(pred_path, gt_path, mask_dir, out_dir, size=512, lpips_enabled=True):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pred, gt = cv2.VideoCapture(str(pred_path)), cv2.VideoCapture(str(gt_path))
    pn, gn = int(pred.get(cv2.CAP_PROP_FRAME_COUNT)), int(gt.get(cv2.CAP_PROP_FRAME_COUNT))
    pf, gf = pred.get(cv2.CAP_PROP_FPS), gt.get(cv2.CAP_PROP_FPS)
    if pn != gn or abs(pf-gf) > 1e-3:
        raise ValueError(f'Temporal correspondence rejected: {pn}@{pf} versus {gn}@{gf}')
    if pn < 2:
        raise ValueError('At least two synchronized frames required')
    masks = sorted(Path(mask_dir).glob('*.png')) if mask_dir else []
    if masks and len(masks) != gn:
        raise ValueError('Mask count differs from GT frame count')
    perceptual = None
    if lpips_enabled:
        import torch
        import lpips
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        perceptual = lpips.LPIPS(net='alex').to(device).eval()
    rows, previous, previous_mask = [], None, None
    preview_ids = set(np.linspace(0, pn-1, min(6, pn)).round().astype(int).tolist())
    for i in range(pn):
        okp, p = pred.read()
        okg, g = gt.read()
        if not okp or not okg:
            raise ValueError(f'Unexpected decode failure at frame {i}')
        p = cv2.resize(p, (size, size), interpolation=cv2.INTER_AREA)
        g = cv2.resize(g, (size, size), interpolation=cv2.INTER_AREA)
        mask = None
        if masks:
            m = cv2.imread(str(masks[i]), cv2.IMREAD_GRAYSCALE)
            if m is None:
                raise ValueError('Unreadable GT mask')
            mask = cv2.resize(m, (size, size), interpolation=cv2.INTER_NEAREST) > 0
        r = {'frame_index': i, 'time_seconds': i / gf, 'psnr_full_db': psnr(p, g),
             'ssim_full': float(structural_similarity(g, p, channel_axis=2, data_range=255)),
             'psnr_masked_db': psnr(p, g, mask) if mask is not None else None}
        if mask is not None:
            r['foreground_pixels'] = int(mask.sum())
            r['psnr_gt_bbox_db'] = psnr(crop(p, bbox(mask)), crop(g, bbox(mask)))
            # Descriptive normalization only: it is not a camera-calibrated metric.
            pb = bbox(np.any(p < 245, axis=-1))
            pa = cv2.resize(crop(p, pb), (size, size))
            ga = cv2.resize(crop(g, bbox(mask)), (size, size))
            r['psnr_independent_bbox_aligned_db'] = psnr(pa, ga)
        if perceptual is not None:
            def tensor(x):
                rgb = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
                return torch.from_numpy(rgb.copy()).permute(2, 0, 1)[None].float().to(device) / 127.5 - 1
            with torch.no_grad():
                r['lpips_full'] = float(perceptual(tensor(p), tensor(g)).item())
        pp, gg = p.astype(np.float32) / 255, g.astype(np.float32) / 255
        if previous is not None:
            d = np.abs((pp-previous[0]) - (gg-previous[1]))
            r['temporal_signed_delta_l1_0_1'] = float(d.mean())
            if mask is not None:
                active = mask | previous_mask
                r['temporal_signed_delta_masked_l1_0_1'] = float(d[active].mean()) if active.any() else None
        else:
            r['temporal_signed_delta_l1_0_1'] = None
            r['temporal_signed_delta_masked_l1_0_1'] = None
        previous, previous_mask = (pp, gg), mask
        if i in preview_ids:
            diff = np.clip(np.abs(pp-gg).mean(-1) * 255, 0, 255).astype(np.uint8)
            heat = cv2.applyColorMap(diff, cv2.COLORMAP_TURBO)
            strip = np.hstack((g, p, heat))
            cv2.imwrite(str(out / f'frame_{i:04d}_gt_prediction_error.jpg'), strip)
        rows.append(r)
    pred.release()
    gt.release()
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with (out/'per_frame.csv').open('w', newline='', encoding='utf8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary = {k: bootstrap_blocks([r.get(k) for r in rows]) for k in fields
               if k not in ['frame_index', 'time_seconds', 'foreground_pixels']}
    result = {'prediction': str(pred_path), 'gt': str(gt_path), 'frames': pn, 'fps': gf,
              'evaluation_size': [size, size], 'lpips_frames': pn if perceptual is not None else 0,
              'temporal_alignment': 'equal frame count and fps; corrected single-pass motion extraction',
              'camera_alignment': 'must be established by rendering metadata, not inferred by evaluator',
              'ci_caution': 'Descriptive frame-block intervals; not independent-subject generalization',
              'metrics': summary}
    (out/'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf8')
    return result


if __name__ == '__main__':
    a = argparse.ArgumentParser()
    a.add_argument('--pred', required=True)
    a.add_argument('--gt', required=True)
    a.add_argument('--mask-dir')
    a.add_argument('--out', required=True)
    a.add_argument('--size', type=int, default=512)
    a.add_argument('--no-lpips', action='store_true')
    args = a.parse_args()
    evaluate(args.pred, args.gt, args.mask_dir, args.out, args.size, not args.no_lpips)
