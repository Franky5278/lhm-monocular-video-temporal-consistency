"""Exercise CUDA extension kernels before loading the full reconstruction model."""
import json
from pathlib import Path
import sys
import time
import traceback

import torch

records = []


def check(name, fn):
    start = time.time()
    try:
        detail = fn()
        torch.cuda.synchronize()
        records.append({'name': name, 'status': 'pass', 'seconds': time.time()-start,
                        'detail': detail})
    except Exception:
        records.append({'name': name, 'status': 'failed', 'seconds': time.time()-start,
                        'traceback': traceback.format_exc()})


def knn():
    from simple_knn._C import distCUDA2
    torch.manual_seed(42)
    points = torch.rand(64, 3, device='cuda')
    actual = distCUDA2(points)
    expected = torch.cdist(points, points).square().topk(4, largest=False).values[:, 1:].mean(1)
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-4)
    return {'points': 64, 'max_error': (actual-expected).abs().max().item()}


def pytorch3d():
    from pytorch3d.ops import knn_points
    points = torch.rand(1, 32, 3, device='cuda')
    result = knn_points(points, points, K=2)
    assert result.dists.is_cuda and torch.isfinite(result.dists).all()
    torch.testing.assert_close(result.dists[..., 0], torch.zeros_like(result.dists[..., 0]))
    return {'points': 32}


def mmcv():
    from mmcv.ops import nms
    boxes = torch.tensor([[0., 0., 10., 10.], [1., 1., 9., 9.], [20., 20., 30., 30.]], device='cuda')
    scores = torch.tensor([.9, .8, .7], device='cuda')
    _, indices = nms(boxes, scores, .5)
    assert indices.tolist() == [0, 2]
    return {'retained': indices.tolist()}


if __name__ == '__main__':
    if not torch.cuda.is_available():
        raise SystemExit('CUDA unavailable; GPU smoke cannot pass')
    check('simple_knn_CUDA_vs_pairwise', knn)
    check('pytorch3d_CUDA_knn', pytorch3d)
    check('mmcv_CUDA_nms', mmcv)
    result = {'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__,
              'records': records, 'passed': all(r['status'] == 'pass' for r in records),
              'full_LHM_inference_verified': False}
    Path(sys.argv[1]).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['passed'] else 1)
