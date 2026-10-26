"""CPU import/asset checks. This does not certify successful GPU inference."""
from pathlib import Path
import importlib
import inspect
import json
import os
import sys
import traceback
import numpy as np

root = Path('/workspace/lhm')
source = root/'LHM'
sys.path.insert(0, str(source))
os.chdir(source)
if not hasattr(inspect, 'getargspec'):
    inspect.getargspec = inspect.getfullargspec
for name, value in {'bool': bool, 'int': int, 'float': float, 'complex': complex,
                    'object': object, 'str': str, 'unicode': str}.items():
    if name not in np.__dict__:
        setattr(np, name, value)
os.environ['XFORMERS_DISABLED'] = '1'
records = []
for module in ['torch', 'torchvision', 'pytorch3d', 'diff_gaussian_rasterization',
               'simple_knn._C', 'mmcv', 'mmpose', 'ultralytics', 'diffusers',
               'lpips', 'skimage', 'app_motion_ms']:
    try:
        imported = importlib.import_module(module)
        records.append({'module': module, 'status': 'pass',
                        'version': getattr(imported, '__version__', None)})
    except Exception as e:
        records.append({'module': module, 'status': 'failed', 'error': str(e),
                        'traceback': traceback.format_exc()})
weights = source/'pretrained_models/human_model_files/pose_estimate'
for file in ['multiHMR_896_L.pt', 'vitpose-h-wholebody.pth', 'yolov8x.pt']:
    p = weights/file
    records.append({'asset': str(p), 'status': 'pass' if p.is_file() and p.stat().st_size > 1000 else 'failed'})
result = {'kind': 'CPU import and asset preflight', 'python': sys.version,
          'records': records, 'passed': all(r['status'] == 'pass' for r in records),
          'GPU_inference_verified': False}
(root/'results/cpu_preflight.json').write_text(json.dumps(result, indent=2), encoding='utf8')
for r in records:
    print(r.get('module', r.get('asset')), r['status'], r.get('error', ''))
if not result['passed']:
    raise SystemExit(1)
