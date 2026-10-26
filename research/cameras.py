"""ZJU calibrated relative cameras. Extrinsic translations are millimeters."""
import numpy as np


def relative_camera(cameras, source=0, target=1):
    ra, rb = np.asarray(cameras['R'][source]), np.asarray(cameras['R'][target])
    ta = np.asarray(cameras['T'][source]).reshape(3) / 1000
    tb = np.asarray(cameras['T'][target]).reshape(3) / 1000
    w2c = np.eye(4)
    w2c[:3, :3] = rb @ ra.T
    w2c[:3, 3] = tb - w2c[:3, :3] @ ta
    return np.linalg.inv(w2c), np.asarray(cameras['K'][target], dtype=np.float64)


def scale_intrinsics(k, original_wh, new_wh):
    k = np.array(k, dtype=np.float64, copy=True)
    k[0] *= new_wh[0] / original_wh[0]
    k[1] *= new_wh[1] / original_wh[1]
    return k


def projection_consistency(cameras, source=0, target=1, seed=42):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(100, 3)) * 0.2
    ra, rb = np.asarray(cameras['R'][source]), np.asarray(cameras['R'][target])
    ta = np.asarray(cameras['T'][source]).reshape(3) / 1000
    tb = np.asarray(cameras['T'][target]).reshape(3) / 1000
    xa, xb = x @ ra.T + ta, x @ rb.T + tb
    c2w, k = relative_camera(cameras, source, target)
    w2c = np.linalg.inv(c2w)
    via_source = xa @ w2c[:3, :3].T + w2c[:3, 3]
    p, q = xb @ k.T, via_source @ k.T
    p, q = p[:, :2] / p[:, 2:], q[:, :2] / q[:, 2:]
    error = float(np.abs(p-q).max())
    if error > 1e-3:
        raise ValueError('Relative-camera projection check failed: ' + str(error))
    return {'points': len(x), 'max_pixel_error': error,
            'translation_conversion': 'millimeters to meters',
            'scope': 'camera algebra validation; not avatar pose or quality validation'}
