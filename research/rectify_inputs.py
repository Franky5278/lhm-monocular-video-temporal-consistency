"""Undistort synchronized input/GT without introducing held-out image feedback."""
import json
from pathlib import Path
import cv2
import numpy as np


def video_maps(path, k, d):
    cap = cv2.VideoCapture(str(path))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    # ZJU images in the retained sequences are at their annotated 1024x1024 size.
    if (w, h) != (1024, 1024):
        raise ValueError('Annotation/image scaling must be specified for non-1024 ZJU input')
    maps = cv2.initUndistortRectifyMap(k, d, None, k, (w, h), cv2.CV_32FC1)
    return maps, (w, h), fps


def rectify(source, camera_file, output):
    source, output = Path(source), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    record = json.loads((source/'manifest.json').read_text())
    data = json.loads(Path(camera_file).read_text())
    cameras = data['cameras']
    source_cam, target_cam = int(record['source_cam']), int(record['heldout_cam'])
    count = {}
    for side, ci in [('source_view_A', source_cam), ('heldout_view_B', target_cam)]:
        k = np.asarray(cameras['K'][ci], dtype=np.float64)
        d = np.asarray(cameras['D'][ci], dtype=np.float64)
        maps, wh, fps = video_maps(source/(side+'.mp4'), k, d)
        cap = cv2.VideoCapture(str(source/(side+'.mp4')))
        name = 'source_calibrated_A.mp4' if side == 'source_view_A' else side+'.mp4'
        writer = cv2.VideoWriter(str(output/name), cv2.VideoWriter_fourcc(*'mp4v'), fps, wh)
        if not writer.isOpened():
            raise RuntimeError('Cannot write rectified video')
        n = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            writer.write(cv2.remap(frame, *maps, interpolation=cv2.INTER_LINEAR))
            n += 1
        cap.release()
        writer.release()
        count[side] = n
        if side == 'source_view_A':
            # Keep the headless core's conventional filename, but unique content/hash.
            import shutil
            shutil.copy2(output/name, output/'source_view_A.mp4')
            target = output/'source_view_A_frames'
            target.mkdir(exist_ok=True)
            for f in sorted((source/'source_view_A_frames').glob('*.jpg')):
                image = cv2.imread(str(f))
                cv2.imwrite(str(target/f.name), cv2.remap(image, *maps, interpolation=cv2.INTER_LINEAR))
            (output/'source_camera.json').write_text(json.dumps({'K': k.tolist(),
                'image_size_wh': list(wh), 'distortion': 'already removed',
                'source_cam': source_cam}, indent=2), encoding='utf8')
        else:
            target = output/'heldout_view_B_masks'
            target.mkdir(exist_ok=True)
            for f in sorted((source/'heldout_view_B_masks').glob('*.png')):
                image = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
                cv2.imwrite(str(target/f.name), cv2.remap(image, *maps, interpolation=cv2.INTER_NEAREST))
    if count['source_view_A'] != count['heldout_view_B']:
        raise ValueError('Rectified frame counts differ')
    record['rectification'] = {'method': 'OpenCV calibrated undistort, unchanged K/output size',
                              'frames': count, 'camera_file': str(camera_file)}
    (output/'manifest.json').write_text(json.dumps(record, indent=2), encoding='utf8')
    return {'source_cam': source_cam, 'target_cam': target_cam, 'cameras': cameras,
            'frames': count['source_view_A'], 'output': str(output)}
