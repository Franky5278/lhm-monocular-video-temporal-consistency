"""Run the recovered LHM prototype without a browser or Gradio server.

Every stage records success/failure independently. An asset or video is never
treated as a successful experiment until its corresponding stage record passes.
"""
import argparse
import contextlib
import gc
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time
import traceback
from types import SimpleNamespace


def dump(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str), encoding='utf8')


class Run:
    def __init__(self, root, deadline):
        self.root = Path(root)
        self.deadline = deadline
        self.root.mkdir(parents=True, exist_ok=True)

    def allowed(self, reserve=600):
        return time.time() < self.deadline-reserve

    def stage(self, name, fn, config):
        stage = self.root/name
        stage.mkdir(parents=True, exist_ok=True)
        record = stage/'stage.json'
        if record.exists() and json.loads(record.read_text()).get('status') == 'success':
            print('REUSE', name, flush=True)
            return json.loads(record.read_text()).get('output')
        if not self.allowed():
            dump(record, {'status': 'not_run', 'reason': 'budget deadline reserve', 'config': config})
            return None
        import torch
        torch.cuda.reset_peak_memory_stats()
        start = time.time()
        dump(record, {'status': 'running', 'started_unix': start, 'config': config})
        print('START', name, flush=True)
        with (stage/'stdout.log').open('a', buffering=1) as log:
            try:
                with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                    result = fn(stage)
                    torch.cuda.synchronize()
                data = {'status': 'success', 'started_unix': start,
                        'seconds': time.time()-start, 'config': config, 'output': result,
                        'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                        'peak_reserved_bytes': torch.cuda.max_memory_reserved()}
            except Exception as error:
                log.write(traceback.format_exc())
                data = {'status': 'failed', 'started_unix': start, 'seconds': time.time()-start,
                        'config': config, 'error_type': type(error).__name__, 'error': str(error)}
                result = None
        dump(record, data)
        print('END', name, data['status'], round(data['seconds'], 2), flush=True)
        gc.collect()
        torch.cuda.empty_cache()
        return result


def core_run(app, core, sequence, source_image, stage, shared_betas, source_manager=False):
    import cv2
    os.environ['LHM_ASSET_DIR'] = str(stage)
    os.environ['LHM_SHARED_BETAS'] = str(shared_betas)
    os.environ['LHM_URECA_RESULTS_ROOT'] = str(stage/'legacy_outputs')
    source_camera = sequence/'source_camera.json'
    if source_camera.exists():
        os.environ['LHM_SOURCE_CAMERA_JSON'] = str(source_camera)
    else:
        os.environ.pop('LHM_SOURCE_CAMERA_JSON', None)
    work = stage/'work'
    work.mkdir(exist_ok=True)
    result = core(
        image_upload=None, input_image_path=str(source_image), video_upload=None,
        target_motion_video_path=str(sequence/'source_view_A.mp4'),
        use_prefilter=True, use_pose_joint=True, use_trans_vel=True,
        pose_alpha=0.25, trans_alpha=0.25,
        enable_multi_keyframe_fusion=source_manager, num_source_keyframes=5,
        source_frames_upload=None, project_source_path=str(sequence/'source_view_A_frames'),
        enable_heldout_eval=False, heldout_gt_video_upload=None, heldout_gt_video_path='',
        enable_lpips_eval=False, enable_diffusion_restoration=False,
        diffusion_method='diffusers_img2img', diffusion_model_path='',
        diffusion_strength=0.12, diffusion_steps=12, diffusion_max_frames=60,
        working_dir=SimpleNamespace(name=str(work)),
    )
    paths = {}
    for name, item in zip(('source_preview', 'baseline', 'stabilized', 'diffusion', 'comparison'), result[:5]):
        if item and Path(item).is_file():
            suffix = Path(item).suffix
            target = stage/(name+suffix)
            if Path(item).resolve() != target.resolve():
                shutil.copy2(item, target)
            paths[name] = str(target)
    dump(stage/'motion_metrics_legacy_definition.json', {'rows': result[5],
          'columns': ['variant', 'jitter_pose_6d_second_difference', 'jitter_translation_second_difference',
                      'endpoint_displacement_m', 'pose_deviation_6d_norm', 'translation_deviation_m'],
          'note': 'Legacy field drift_trans is endpoint displacement, not a GT drift error.'})
    paths['asset'] = str(stage/'canonical.pt')
    if not Path(paths['asset']).is_file():
        raise RuntimeError('No exported canonical asset')
    gt = cv2.VideoCapture(str(sequence/'heldout_view_B.mp4'))
    expected = int(gt.get(cv2.CAP_PROP_FRAME_COUNT))
    gt.release()
    for name in ('baseline', 'stabilized'):
        cap = cv2.VideoCapture(paths[name])
        count = 0
        while cap.read()[0]:
            count += 1
        cap.release()
        if count != expected:
            raise ValueError(f'{name}: {count} decoded frames versus {expected} GT frames')
    return paths


def render_bundle(model, bundle, path, camera=None, motion=None, size=512):
    import cv2
    import numpy as np
    import torch
    from research_assets import restore_gs
    gs = restore_gs(bundle)
    q = bundle['query_points'].cuda()
    seq = bundle['motion'] if motion is None else motion
    params = seq['smplx_params']
    total = params['root_pose'].shape[1]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(path), fourcc, 30., (size, size))
    if not writer.isOpened():
        raise RuntimeError('Video writer failed')
    coverage = []
    with torch.no_grad():
        for i in range(total):
            p = {k: (v if k in ['betas', 'transform_mat_neutral_pose'] else v[:, i:i+1]).cuda()
                 for k, v in params.items() if torch.is_tensor(v)}
            p['betas'] = bundle['betas'].cuda()
            p['transform_mat_neutral_pose'] = bundle['transform_mat_neutral_pose'].cuda()
            if camera is None:
                c2w = seq['render_c2ws'][:, i:i+1].clone().cuda()
                intr = seq['render_intrs'][:, i:i+1].clone().cuda()
                # Preserve the original view; resize completed render explicitly.
            else:
                c2w, k = camera
                c2w = torch.as_tensor(c2w, dtype=torch.float32, device='cuda')[None, None]
                intr = torch.eye(4, device='cuda')[None, None]
                intr[0, 0, :3, :3] = torch.as_tensor(k, dtype=torch.float32, device='cuda')
            result = model.renderer.forward_animate_gs(
                gs, q, p, c2w, intr, size, size,
                torch.ones((1, 1, 3), device='cuda'))
            rgb, mask = result['comp_rgb'][0, 0], result['comp_mask'][0, 0]
            coverage.append(float((mask > .05).float().mean().item()))
            rgb = rgb * mask + (1-mask)
            image = rgb.permute(1, 2, 0).detach().float().cpu().numpy()
            image = np.clip(image * 255, 0, 255).astype(np.uint8)
            image = cv2.cvtColor(cv2.resize(image, (size, size)), cv2.COLOR_RGB2BGR)
            writer.write(image)
    writer.release()
    if max(coverage, default=0) < .001:
        raise ValueError('Rendered foreground coverage is effectively empty')
    return {'video': str(path), 'frames': total, 'fps': 30, 'size': [size, size],
            'foreground_coverage_mean': float(np.mean(coverage)),
            'foreground_coverage_min': float(np.min(coverage))}


def fusion_stage(model, inputs, stage, confidence, graph=0.):
    import torch
    from research_assets import load_bundle, fuse_bundles, refine_graph
    bundles = [load_bundle(p) for p in inputs]
    fused = fuse_bundles(bundles, agreement_scale=0.02 if confidence else 1e9)
    if graph:
        fused = refine_graph(fused, amount=graph)
    asset = stage/'fused_canonical.pt'
    torch.save(fused, asset)
    reloaded = load_bundle(asset)
    # Verify serialization preserves every Gaussian attribute exactly.
    for a, b in zip(fused['gs'], reloaded['gs']):
        for k, v in a.items():
            if torch.is_tensor(v) and not torch.equal(v, b[k]):
                raise ValueError('Serialization changed ' + k)
    result = render_bundle(model, reloaded, stage/'render.mp4')
    result.update(asset=str(asset), reload_attributes_exact=True,
                  fusion=fused['metadata'].get('fusion'), refinement=fused['metadata'].get('refinement'))
    return result


def smoothing_stage(app, model, base, stage, alpha, branch):
    from research_assets import load_bundle
    from engine.pose_estimation.pose_utils.postprocess import (
        make_variant_from_raw, load_smplx_dir, compute_metrics_vs)
    directories = json.loads((Path(base['asset']).parent/'motion_directories.json').read_text())
    raw_dir = directories['raw']
    config = {'use_prefilter': branch != 'raw', 'use_pose_joint': branch in ['pose', 'full'],
              'use_trans_vel': branch in ['translation', 'full'],
              'joint_alpha': alpha, 'vel_alpha_trans': alpha}
    motion_dir = stage/'smplx_params'
    make_variant_from_raw(raw_dir, str(motion_dir), fps=30, compose_cfg=config)
    _, _, raw_pose, _, raw_trans = load_smplx_dir(raw_dir)
    _, _, pose, _, trans = load_smplx_dir(str(motion_dir))
    metrics = compute_metrics_vs(raw_pose, raw_trans, pose, trans)
    metrics['endpoint_displacement_m'] = metrics.pop('drift_trans')
    dump(stage/'motion_metrics.json', {'metrics': metrics,
         'rotation_space': '6D rotation representation, flattened norm',
         'second_difference_time_scaling': 'per-frame; no fps^2 normalization',
         'jitter_note': 'Second finite difference, an acceleration proxy rather than third-derivative jerk'})
    motion = app.prepare_motion_seqs(str(motion_dir), None, str(stage), fps=30,
        bg_color=1., aspect_standard=5/3, enlarge_ratio=[1., 1, 0],
        render_image_res=512, need_mask=False, multiply=16, vis_motion=False, motion_size=3000)
    bundle = load_bundle(base['asset'])
    result = render_bundle(model, bundle, stage/'render.mp4', motion=motion)
    result['motion_metrics'] = metrics
    return result


def main(args):
    import numpy as np
    import torch
    # Compatibility for licensed legacy SMPL/Chumpy pickles on modern Python/NumPy.
    import inspect
    if not hasattr(inspect, 'getargspec'):
        inspect.getargspec = inspect.getfullargspec
    for name, value in {'bool': bool, 'int': int, 'float': float, 'complex': complex,
                        'object': object, 'str': str, 'unicode': str}.items():
        if name not in np.__dict__:
            setattr(np, name, value)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    if not torch.cuda.is_available():
        raise RuntimeError('GPU run requested without CUDA')
    os.environ['LHM_HEADLESS'] = '1'
    os.environ['LHM_RENDER_RESOLUTION'] = '512'
    source = Path(args.source).resolve()
    sys.path.insert(0, str(source))
    os.chdir(source)
    import app_motion_ms as app
    from omegaconf import OmegaConf
    checkpoint = Path(args.checkpoint).resolve()
    cfg = OmegaConf.create({'model_name': str(checkpoint), 'source_size': 1024,
                           'src_head_size': 112, 'render_size': 512,
                           'motion_video_read_fps': 30})
    app.download_geo_files()
    model = app._build_model(cfg).cuda().eval()
    core = app.demo_lhm(model, None, cfg)
    run = Run(args.out, args.deadline)
    dump(run.root/'run_config.json', {**vars(args), 'seed': 42,
        'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__,
        'checkpoint': str(checkpoint), 'model_training': False,
        'evaluation_views_used_for_selection': False,
        'pending': ['trained 3D diffusion', 'optimization-method baselines']})
    observations = []
    calibrated_observations = []
    for seq_name in ['zju_313_wk2', 'zju_390_kick_wk2', 'zju_390_mid10']:
        seq = Path(args.inputs)/seq_name
        frames = sorted((seq/'source_view_A_frames').glob('*.jpg'))
        if not frames:
            continue
        shared = run.root/seq_name/'shared_betas.json'
        ref = frames[len(frames)//2]
        config = {'sequence': seq_name, 'reference': ref.name, 'alpha_pose': .25, 'alpha_translation': .25}
        base = run.stage(seq_name+'/core', lambda stage: core_run(app, core, seq, ref, stage, shared), config)
        if not base:
            continue
        observations.append({'sequence': seq_name, 'variant': 'baseline', 'video': base['baseline']})
        observations.append({'sequence': seq_name, 'variant': 'stabilized', 'video': base['stabilized']})
        assets = [base['asset']]
        if seq_name == 'zju_313_wk2':
            for branch, alpha in [('raw', .25), ('gaussian', .25), ('pose', .25),
                                  ('translation', .25), ('full', .10), ('full', .50), ('full', .80)]:
                variant = f'motion_{branch}_a{alpha}'
                result = run.stage(seq_name+'/'+variant,
                    lambda stage, a=alpha, b=branch: smoothing_stage(app, model, base, stage, a, b),
                    {'branch': branch, 'alpha': alpha, 'source': 'same canonical asset and raw fitted motion'})
                if result:
                    observations.append({'sequence': seq_name, 'variant': variant, 'video': result['video']})
            for i, frame in enumerate(frames):
                if frame == ref:
                    continue
                result = run.stage(seq_name+f'/keyframe_{i:02}',
                    lambda stage, f=frame: core_run(app, core, seq, f, stage, shared),
                    {'sequence': seq_name, 'reference': frame.name, 'shared_shape': str(shared)})
                if result:
                    assets.append(result['asset'])
                    observations.append({'sequence': seq_name, 'variant': f'keyframe_{i:02}', 'video': result['stabilized']})
            for k in [2, 3, 5]:
                if len(assets) < k:
                    continue
                for weighted in [False, True]:
                    variant = f'fusion_k{k}_' + ('agreement' if weighted else 'uniform')
                    result = run.stage(seq_name+'/'+variant,
                        lambda stage, n=k, w=weighted: fusion_stage(model, assets[:n], stage, w),
                        {'sources': k, 'agreement_confidence': weighted, 'anchor_validation': True})
                    if result:
                        observations.append({'sequence': seq_name, 'variant': variant, 'video': result['video']})
            if len(assets) >= 3:
                for amount in [.1, .25]:
                    variant = f'fusion_graph_{amount}'
                    result = run.stage(seq_name+'/'+variant,
                        lambda stage, a=amount: fusion_stage(model, assets[:3], stage, True, a),
                        {'sources': 3, 'graph_color_amount': amount, 'trained': False})
                    if result:
                        observations.append({'sequence': seq_name, 'variant': variant, 'video': result['video']})
    # A separate calibrated attempt: rectify source/GT, fit with source K, render with B K/R/T.
    if run.allowed(reserve=1800):
        from rectify_inputs import rectify
        from cameras import relative_camera, scale_intrinsics, projection_consistency
        original = Path(args.inputs)/'zju_313_wk2'
        rectified = Path(args.inputs)/'zju_313_calibrated'
        prepared = run.stage('zju_313_calibrated/prepare',
            lambda stage: rectify(original, Path(args.inputs)/'cameras_313.json', rectified),
            {'distortion_removed': True, 'heldout_used_only_for_evaluation': True})
        if prepared:
            sources = sorted((rectified/'source_view_A_frames').glob('*.jpg'))
            result = run.stage('zju_313_calibrated/core',
                lambda stage: core_run(app, core, rectified, sources[len(sources)//2], stage,
                                      run.root/'zju_313_calibrated/shared_betas.json'),
                {'source_intrinsics': 'ZJU annotation with padding compensation', 'render': 'default source camera'})
            if result:
                from research_assets import load_bundle
                asset = load_bundle(result['asset'])
                c2w, k = relative_camera(prepared['cameras'], prepared['source_cam'], prepared['target_cam'])
                k = scale_intrinsics(k, (1024, 1024), (512, 512))
                cam_check = projection_consistency(prepared['cameras'], prepared['source_cam'], prepared['target_cam'])
                target = run.stage('zju_313_calibrated/camera_B',
                    lambda stage: render_bundle(model, asset, stage/'camera_B.mp4', camera=(c2w, k)),
                    {'camera_check': cam_check, 'source_pose': 'estimated SMPL-X, not GT',
                     'extrinsics': 'relative B from source-A coordinates, translations in meters',
                     'distortion': 'removed from both views', 'intrinsics': k.tolist(), 'c2w': c2w.tolist()})
                if target:
                    calibrated_observations.append({'sequence': 'zju_313_calibrated',
                        'variant': 'camera_B_calibrated_gaussian_motion', 'video': target['video']})
    dump(run.root/'observations.json', observations+calibrated_observations)
    del model, core
    gc.collect()
    torch.cuda.empty_cache()
    # Run image restoration after releasing the LHM backbone.
    for seq_name in ['zju_313_wk2', 'zju_390_kick_wk2']:
        inputs = [o for o in observations if o['sequence'] == seq_name and o['variant'] == 'stabilized']
        if not inputs or not args.sd_model:
            continue
        for strength, steps in [(.10, 12), (.15, 12), (.10, 20), (.15, 20)]:
            name = f'diffusion_s{strength}_steps{steps}'
            def restore(stage, source_video=inputs[0]['video'], strength=strength, steps=steps):
                import cv2
                cap = cv2.VideoCapture(source_video)
                n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
                result = app._run_level15_diffusion_restoration(
                    source_video, str(stage/'restored.mp4'), enable=True,
                    method='diffusers_img2img', model_path=args.sd_model,
                    strength=strength, num_inference_steps=steps, max_frames=n, seed=42)
                if not result.get('active') or result.get('method') != 'diffusers_img2img':
                    raise RuntimeError('Diffusion inactive: '+str(result))
                return result
            result = run.stage(seq_name+'/'+name, restore,
                               {'strength': strength, 'steps': steps, 'effective_steps': int(steps*strength), 'paired_seed_base': 42, 'frames': 'all', 'trained': False})
            if result:
                observations.append({'sequence': seq_name, 'variant': name, 'video': result['output_path']})
    dump(run.root/'observations.json', observations+calibrated_observations)
    from evaluate_video import evaluate
    for item in observations+calibrated_observations:
        seq = Path(args.inputs)/item['sequence']
        run.stage(item['sequence']+'/eval_'+item['variant'],
            lambda stage, item=item, seq=seq: evaluate(item['video'], seq/'heldout_view_B.mp4',
                seq/'heldout_view_B_masks', stage, size=512, lpips_enabled=True),
            {'variant': item['variant'], 'size': 512, 'lpips': 'all frames',
             'camera': 'ZJU relative K/R/T after source-K fitting and undistortion' if item['sequence'].endswith('calibrated') else 'default LHM; independent-bbox alignment reported separately'})
    dump(run.root/'RUN_FINISHED.json', {'time_unix': time.time(),
         'note': 'Finished requested stages or recorded failures/skips; inspect individual stage.json files.'})


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--source', required=True)
    p.add_argument('--inputs', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--sd-model', default='')
    p.add_argument('--deadline', type=float, required=True)
    main(p.parse_args())
