"""
Patch app_motion_ms.py for URECA Week-1 v0.2.

Run from LHM-main root after copying LHM/ureca modules:
    python scripts/patch_app_motion_ms_ureca_week1_v02.py

It injects:
- URECA imports
- Gradio controls for multi-keyframe GS fusion and held-out evaluation
- true multi-source infer_single_view + confidence-weighted canonical GS fusion
- render-only video saving for clean PSNR/SSIM/LPIPS
- held-out render metrics table + JSON/CSV reports

The patch is intentionally string-based and conservative. It makes a backup.
"""
from __future__ import annotations

from pathlib import Path

APP = Path("app_motion_ms.py")
if not APP.exists():
    raise FileNotFoundError("app_motion_ms.py not found. Run this script from LHM-main root.")

text = APP.read_text(encoding="utf-8")
backup = APP.with_suffix(".py.bak_ureca_week1_v02")
if not backup.exists():
    backup.write_text(text, encoding="utf-8")


def require(cond: bool, msg: str) -> None:
    if not cond:
        raise RuntimeError(msg)


def replace_once(src: str, old: str, new: str, label: str) -> str:
    if new in src:
        print(f"[skip] {label}: already patched")
        return src
    require(old in src, f"[fail] Could not find patch anchor for: {label}")
    print(f"[patch] {label}")
    return src.replace(old, new, 1)

# -----------------------------------------------------------------------------
# 1) Imports
# -----------------------------------------------------------------------------
old = "import pandas as pd\n"
new = """import pandas as pd
from LHM.ureca.experiment_io import (
    make_experiment_dir,
    save_json as ureca_save_json,
    save_csv as ureca_save_csv,
    copy_if_exists as ureca_copy_if_exists,
    dataframe_rows_from_render_metrics,
)
from LHM.ureca.keyframe_selector import (
    select_keyframes_from_smplx,
    select_source_keyframes,
)
from LHM.ureca.confidence_weighting import compute_source_confidence_weights
from LHM.ureca.gaussian_fusion import fuse_gs_model_lists
from LHM.ureca.heldout_eval import compute_render_video_metrics
from LHM.ureca.diffusion_refiner import refine_canonical_gs_with_diffusion
"""
if "from LHM.ureca.experiment_io" not in text:
    require(old in text, "Could not find import pandas anchor")
    text = text.replace(old, new, 1)
else:
    print("[skip] imports already patched")

# -----------------------------------------------------------------------------
# 2) core_fn signature
# -----------------------------------------------------------------------------
old = "    def core_fn(image: str, video_params, use_prefilter, use_pose_joint, use_trans_vel, pose_alpha, trans_alpha, working_dir):\n"
new = "    def core_fn(image: str, video_params, use_prefilter, use_pose_joint, use_trans_vel, pose_alpha, trans_alpha, enable_ureca_gs_fusion, ureca_source_path, ureca_num_keyframes, enable_conf_weighting, enable_diffusion_refiner, diffusion_ckpt_path, diffusion_steps, diffusion_strength, enable_heldout_eval, heldout_gt_video_path, working_dir):\n"
text = replace_once(text, old, new, "core_fn signature")

# -----------------------------------------------------------------------------
# 3) experiment directory initialization
# -----------------------------------------------------------------------------
old = """        with Image.fromarray(image) as img:
            img.save(image_raw)
        
        
        base_vid = os.path.basename(video_params).split(".")[0]
"""
new = """        with Image.fromarray(image) as img:
            img.save(image_raw)

        # URECA Week-1 experiment directory on /autodl-fs by default.
        ureca_exp_dir = make_experiment_dir(prefix="ureca_wk1")
        ureca_reports_dir = os.path.join(ureca_exp_dir, "reports")
        ureca_keyframes_dir = os.path.join(ureca_exp_dir, "selected_keyframes")
        ureca_render_dir = os.path.join(ureca_exp_dir, "render_only")
        ureca_copy_if_exists(image_raw, os.path.join(ureca_exp_dir, "input_image_raw.png"))
        ureca_copy_if_exists(video_params, os.path.join(ureca_exp_dir, "input_motion_video" + os.path.splitext(video_params)[1]))
        ureca_config = {
            "enable_ureca_gs_fusion": bool(enable_ureca_gs_fusion),
            "ureca_source_path": str(ureca_source_path or ""),
            "ureca_num_keyframes": int(ureca_num_keyframes),
            "enable_conf_weighting": bool(enable_conf_weighting),
            "enable_diffusion_refiner": bool(enable_diffusion_refiner),
            "diffusion_ckpt_path": str(diffusion_ckpt_path or ""),
            "diffusion_steps": int(diffusion_steps),
            "diffusion_strength": float(diffusion_strength),
            "enable_heldout_eval": bool(enable_heldout_eval),
            "heldout_gt_video_path": str(heldout_gt_video_path or ""),
            "pose_alpha": float(pose_alpha),
            "trans_alpha": float(trans_alpha),
        }
        ureca_save_json(ureca_config, os.path.join(ureca_reports_dir, "config.json"))
        print("[URECA-WK1] experiment_dir:", ureca_exp_dir)
        
        base_vid = os.path.basename(video_params).split(".")[0]
"""
text = replace_once(text, old, new, "experiment directory init")

# -----------------------------------------------------------------------------
# 4) SMPL-X keyframe report after baseline/improved dirs are known
# -----------------------------------------------------------------------------
old = """        # Compute metrics: only two rows (baseline vs improved combo)
        metrics_df = pd.DataFrame()
"""
new = """        # URECA Week-1: motion keyframe report. This is for audit/logging;
        # source-image keyframes for true GS fusion are selected later from ureca_source_path.
        try:
            kf_motion_report = select_keyframes_from_smplx(
                motion_seqs_dir_improved,
                num_keyframes=int(ureca_num_keyframes),
                min_gap=8,
                save_json=os.path.join(ureca_reports_dir, "motion_smplx_keyframes.json"),
            )
            print("[URECA-WK1] motion SMPL-X keyframes:", kf_motion_report)
        except Exception as e:
            print(f"[URECA-WK1] motion keyframe report skipped: {e}")

        # Compute metrics: only two rows (baseline vs improved combo)
        metrics_df = pd.DataFrame()
"""
text = replace_once(text, old, new, "SMPL-X keyframe report")

# -----------------------------------------------------------------------------
# 5) Extra source-image preprocessing helper, inserted after render settings.
# -----------------------------------------------------------------------------
old = """        motion_img_need_mask = cfg.get("motion_img_need_mask", False)  # False
        vis_motion = cfg.get("vis_motion", False)  # False

        # --- On-demand SAM2/rembg loading and execution ---
"""
new = """        motion_img_need_mask = cfg.get("motion_img_need_mask", False)  # False
        vis_motion = cfg.get("vis_motion", False)  # False

        def ureca_preprocess_source_image_as_tensors(src_path: str):
            # Same image pathway as the main source image, but without recomputing beta.
            # This keeps multi-keyframe inference low-intrusion and uses the main input beta.
            src_mask = None
            src_parsingnet = None
            try:
                print(f"[URECA-WK1] Loading SAM2Seg for source: {src_path}")
                src_parsingnet = SAM2Seg()
                with torch.no_grad():
                    src_parsing_out = src_parsingnet(img_path=src_path, bbox=None)
                    src_mask = (src_parsing_out.masks * 255).astype(np.uint8)
            except Exception as e:
                print(f"[URECA-WK1] SAM2 failed for {src_path}: {e}. Falling back to rembg.")
            finally:
                if src_parsingnet is not None:
                    del src_parsingnet
                    torch.cuda.empty_cache()
            if src_mask is None:
                img_np = cv2.imread(src_path)
                remove_np = remove(img_np)
                src_mask = remove_np[..., 3]
            src_image, _, _ = infer_preprocess_image(
                src_path,
                mask=src_mask,
                intr=None,
                pad_ratio=0,
                bg_color=1.0,
                max_tgt_size=896,
                aspect_standard=aspect_standard,
                enlarge_ratio=[1.0, 1.0],
                render_tgt_size=source_size,
                multiply=14,
                need_mask=True,
            )
            src_head = None
            try:
                device_local = avaliable_device()
                facedetector_src = VGGHeadDetector(
                    "./pretrained_models/gagatracker/vgghead/vgg_heads_l.trcd",
                    device=device_local,
                )
                rgb_face = np.array(Image.open(src_path))[..., :3]
                rgb_face = torch.from_numpy(rgb_face).permute(2, 0, 1).to(device_local)
                bbox = facedetector_src.detect_face(rgb_face)
                head_rgb = rgb_face[:, int(bbox[1]) : int(bbox[3]), int(bbox[0]) : int(bbox[2])]
                src_head = head_rgb.permute(1, 2, 0).cpu().numpy()
                del facedetector_src, rgb_face, head_rgb
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"[URECA-WK1] source face crop failed for {src_path}: {e}")
                src_head = np.zeros((112, 112, 3), dtype=np.uint8)
            try:
                src_head = cv2.resize(src_head, dsize=(cfg.src_head_size, cfg.src_head_size), interpolation=cv2.INTER_AREA)
            except Exception:
                src_head = np.zeros((cfg.src_head_size, cfg.src_head_size, 3), dtype=np.uint8)
            src_head = torch.from_numpy(src_head / 255.0).float().permute(2, 0, 1).unsqueeze(0)
            return src_image, src_head

        # --- On-demand SAM2/rembg loading and execution ---
"""
text = replace_once(text, old, new, "source-image preprocessing helper")

# -----------------------------------------------------------------------------
# 6) Replace single infer_single_view with true multi-source fusion.
# -----------------------------------------------------------------------------
old = """        gs_model_list, query_points, transform_mat_neutral_pose = lhm.infer_single_view(
            image.unsqueeze(0).to(device, dtype),
            src_head_rgb.unsqueeze(0).to(device, dtype),
            None,
            None,
            render_c2ws=motion_seq["render_c2ws"].to(device),
            render_intrs=motion_seq["render_intrs"].to(device),
            render_bg_colors=motion_seq["render_bg_colors"].to(device),
            smplx_params={
                k: v.to(device) for k, v in smplx_params.items()
            },
        )

        # rendering !!!!
"""
new = """        # URECA Week-1: true multi-source canonical 3DGS fusion.
        # Main uploaded image is always source #0. If ureca_source_path is provided,
        # additional keyframes are selected and independently passed through infer_single_view.
        ureca_source_records = [{"path": image_raw, "role": "main_uploaded_image"}]
        ureca_extra_source_paths = []
        if bool(enable_ureca_gs_fusion) and ureca_source_path and os.path.exists(str(ureca_source_path)):
            try:
                source_kf_report = select_source_keyframes(
                    str(ureca_source_path),
                    out_dir=ureca_keyframes_dir,
                    num_keyframes=int(ureca_num_keyframes),
                    min_gap=3,
                    video_stride=5,
                    save_json=os.path.join(ureca_reports_dir, "source_keyframes.json"),
                )
                ureca_extra_source_paths = list(source_kf_report.get("selected_paths", []))
                print("[URECA-WK1] selected source keyframes:", source_kf_report)
            except Exception as e:
                print(f"[URECA-WK1] source keyframe selection skipped: {e}")
        elif bool(enable_ureca_gs_fusion):
            print("[URECA-WK1] GS fusion enabled but no valid ureca_source_path provided; running single-source mode.")

        source_inputs = [(image_raw, image, src_head_rgb)]
        for p_src in ureca_extra_source_paths:
            try:
                img_i, head_i = ureca_preprocess_source_image_as_tensors(p_src)
                source_inputs.append((p_src, img_i, head_i))
                ureca_source_records.append({"path": p_src, "role": "selected_keyframe"})
            except Exception as e:
                print(f"[URECA-WK1] failed to preprocess source keyframe {p_src}: {e}")

        if not bool(enable_ureca_gs_fusion):
            source_inputs = source_inputs[:1]
            ureca_source_records = ureca_source_records[:1]

        weights = None
        conf_report = {"mode": "uniform", "weights": None}
        if bool(enable_conf_weighting) and len(source_inputs) >= 2:
            weights, conf_report = compute_source_confidence_weights([x[0] for x in source_inputs], mode="sharpness")
        else:
            weights, conf_report = compute_source_confidence_weights([x[0] for x in source_inputs], mode="uniform")
        ureca_save_json({"sources": ureca_source_records, "confidence": conf_report}, os.path.join(ureca_reports_dir, "source_confidence.json"))

        gs_sources = []
        query_points = None
        transform_mat_neutral_pose = None
        for src_idx, (src_path_i, image_i, head_i) in enumerate(source_inputs):
            print(f"[URECA-WK1] infer_single_view source {src_idx + 1}/{len(source_inputs)}: {src_path_i}")
            gs_i, qp_i, tm_i = lhm.infer_single_view(
                image_i.unsqueeze(0).to(device, dtype),
                head_i.unsqueeze(0).to(device, dtype),
                None,
                None,
                render_c2ws=motion_seq["render_c2ws"].to(device),
                render_intrs=motion_seq["render_intrs"].to(device),
                render_bg_colors=motion_seq["render_bg_colors"].to(device),
                smplx_params={
                    k: v.to(device) for k, v in smplx_params.items()
                },
            )
            gs_sources.append(gs_i)
            if query_points is None:
                query_points = qp_i
                transform_mat_neutral_pose = tm_i
            torch.cuda.empty_cache()

        gs_model_list, gs_fusion_report = fuse_gs_model_lists(
            gs_sources,
            weights=weights,
            mode="weighted_mean",
        )
        gs_fusion_report["source_paths"] = [x[0] for x in source_inputs]
        gs_fusion_report["confidence_report"] = conf_report
        ureca_save_json(gs_fusion_report, os.path.join(ureca_reports_dir, "canonical_gs_fusion_report.json"))
        print("[URECA-WK1] canonical GS fusion report:", gs_fusion_report)

        # URECA Week-1: exact reserved position for canonical diffusion refiner.
        # Week-1 keeps this hook identity-safe; Week-3 will load the trained Gaussian denoiser here.
        gs_model_list, diffusion_report = refine_canonical_gs_with_diffusion(
            gs_model_list,
            enable=bool(enable_diffusion_refiner),
            checkpoint_path=str(diffusion_ckpt_path or "").strip() or None,
            num_steps=int(diffusion_steps),
            strength=float(diffusion_strength),
        )
        ureca_save_json(diffusion_report, os.path.join(ureca_reports_dir, "canonical_diffusion_refiner_report.json"))
        print("[URECA-WK1] canonical diffusion refiner report:", diffusion_report)

        # rendering !!!!
"""
text = replace_once(text, old, new, "multi-source infer + GS fusion")

# -----------------------------------------------------------------------------
# 7) Render-only baseline video before optional vis concat.
# -----------------------------------------------------------------------------
old = """        rgb = np.concatenate(batch_list, axis=0)
        print(f"time elapsed: {time.time() - start_time}")

        if vis_motion:
"""
new = """        rgb = np.concatenate(batch_list, axis=0)
        print(f"time elapsed: {time.time() - start_time}")

        # URECA Week-1: save render-only video BEFORE optional visualization concatenation.
        dump_video_path_render_only = os.path.join(ureca_render_dir, "baseline_render_only.mp4")
        images_to_video(
            rgb.copy(),
            output_path=dump_video_path_render_only,
            fps=render_fps,
            gradio_codec=False,
            verbose=True,
        )

        if vis_motion:
"""
text = replace_once(text, old, new, "baseline render-only video")

# -----------------------------------------------------------------------------
# 8) Render-only improved video before optional vis concat.
# -----------------------------------------------------------------------------
old = """        rgb2 = np.concatenate(batch_list2, axis=0)

        if vis_motion:
"""
new = """        rgb2 = np.concatenate(batch_list2, axis=0)

        # URECA Week-1: save render-only improved video BEFORE optional visualization concatenation.
        dump_video_path_improved_render_only = os.path.join(ureca_render_dir, "improved_render_only.mp4")
        images_to_video(
            rgb2.copy(),
            output_path=dump_video_path_improved_render_only,
            fps=render_fps,
            gradio_codec=False,
            verbose=True,
        )

        if vis_motion:
"""
text = replace_once(text, old, new, "improved render-only video")

# -----------------------------------------------------------------------------
# 9) Held-out eval before return.
# -----------------------------------------------------------------------------
old = """        return dump_image_path, dump_video_path, dump_video_path_improved, final_metrics_for_gradio
"""
new = """        # URECA Week-1: held-out evaluation uses render-only videos, never side-by-side/visualization videos.
        render_metrics = {
            "baseline_render_only": {"valid_eval": False, "psnr": float("nan"), "ssim": float("nan"), "lpips": float("nan"), "num_eval_frames": 0},
            "improved_render_only": {"valid_eval": False, "psnr": float("nan"), "ssim": float("nan"), "lpips": float("nan"), "num_eval_frames": 0},
        }
        try:
            if bool(enable_heldout_eval):
                gt_path = str(heldout_gt_video_path or "").strip()
                render_metrics["baseline_render_only"] = compute_render_video_metrics(dump_video_path_render_only, gt_video=gt_path, max_frames=120)
                render_metrics["improved_render_only"] = compute_render_video_metrics(dump_video_path_improved_render_only, gt_video=gt_path, max_frames=120)
            else:
                print("[URECA-WK1] held-out eval disabled by checkbox.")
            ureca_save_json(render_metrics, os.path.join(ureca_reports_dir, "heldout_render_metrics.json"))
            ureca_save_csv([
                {"variant": k, **v} for k, v in render_metrics.items()
            ], os.path.join(ureca_reports_dir, "heldout_render_metrics.csv"))
            print("[URECA-WK1] held-out render metrics:", render_metrics)
        except Exception as e:
            print(f"[URECA-WK1] held-out render metrics skipped: {e}")
        render_metrics_for_gradio = dataframe_rows_from_render_metrics(render_metrics)

        return dump_image_path, dump_video_path, dump_video_path_improved, final_metrics_for_gradio, render_metrics_for_gradio
"""
text = replace_once(text, old, new, "held-out eval + return")

# -----------------------------------------------------------------------------
# 10) UI controls
# -----------------------------------------------------------------------------
old = """                trans_alpha_slider = gr.Slider(label="Trans EMA alpha", minimum=0.0, maximum=1.0, value=0.25, step=0.01)
                submit = gr.Button('Generate', elem_id="openlrm_generate", variant='primary')
"""
new = """                trans_alpha_slider = gr.Slider(label="Trans EMA alpha", minimum=0.0, maximum=1.0, value=0.25, step=0.01)
                gr.Markdown("### URECA Week-1 Modules")
                enable_ureca_gs_fusion_checkbox = gr.Checkbox(label="Multi-keyframe canonical GS fusion", value=True)
                ureca_source_path_text = gr.Textbox(label="URECA source frames folder / source video path", value="", placeholder="/autodl-fs/data/lhm_shared/ureca_inputs/source_view_A_frames or source_view_A.mp4")
                ureca_num_keyframes_slider = gr.Slider(label="Number of source keyframes", minimum=1, maximum=8, value=3, step=1)
                enable_conf_weighting_checkbox = gr.Checkbox(label="Confidence-weighted fusion", value=True)
                enable_diffusion_refiner_checkbox = gr.Checkbox(label="Canonical diffusion refiner hook", value=False)
                diffusion_ckpt_path_text = gr.Textbox(label="Diffusion checkpoint path", value="", placeholder="Week-3: /autodl-fs/data/lhm_shared/ureca_checkpoints/gs_diffusion_refiner.pt")
                diffusion_steps_slider = gr.Slider(label="Diffusion steps", minimum=1, maximum=50, value=10, step=1)
                diffusion_strength_slider = gr.Slider(label="Diffusion strength", minimum=0.0, maximum=1.0, value=0.3, step=0.01)
                enable_heldout_eval_checkbox = gr.Checkbox(label="Held-out view evaluation", value=False)
                heldout_gt_video_text = gr.Textbox(label="Held-out GT video path", value="", placeholder="/autodl-fs/data/lhm_shared/ureca_inputs/heldout_view_B.mp4")
                submit = gr.Button('Generate', elem_id="openlrm_generate", variant='primary')
"""
text = replace_once(text, old, new, "URECA UI controls")

# -----------------------------------------------------------------------------
# 11) Render metrics DataFrame
# -----------------------------------------------------------------------------
old = """        metrics_table = gr.Dataframe(headers=["variant", "jitter_pose ↓", "jitter_trans ↓", "drift_trans ↓", "deviation_pose ↓", "deviation_trans ↓"], label="Metrics: baseline vs improved", interactive=False)

        submit.click(
"""
new = """        metrics_table = gr.Dataframe(headers=["variant", "jitter_pose ↓", "jitter_trans ↓", "drift_trans ↓", "deviation_pose ↓", "deviation_trans ↓"], label="Metrics: baseline vs improved", interactive=False)
        render_metrics_table = gr.Dataframe(headers=["variant", "valid_eval", "PSNR ↑", "SSIM ↑", "LPIPS ↓", "frames"], label="URECA held-out render-only metrics", interactive=False)

        submit.click(
"""
text = replace_once(text, old, new, "render metrics DataFrame")

# -----------------------------------------------------------------------------
# 12) submit inputs/outputs
# -----------------------------------------------------------------------------
old = """            inputs=[input_image, video_input, use_prefilter_checkbox, use_pose_joint_checkbox, use_trans_vel_checkbox, pose_alpha_slider, trans_alpha_slider, working_dir],
            outputs=[processed_image, output_video_baseline, output_video_improved, metrics_table],
"""
new = """            inputs=[input_image, video_input, use_prefilter_checkbox, use_pose_joint_checkbox, use_trans_vel_checkbox, pose_alpha_slider, trans_alpha_slider, enable_ureca_gs_fusion_checkbox, ureca_source_path_text, ureca_num_keyframes_slider, enable_conf_weighting_checkbox, enable_diffusion_refiner_checkbox, diffusion_ckpt_path_text, diffusion_steps_slider, diffusion_strength_slider, enable_heldout_eval_checkbox, heldout_gt_video_text, working_dir],
            outputs=[processed_image, output_video_baseline, output_video_improved, metrics_table, render_metrics_table],
"""
text = replace_once(text, old, new, "submit inputs/outputs")

APP.write_text(text, encoding="utf-8")
print("Patched app_motion_ms.py successfully.")
print("Backup:", backup)
print("Next: python -m py_compile app_motion_ms.py LHM/ureca/*.py")
