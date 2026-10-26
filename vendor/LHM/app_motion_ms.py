# --- project metric/ui helper imports ---
from pathlib import Path
import json as _project_json
# --- end project metric/ui helper imports ---

# Copyright (c) 2023-2024, Qi Zuo & Lingteng Qiu
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# memory save version


import base64
import os
import tempfile
import time

import cv2
import gradio as gr
import numpy as np
import spaces
import torch
from PIL import Image
from collections import defaultdict

torch._dynamo.config.disable = True
import argparse
import os
import pdb
import shutil
import subprocess

import torch
from accelerate import Accelerator
from omegaconf import OmegaConf

from engine.pose_estimation.pose_estimator import PoseEstimator
from engine.SegmentAPI.base import Bbox
from LHM.utils.model_download_utils import AutoModelQuery
from LHM.utils.model_query_utils import AutoModelSwitcher

# Initialize remove function to None initially
remove = None
try:
    from engine.SegmentAPI.SAM import SAM2Seg
    print("SAM2Seg found and imported.")
except ImportError:
    print("\033[31mSAM2Seg import failed. Attempting to import rembg...\033[0m")
    try:
        from rembg import remove
        print("rembg imported successfully.")
    except ImportError:
        print("\033[31mError: Neither SAM2Seg nor rembg could be imported. Background removal will not work.\033[0m")
except Exception as e:
    print(f"\033[31mAn unexpected error occurred during import: {e}\033[0m")
    # Attempt to import rembg as a fallback even on unexpected errors during SAM2 import
    try:
        from rembg import remove
        print("rembg imported successfully as fallback.")
    except ImportError:
         print("\033[31mError: rembg could not be imported as fallback either.\033[0m")


from engine.pose_estimation.video2motion import Video2MotionPipeline
from engine.pose_estimation.pose_utils.postprocess import (
    make_variant_from_raw,
    compute_metrics_vs,
    load_smplx_dir,
    compute_self_metrics,
)
import pandas as pd
from LHM.runners.infer.utils import (
    calc_new_tgt_size_by_aspect,
    center_crop_according_to_mask,
    prepare_motion_seqs,
    resize_image_keepaspect_np,
)
from LHM.utils.download_utils import download_extract_tar_from_url, download_from_url
from LHM.utils.face_detector import VGGHeadDetector
from LHM.utils.ffmpeg_utils import images_to_video
from LHM.utils.gpu_utils import check_single_gpu_memory
from LHM.utils.hf_hub import wrap_model_hub
from LHM.utils.model_card import MEMORY_MODEL_CARD, MODEL_CARD, MODEL_CONFIG
from LHM.utils.video_utils import get_video_hash


def download_geo_files():
    if not os.path.exists('./pretrained_models/dense_sample_points/1_20000.ply'):
        download_from_url('https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/aigc3d/data/LHM/1_20000.ply','./pretrained_models/dense_sample_points/')

def avaliable_device():
    if torch.cuda.is_available():
        current_device_id = torch.cuda.current_device()
        device = f"cuda:{current_device_id}"
    else:
        device = "cpu"

    return device

def query_model_config(model_name):
    try:
        model_params = model_name.split('-')[1]
        
        return MODEL_CONFIG[model_params] 
    except:
        return None

def prior_check():
    if not os.path.exists('./pretrained_models'):
        prior_data = MODEL_CARD['prior_model']
        download_extract_tar_from_url(prior_data)

def get_bbox(mask):
    height, width = mask.shape
    pha = mask / 255.0
    pha[pha < 0.5] = 0.0
    pha[pha >= 0.5] = 1.0

    # obtain bbox
    _h, _w = np.where(pha == 1)

    whwh = [
        _w.min().item(),
        _h.min().item(),
        _w.max().item(),
        _h.max().item(),
    ]

    box = Bbox(whwh)

    # scale box to 1.05
    scale_box = box.scale(1.1, width=width, height=height)
    return scale_box


def infer_preprocess_image(
    rgb_path,
    mask,
    intr,
    pad_ratio,
    bg_color,
    max_tgt_size,
    aspect_standard,
    enlarge_ratio,
    render_tgt_size,
    multiply,
    need_mask=True,
):
    """inferece
    image, _, _ = preprocess_image(image_path, mask_path=None, intr=None, pad_ratio=0, bg_color=1.0,
                                        max_tgt_size=896, aspect_standard=aspect_standard, enlarge_ratio=[1.0, 1.0],
                                        render_tgt_size=source_size, multiply=14, need_mask=True)

    """

    rgb = np.array(Image.open(rgb_path))
    rgb_raw = rgb.copy()

    bbox = get_bbox(mask)
    bbox_list = bbox.get_box()

    rgb = rgb[bbox_list[1] : bbox_list[3], bbox_list[0] : bbox_list[2]]
    mask = mask[bbox_list[1] : bbox_list[3], bbox_list[0] : bbox_list[2]]


    h, w, _ = rgb.shape
    assert w < h
    cur_ratio = h / w
    scale_ratio = cur_ratio / aspect_standard


    target_w = int(min(w * scale_ratio, h))
    if target_w - w >0:
        offset_w = (target_w - w) // 2

        rgb = np.pad(
            rgb,
            ((0, 0), (offset_w, offset_w), (0, 0)),
            mode="constant",
            constant_values=255,
        )

        mask = np.pad(
            mask,
            ((0, 0), (offset_w, offset_w)),
            mode="constant",
            constant_values=0,
        )
    else:
        target_h = w * aspect_standard
        offset_h = int(target_h - h)

        rgb = np.pad(
            rgb,
            ((offset_h, 0), (0, 0), (0, 0)),
            mode="constant",
            constant_values=255,
        )

        mask = np.pad(
            mask,
            ((offset_h, 0), (0, 0)),
            mode="constant",
            constant_values=0,
        )

    rgb = rgb / 255.0  # normalize to [0, 1]
    mask = mask / 255.0

    mask = (mask > 0.5).astype(np.float32)
    rgb = rgb[:, :, :3] * mask[:, :, None] + bg_color * (1 - mask[:, :, None])

    # resize to specific size require by preprocessor of smplx-estimator.
    rgb = resize_image_keepaspect_np(rgb, max_tgt_size)
    mask = resize_image_keepaspect_np(mask, max_tgt_size)

    # crop image to enlarge human area.
    rgb, mask, offset_x, offset_y = center_crop_according_to_mask(
        rgb, mask, aspect_standard, enlarge_ratio
    )
    if intr is not None:
        intr[0, 2] -= offset_x
        intr[1, 2] -= offset_y

    # resize to render_tgt_size for training

    tgt_hw_size, ratio_y, ratio_x = calc_new_tgt_size_by_aspect(
        cur_hw=rgb.shape[:2],
        aspect_standard=aspect_standard,
        tgt_size=render_tgt_size,
        multiply=multiply,
    )

    rgb = cv2.resize(
        rgb, dsize=(tgt_hw_size[1], tgt_hw_size[0]), interpolation=cv2.INTER_AREA
    )
    mask = cv2.resize(
        mask, dsize=(tgt_hw_size[1], tgt_hw_size[0]), interpolation=cv2.INTER_AREA
    )

    if intr is not None:

        # ******************** Merge *********************** #
        intr = scale_intrs(intr, ratio_x=ratio_x, ratio_y=ratio_y)
        assert (
            abs(intr[0, 2] * 2 - rgb.shape[1]) < 2.5
        ), f"{intr[0, 2] * 2}, {rgb.shape[1]}"
        assert (
            abs(intr[1, 2] * 2 - rgb.shape[0]) < 2.5
        ), f"{intr[1, 2] * 2}, {rgb.shape[0]}"

        # ******************** Merge *********************** #
        intr[0, 2] = rgb.shape[1] // 2
        intr[1, 2] = rgb.shape[0] // 2

    rgb = torch.from_numpy(rgb).float().permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
    mask = (
        torch.from_numpy(mask[:, :, None]).float().permute(2, 0, 1).unsqueeze(0)
    )  # [1, 1, H, W]
    return rgb, mask, intr

def parse_configs():

    cli_cfg = OmegaConf.create()
    cfg = OmegaConf.create()

    query_model = AutoModelQuery()

    # parse from ENV
    if os.environ.get("APP_MODEL_NAME") is not None:
        model_path = query_model.query(os.environ.get("APP_MODEL_NAME"))
        model_name = os.environ.get("APP_MODEL_NAME")
    else:
        raise NotImplementedError

    cli_cfg.model_name = model_path 

    model_config = query_model_config(model_name)

    if model_config is not None:
        cfg_train = OmegaConf.load(model_config)
        cfg.source_size = cfg_train.dataset.source_image_res
        try:
            cfg.src_head_size = cfg_train.dataset.src_head_size
        except:
            cfg.src_head_size = 112
        cfg.render_size = cfg_train.dataset.render_image.high
        _relative_path = os.path.join(
            cfg_train.experiment.parent,
            cfg_train.experiment.child,
            os.path.basename(cli_cfg.model_name).split("_")[-1],
        )

        cfg.save_tmp_dump = os.path.join("exps", "save_tmp", _relative_path)
        cfg.image_dump = os.path.join("exps", "images", _relative_path)
        cfg.video_dump = os.path.join("exps", "videos", _relative_path)  # output path

    cfg.motion_video_read_fps = 6
    cfg.merge_with(cli_cfg)

    cfg.setdefault("logger", "INFO")

    assert cfg.model_name is not None, "model_name is required"

    return cfg, cfg_train

def _build_model(cfg):
    from LHM.models import model_dict

    hf_model_cls = wrap_model_hub(model_dict["human_lrm_sapdino_bh_sd3_5"])
    model = hf_model_cls.from_pretrained(cfg.model_name)
    model = model.half()
    return model

def animation_infer(renderer, gs_model_list, query_points, smplx_params, render_c2ws, render_intrs, render_bg_colors):
    '''Inference code avoid repeat forward.
    '''
    render_h, render_w = int(render_intrs[0, 0, 1, 2] * 2), int(
        render_intrs[0, 0, 0, 2] * 2
    )
    # render target views
    render_res_list = []
    num_views = render_c2ws.shape[1]
    start_time = time.time()

    # render target views
    render_res_list = []

    for view_idx in range(num_views):
        render_res = renderer.forward_animate_gs(
            gs_model_list,
            query_points,
            renderer.get_single_view_smpl_data(smplx_params, view_idx),
            render_c2ws[:, view_idx : view_idx + 1],
            render_intrs[:, view_idx : view_idx + 1],
            render_h,
            render_w,
            render_bg_colors[:, view_idx : view_idx + 1],
        )
        render_res_list.append(render_res)
    print(
        f"time elpased(animate gs model per frame):{(time.time() -  start_time)/num_views}"
    )

    out = defaultdict(list)
    for res in render_res_list:
        for k, v in res.items():
            if isinstance(v[0], torch.Tensor):
                out[k].append(v.detach().cpu())
            else:
                out[k].append(v)
    for k, v in out.items():
        # print(f"out key:{k}")
        if isinstance(v[0], torch.Tensor):
            out[k] = torch.concat(v, dim=1)
            if k in ["comp_rgb", "comp_mask", "comp_depth"]:
                out[k] = out[k][0].permute(
                    0, 2, 3, 1
                )  # [1, Nv, 3, H, W] -> [Nv, 3, H, W] - > [Nv, H, W, 3]
        else:
            out[k] = v
    return out

def assert_input_image(input_image):
    if input_image is None:
        raise gr.Error("No image selected or uploaded!")

def prepare_working_dir():
    import tempfile
    working_dir = tempfile.TemporaryDirectory()
    return working_dir

def init_preprocessor():
    from LHM.utils.preprocess import Preprocessor
    global preprocessor
    preprocessor = Preprocessor()

def preprocess_fn(image_in: np.ndarray, remove_bg: bool, recenter: bool, working_dir):
    image_raw = os.path.join(working_dir.name, "raw.png")
    with Image.fromarray(image_in) as img:
        img.save(image_raw)
    image_out = os.path.join(working_dir.name, "rembg.png")
    success = preprocessor.preprocess(image_path=image_raw, save_path=image_out, rmbg=remove_bg, recenter=recenter)
    assert success, f"Failed under preprocess_fn!"
    return image_out

def get_image_base64(path):
    with open(path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    return f"data:image/png;base64,{encoded_string}"


# ========================= project upload/path helpers =========================
def _get_upload_path(x):
    """Return a filesystem path from Gradio upload outputs or a plain text path."""
    if x is None:
        return None

    if isinstance(x, str):
        return x

    if isinstance(x, dict):
        return x.get("path") or x.get("name")

    if hasattr(x, "path"):
        return x.path

    if hasattr(x, "name"):
        return x.name

    return str(x)


def _get_uploaded_folder(files):
    """Return the common parent folder for a Gradio directory upload."""
    if not files:
        return None

    if not isinstance(files, (list, tuple)):
        files = [files]

    paths = []
    for item in files:
        path = _get_upload_path(item)
        if path:
            paths.append(os.path.abspath(path))

    if not paths:
        return None

    if len(paths) == 1:
        return os.path.dirname(paths[0])

    return os.path.commonpath(paths)


def resolve_source_input(source_frames_upload=None, source_video_upload=None, source_path_text=None):
    """Resolve multi-keyframe source input. Server path has priority for reproducible AutoDL experiments."""
    if source_path_text and str(source_path_text).strip():
        source_path = str(source_path_text).strip()
        if not os.path.exists(source_path):
            raise gr.Error(f"Source frames/video path does not exist: {source_path}")
        return source_path

    source_frames_dir = _get_uploaded_folder(source_frames_upload)
    if source_frames_dir:
        return source_frames_dir

    source_video_path = _get_upload_path(source_video_upload)
    if source_video_path:
        return source_video_path

    return None


def resolve_video_input(video_upload=None, path_text=None):
    """Resolve a video input. Server path has priority so default/example uploads do not override experiment paths."""
    if path_text and str(path_text).strip():
        video_path = str(path_text).strip()
        if not os.path.exists(video_path):
            raise gr.Error(f"Video path does not exist: {video_path}")
        return video_path

    video_path = _get_upload_path(video_upload)
    if video_path:
        return video_path

    return None


def resolve_image_input(image_upload=None, path_text=None):
    """Resolve an image input. Server path has priority so default/example uploads do not override experiment paths."""
    if path_text and str(path_text).strip():
        image_path = str(path_text).strip()
        if not os.path.exists(image_path):
            raise gr.Error(f"Input image path does not exist: {image_path}")
        with Image.open(image_path) as img:
            return np.array(img.convert("RGBA"))

    if image_upload is not None:
        return image_upload

    return None


def assert_core_inputs(input_image, input_image_path, input_video, input_video_path):
    """Validate that image and target motion are provided by upload or server path."""
    has_image_upload = input_image is not None
    has_image_path = bool(input_image_path and str(input_image_path).strip())
    has_video_upload = input_video is not None
    has_video_path = bool(input_video_path and str(input_video_path).strip())

    if not (has_image_upload or has_image_path):
        raise gr.Error("Please upload an input image or provide a server image path.")

    if has_image_path and not os.path.exists(str(input_image_path).strip()):
        raise gr.Error(f"Input image path does not exist: {str(input_image_path).strip()}")

    if not (has_video_upload or has_video_path):
        raise gr.Error("Please upload a target motion video or provide a server video path.")

    if has_video_path and not os.path.exists(str(input_video_path).strip()):
        raise gr.Error(f"Target motion video path does not exist: {str(input_video_path).strip()}")



def _read_video_rgb_frames(video_path, max_frames=None):
    """Read video frames as RGB uint8 arrays."""
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if max_frames is not None and len(frames) >= max_frames:
            break
    cap.release()
    return frames


def _simple_psnr(pred, gt):
    pred = pred.astype(np.float32)
    gt = gt.astype(np.float32)
    mse = np.mean((pred - gt) ** 2)
    if mse <= 1e-12:
        return float("inf")
    return float(20.0 * np.log10(255.0 / np.sqrt(mse)))


def _simple_ssim(pred, gt):
    try:
        from skimage.metrics import structural_similarity
        return float(structural_similarity(gt, pred, channel_axis=2, data_range=255))
    except Exception as e:
        print(f"[heldout metrics] SSIM unavailable: {e}")
        return float("nan")


def _sorted_mask_paths(mask_dir):
    from pathlib import Path
    if not mask_dir:
        return []
    mask_dir = str(mask_dir).strip()
    if not mask_dir or not os.path.isdir(mask_dir):
        return []
    paths = []
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        paths.extend(Path(mask_dir).glob(ext))
    return sorted(paths)


def _crop_by_mask(pred, gt, mask):
    if mask is None:
        return pred, gt
    if mask.ndim == 3:
        mask = mask[..., 0]
    mask = (mask > 0).astype(np.uint8)
    if mask.sum() == 0:
        return pred, gt
    x, y, w, h = cv2.boundingRect(mask)
    if w <= 0 or h <= 0:
        return pred, gt
    return pred[y:y+h, x:x+w], gt[y:y+h, x:x+w]


def compute_heldout_video_metrics(rendered_video_path, gt_video_path, mask_dir=None, variant="rendered", enable_lpips=False):
    """Compute lightweight held-out video metrics for display in Gradio."""
    if not rendered_video_path or not gt_video_path:
        return [variant, float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), float("nan")]
    if not os.path.exists(str(rendered_video_path)) or not os.path.exists(str(gt_video_path)):
        return [variant, float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), float("nan")]

    pred_frames = _read_video_rgb_frames(rendered_video_path)
    gt_frames = _read_video_rgb_frames(gt_video_path)
    n = min(len(pred_frames), len(gt_frames))
    if n == 0:
        return [variant, float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), float("nan")]

    mask_paths = _sorted_mask_paths(mask_dir)

    psnrs, ssims, psnrs_fg, ssims_fg = [], [], [], []
    temporal_delta_errors = []

    prev_pred = None
    prev_gt = None
    for i in range(n):
        pred = pred_frames[i]
        gt = gt_frames[i]
        if pred.shape[:2] != gt.shape[:2]:
            pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_AREA)

        psnrs.append(_simple_psnr(pred, gt))
        ssims.append(_simple_ssim(pred, gt))

        if i < len(mask_paths):
            mask = cv2.imread(str(mask_paths[i]), cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                if mask.shape[:2] != gt.shape[:2]:
                    mask = cv2.resize(mask, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
                pred_crop, gt_crop = _crop_by_mask(pred, gt, mask)
                if pred_crop.size > 0 and gt_crop.size > 0:
                    psnrs_fg.append(_simple_psnr(pred_crop, gt_crop))
                    ssims_fg.append(_simple_ssim(pred_crop, gt_crop))

        if prev_pred is not None and prev_gt is not None:
            pred_delta = pred.astype(np.float32) - prev_pred.astype(np.float32)
            gt_delta = gt.astype(np.float32) - prev_gt.astype(np.float32)
            temporal_delta_errors.append(float(np.mean(np.abs(pred_delta - gt_delta))))

        prev_pred = pred
        prev_gt = gt

    def safe_mean(xs):
        xs = [x for x in xs if not (isinstance(x, float) and np.isnan(x))]
        return float(np.mean(xs)) if xs else float("nan")

    return [
        str(variant),
        safe_mean(psnrs),
        safe_mean(ssims),
        safe_mean(psnrs_fg),
        safe_mean(ssims_fg),
        safe_mean(temporal_delta_errors),
        _compute_lpips_mean_rgb(pred_frames, gt_frames) if bool(enable_lpips) else float("nan"),
    ]



def _as_existing_path_for_metrics(x):
    """Normalize Gradio/file/text paths for held-out metric computation."""
    if x is None:
        return None
    if isinstance(x, (list, tuple)):
        for item in x:
            p = _as_existing_path_for_metrics(item)
            if p and os.path.exists(p):
                return p
        return None
    if isinstance(x, dict):
        for k in ("path", "name", "orig_name"):
            v = x.get(k)
            if v:
                p = _as_existing_path_for_metrics(v)
                if p:
                    return p
        return None
    if hasattr(x, "name"):
        return _as_existing_path_for_metrics(x.name)
    p = str(x).strip().strip("'\"")
    return p if p else None


def _heldout_status_row(status):
    return [str(status), float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), float("nan")]


def compute_heldout_rows_safe(baseline_video_path, improved_video_path, heldout_gt_path, mask_dir=None, enable_lpips=False):
    """Return two Gradio rows and never silently disappear."""
    baseline_video_path = _as_existing_path_for_metrics(baseline_video_path)
    improved_video_path = _as_existing_path_for_metrics(improved_video_path)
    heldout_gt_path = _as_existing_path_for_metrics(heldout_gt_path)
    mask_dir = _as_existing_path_for_metrics(mask_dir)

    print("[heldout metrics safe] baseline=", baseline_video_path, "exists=", os.path.exists(baseline_video_path) if baseline_video_path else None)
    print("[heldout metrics safe] improved=", improved_video_path, "exists=", os.path.exists(improved_video_path) if improved_video_path else None)
    print("[heldout metrics safe] gt=", heldout_gt_path, "exists=", os.path.exists(heldout_gt_path) if heldout_gt_path else None)
    print("[heldout metrics safe] mask_dir=", mask_dir, "exists=", os.path.isdir(mask_dir) if mask_dir else None)

    if not heldout_gt_path or not os.path.exists(heldout_gt_path):
        return [_heldout_status_row("missing_gt_path")]

    rows = []
    if baseline_video_path and os.path.exists(baseline_video_path):
        rows.append(compute_heldout_video_metrics(baseline_video_path, heldout_gt_path, mask_dir, variant="single_keyframe_lhm_vs_heldout", enable_lpips=enable_lpips))
    else:
        rows.append(_heldout_status_row("missing_baseline_video"))

    if improved_video_path and os.path.exists(improved_video_path):
        rows.append(compute_heldout_video_metrics(improved_video_path, heldout_gt_path, mask_dir, variant="temporal_stabilized_lhm_vs_heldout", enable_lpips=enable_lpips))
    else:
        rows.append(_heldout_status_row("missing_improved_video"))

    # Save a copy to FS if configured, so metrics are not only visible in Gradio.
    try:
        results_root = os.environ.get("LHM_Project_RESULTS_ROOT", "/autodl-fs/data/lhm_shared/poster_results")
        out_dir = os.path.join(results_root, "zju_313_wk2", "gradio_heldout_metrics")
        os.makedirs(out_dir, exist_ok=True)
        import json as _json
        with open(os.path.join(out_dir, "latest_heldout_metrics.json"), "w") as f:
            _json.dump({"baseline_video": baseline_video_path, "improved_video": improved_video_path, "heldout_gt": heldout_gt_path, "mask_dir": mask_dir, "rows": rows}, f, indent=2)
        print("[heldout metrics safe] saved latest metrics to", os.path.join(out_dir, "latest_heldout_metrics.json"))
    except Exception as e:
        print("[heldout metrics safe] save failed:", e)

    return rows

# ======================= end project upload/path helpers =======================





# ========================= LPIPS / comparison / source-selection helpers =========================

def _safe_float_mean(values):
    vals = []
    for v in values:
        try:
            if v is None:
                continue
            fv = float(v)
            if np.isnan(fv):
                continue
            vals.append(fv)
        except Exception:
            continue
    return float(np.mean(vals)) if vals else float("nan")


def _compute_lpips_mean_rgb(pred_frames, gt_frames, max_frames=20):
    """Optional local LPIPS computation. Returns NaN if lpips/torch/weights are unavailable."""
    try:
        import torch
        import lpips
    except Exception as e:
        print(f"[LPIPS] package unavailable: {e}")
        return float("nan")

    n = min(len(pred_frames), len(gt_frames))
    if n <= 0:
        return float("nan")
    if max_frames is not None and n > int(max_frames):
        frame_ids = np.linspace(0, n - 1, int(max_frames)).round().astype(int).tolist()
    else:
        frame_ids = list(range(n))

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        loss_fn = lpips.LPIPS(net="alex").to(device).eval()
        scores = []
        with torch.no_grad():
            for i in frame_ids:
                pred = pred_frames[i]
                gt = gt_frames[i]
                if pred.shape[:2] != gt.shape[:2]:
                    pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_AREA)
                pred_t = torch.from_numpy(pred).float().permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0
                gt_t = torch.from_numpy(gt).float().permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0
                pred_t = pred_t.to(device)
                gt_t = gt_t.to(device)
                score = loss_fn(pred_t, gt_t).detach().float().cpu().item()
                scores.append(score)
        return _safe_float_mean(scores)
    except Exception as e:
        print(f"[LPIPS] computation failed: {e}")
        return float("nan")


def _list_source_frame_paths(source_path, max_items=None):
    """List image frames from a folder path. Used for source-selection status/prototype."""
    if not source_path:
        return []
    source_path = str(source_path).strip()
    if not os.path.isdir(source_path):
        return []
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    paths = []
    for name in sorted(os.listdir(source_path)):
        p = os.path.join(source_path, name)
        if os.path.isfile(p) and os.path.splitext(name)[1].lower() in exts:
            paths.append(p)
    if max_items is not None:
        return paths[: int(max_items)]
    return paths


def _select_best_source_frame(source_path, requested_keyframes=5):
    """A lightweight v0 multi-keyframe source-selection prototype.

    It does not perform canonical GS fusion. It scores the available source frames using
    image sharpness and a simple foreground-size proxy, then returns the best frame path
    and a short note for the status table.
    """
    frames = _list_source_frame_paths(source_path)
    if not frames:
        return None, "no selectable source image frames found"
    if requested_keyframes is not None and int(requested_keyframes) > 0:
        # Uniformly sample the available frames to respect the requested keyframe count.
        n_req = min(int(requested_keyframes), len(frames))
        ids = np.linspace(0, len(frames) - 1, n_req).round().astype(int).tolist()
        frames_to_score = [frames[i] for i in ids]
    else:
        frames_to_score = frames

    best = None
    best_score = -1e18
    debug = []
    for p in frames_to_score:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        # Simple foreground proxy: non-white area ratio. Works for already segmented/keyframe images.
        nonwhite = np.mean(np.any(img < 245, axis=2))
        score = sharpness * (0.5 + float(nonwhite))
        debug.append((score, sharpness, float(nonwhite), p))
        if score > best_score:
            best_score = score
            best = p
    if best is None:
        return None, f"{len(frames)} frame(s) found but none readable"
    debug_sorted = sorted(debug, reverse=True)[:3]
    short = os.path.basename(best)
    note = f"prototype active: selected {short} from {len(frames_to_score)}/{len(frames)} scored frame(s); true GS-level fusion still not active"
    return best, note


def _read_bgr_video_frames_for_grid(video_path, max_frames=None):
    if not video_path or not os.path.exists(str(video_path)):
        return []
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
        if max_frames is not None and len(frames) >= int(max_frames):
            break
    cap.release()
    return frames


def _fit_frame_bgr(frame, width, height):
    if frame is None:
        return np.full((height, width, 3), 255, dtype=np.uint8)
    h, w = frame.shape[:2]
    if h <= 0 or w <= 0:
        return np.full((height, width, 3), 255, dtype=np.uint8)
    scale = min(width / w, height / h)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    y0 = (height - nh) // 2
    x0 = (width - nw) // 2
    canvas[y0:y0+nh, x0:x0+nw] = resized
    return canvas


def _put_label_bgr(frame, label):
    out = frame.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 34), (245, 245, 245), -1)
    cv2.putText(out, str(label), (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 20, 20), 2, cv2.LINE_AA)
    return out


def make_four_way_comparison_video(source_video, gt_video, baseline_video, improved_video, out_path, fps=30, cell_w=360, cell_h=360):
    """Create a 2x2 comparison: source motion, held-out GT, baseline render, temporal-stabilized render."""
    videos = [source_video, gt_video, baseline_video, improved_video]
    labels = ["Source / target motion", "Held-out GT", "Single-keyframe LHM", "Temporal-stabilized"]
    frame_lists = [_read_bgr_video_frames_for_grid(v) for v in videos]
    lengths = [len(x) for x in frame_lists if len(x) > 0]
    if not lengths:
        return None
    n = min(lengths)
    if n <= 0:
        return None
    out_path = str(out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (cell_w * 2, cell_h * 2))
    for i in range(n):
        cells = []
        for frames, label in zip(frame_lists, labels):
            f = frames[i] if i < len(frames) else None
            cell = _fit_frame_bgr(f, cell_w, cell_h)
            cell = _put_label_bgr(cell, label)
            cells.append(cell)
        top = np.concatenate([cells[0], cells[1]], axis=1)
        bottom = np.concatenate([cells[2], cells[3]], axis=1)
        grid = np.concatenate([top, bottom], axis=0)
        writer.write(grid)
    writer.release()
    return out_path if os.path.exists(out_path) else None

# ======================= end LPIPS / comparison / source-selection helpers =======================

# ========================= poster-module UI helpers =========================
def _count_multikeyframe_source_items(source_path):
    """Count visible source frames from a folder/video path for status reporting only."""
    if not source_path:
        return 0
    source_path = str(source_path).strip()
    if not os.path.exists(source_path):
        return 0
    img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    if os.path.isdir(source_path):
        try:
            return len([p for p in Path(source_path).iterdir() if p.suffix.lower() in img_exts])
        except Exception:
            return 0
    if os.path.isfile(source_path):
        if Path(source_path).suffix.lower() in img_exts:
            return 1
        # It is likely a video. We do not decode it here to keep UI fast.
        return -1
    return 0


def _rename_motion_rows_for_poster_ui(rows):
    """Rename generic baseline/improved rows so the UI never implies full multi-keyframe GS fusion."""
    mapping = {
        "baseline": "single_keyframe_motion_baseline",
        "improved": "temporal_stabilized_motion",
        "baseline_motion": "single_keyframe_motion_baseline",
        "improved_motion": "temporal_stabilized_motion",
    }
    if rows is None:
        return rows
    try:
        # pandas DataFrame support
        if hasattr(rows, "copy") and hasattr(rows, "columns") and "variant" in rows.columns:
            rows = rows.copy()
            rows["variant"] = rows["variant"].map(lambda x: mapping.get(str(x), x))
            return rows
    except Exception:
        pass
    try:
        new_rows = []
        for r in rows:
            if isinstance(r, dict):
                r = dict(r)
                r["variant"] = mapping.get(str(r.get("variant")), r.get("variant"))
                new_rows.append(r)
            elif isinstance(r, (list, tuple)) and len(r) > 0:
                rr = list(r)
                rr[0] = mapping.get(str(rr[0]), rr[0])
                new_rows.append(rr)
            else:
                new_rows.append(r)
        return new_rows
    except Exception:
        return rows




# ========================= Level-1.5 diffusion restoration prototype helpers =========================

def _write_video_from_bgr_frames(frames, out_path, fps=30):
    """Write BGR uint8 frames to mp4; returns out_path or None."""
    if not frames:
        return None
    out_path = str(out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (w, h))
    for frame in frames:
        if frame is None:
            frame = np.full((h, w, 3), 255, dtype=np.uint8)
        if frame.shape[:2] != (h, w):
            frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
        writer.write(frame)
    writer.release()
    return out_path


def _unsharp_bgr(frame, amount=0.75, sigma=1.0):
    """Non-diffusion fallback preview. Useful only for debugging the restoration output path."""
    blur = cv2.GaussianBlur(frame, (0, 0), sigmaX=float(sigma), sigmaY=float(sigma))
    sharp = cv2.addWeighted(frame, 1.0 + float(amount), blur, -float(amount), 0)
    return np.clip(sharp, 0, 255).astype(np.uint8)


def _run_level15_diffusion_restoration(
    input_video_path,
    out_video_path,
    enable=False,
    method="diffusers_img2img",
    model_path="",
    prompt="realistic detailed human avatar, natural clothing details, sharp texture, temporally stable",
    negative_prompt="blur, artifacts, distorted face, distorted hands, flicker, inconsistent identity",
    strength=0.15,
    guidance_scale=2.0,
    num_inference_steps=8,
    max_frames=12,
    seed=42,
    fps=30,
):
    """Level-1.5 diffusion/restoration prototype.

    It takes the already rendered temporal-stabilized video and restores frames. When
    method='diffusers_img2img', a local Diffusers img2img model path is required.
    It does not train a diffusion model. It is a pretrained/local checkpoint hook.
    """
    result = {
        "requested": bool(enable),
        "active": False,
        "status": "disabled",
        "method": str(method),
        "model_path": str(model_path or ""),
        "output_path": None,
        "processed_frames": 0,
        "total_frames": 0,
        "note": "disabled by checkbox",
    }

    if not bool(enable):
        return result

    if not input_video_path or not os.path.exists(str(input_video_path)):
        result.update(status="missing_input_video", note=f"input video missing: {input_video_path}")
        return result

    frames = _read_bgr_video_frames_for_grid(input_video_path)
    result["total_frames"] = len(frames)
    if not frames:
        result.update(status="empty_input_video", note="no frames were read from rendered video")
        return result

    try:
        max_frames = int(max_frames) if max_frames is not None else len(frames)
        max_frames = max(1, min(max_frames, len(frames)))
    except Exception:
        max_frames = min(12, len(frames))

    method = str(method or "diffusers_img2img")
    restored = []

    if method == "unsharp_preview":
        for i, frame in enumerate(frames):
            if i < max_frames:
                restored.append(_unsharp_bgr(frame))
            else:
                restored.append(frame)
        _write_video_from_bgr_frames(restored, out_video_path, fps=fps)
        result.update(
            active=True,
            status="fallback_preview_active",
            output_path=str(out_video_path),
            processed_frames=max_frames,
            note="non-diffusion unsharp fallback; use only to verify restoration pipeline wiring",
        )
        return result

    if method != "diffusers_img2img":
        result.update(status="unsupported_method", note=f"unsupported method: {method}")
        return result

    model_path = str(model_path or "").strip()
    if not model_path:
        result.update(status="missing_model_path", note="diffusers_img2img requires a local pretrained model path")
        return result
    if not os.path.exists(model_path):
        result.update(status="model_path_not_found", note=f"model path not found: {model_path}")
        return result

    try:
        import torch as _torch
        from diffusers import StableDiffusionImg2ImgPipeline
        from PIL import Image as _PILImage

        device = "cuda" if _torch.cuda.is_available() else "cpu"
        dtype = _torch.float16 if device == "cuda" else _torch.float32
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            model_path,
            torch_dtype=dtype,
            safety_checker=None,
            requires_safety_checker=False,
            local_files_only=True,
        )
        pipe = pipe.to(device)
        try:
            pipe.enable_attention_slicing()
        except Exception:
            pass

        for i, frame in enumerate(frames):
            if i >= max_frames:
                restored.append(frame)
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil = _PILImage.fromarray(rgb)
            with _torch.no_grad():
                out = pipe(
                    generator=_torch.Generator(device=device).manual_seed(int(seed) + i),
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    image=pil,
                    strength=float(strength),
                    guidance_scale=float(guidance_scale),
                    num_inference_steps=int(num_inference_steps),
                ).images[0]
            out_bgr = cv2.cvtColor(np.array(out), cv2.COLOR_RGB2BGR)
            if out_bgr.shape[:2] != frame.shape[:2]:
                out_bgr = cv2.resize(out_bgr, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_AREA)
            restored.append(out_bgr)

        _write_video_from_bgr_frames(restored, out_video_path, fps=fps)
        result.update(
            active=True,
            status="prototype_active",
            output_path=str(out_video_path),
            processed_frames=max_frames,
            seed=int(seed),
            seed_policy="seed + frame_index, paired across configurations",
            effective_denoising_steps=int(num_inference_steps * strength),
            note=f"pretrained diffusers img2img restored {max_frames}/{len(frames)} frame(s); no training performed",
        )
        try:
            del pipe
            if _torch.cuda.is_available():
                _torch.cuda.empty_cache()
        except Exception:
            pass
        return result
    except Exception as e:
        result.update(status="runtime_error", note=str(e)[:240])
        print("[level1.5 diffusion] failed:", e)
        return result

# ======================= end Level-1.5 diffusion restoration prototype helpers =======================

def _build_poster_module_status(
    use_prefilter,
    use_pose_joint,
    use_trans_vel,
    enable_multi_source_manager,
    num_source_keyframes,
    resolved_source_path,
    enable_heldout_eval,
    resolved_heldout_gt_video_path,
    enable_lpips_eval=False,
    selected_source_frame=None,
    source_selection_note=None,
    comparison_video_path=None,
    diffusion_result=None,
):
    """Return module status rows for honest poster-aligned reporting."""
    source_count = _count_multikeyframe_source_items(resolved_source_path)
    if source_count == -1:
        source_note = f"video source path received: {resolved_source_path}"
    elif source_count > 0:
        source_note = f"{source_count} source frame(s) found; requested keyframes={num_source_keyframes}"
    elif resolved_source_path:
        source_note = f"source path received but no image count available: {resolved_source_path}"
    else:
        source_note = "no source frames path provided"

    temporal_active = bool(use_prefilter or use_pose_joint or use_trans_vel)
    temporal_note = []
    if use_prefilter:
        temporal_note.append("Gaussian prefilter")
    if use_pose_joint:
        temporal_note.append("pose joint-adaptive EMA")
    if use_trans_vel:
        temporal_note.append("translation velocity EMA")
    temporal_note = "; ".join(temporal_note) if temporal_note else "disabled"

    heldout_active = bool(enable_heldout_eval and resolved_heldout_gt_video_path and os.path.exists(str(resolved_heldout_gt_video_path)))

    source_selection_active = bool(enable_multi_source_manager and selected_source_frame)
    lpips_note = "enabled; may be slow and returns NaN if lpips/backbone weights are unavailable" if enable_lpips_eval else "disabled by checkbox"
    comparison_note = f"saved: {comparison_video_path}" if comparison_video_path else "not generated in this run"
    diffusion_result = diffusion_result or {}
    diffusion_requested = bool(diffusion_result.get("requested", False))
    diffusion_active = bool(diffusion_result.get("active", False))
    diffusion_status = "prototype active" if diffusion_active else ("requested but inactive" if diffusion_requested else "inactive")
    diffusion_note = diffusion_result.get("note") or "pretrained Level-1.5 restoration is available when enabled with a local model path"

    return [
        ["Single-keyframe LHM baseline", "active", "uses the source image path / uploaded image for standard LHM reconstruction"],
        ["Temporal motion stabilization", "active" if temporal_active else "inactive", temporal_note],
        ["Multi-keyframe source input manager", "active" if (enable_multi_source_manager and resolved_source_path) else "inactive", source_note],
        ["Multi-keyframe source selection", "prototype active" if source_selection_active else "inactive", source_selection_note or "best-keyframe selection is available when source frames are provided"],
        ["Canonical Gaussian fusion", "not active", "true GS-level fusion is not implemented yet; current prototype uses source-frame selection/management only"],
        ["Diffusion-based dynamic detail restoration", diffusion_status, diffusion_note + "; LPIPS evaluation status: " + lpips_note],
        ["Four-way comparison video", "active" if comparison_video_path else "inactive", comparison_note],
        ["Held-out view evaluation", "active" if heldout_active else "inactive/missing GT", f"GT path={resolved_heldout_gt_video_path}"],
        ["Camera-aligned held-out rendering", "not active", "current render is not yet forced to ZJU held-out camera K/R/T; metrics are weak held-out evaluation"],
    ]
# ======================= end poster-module UI helpers =======================


@torch.no_grad()
# Removed pose_estimator, face_detector, parsing_net from function signature
def demo_lhm(lhm, motion_generation, cfg): # Only lhm and cfg remain from original models

    motion_processing_dir = './train_data/users/motion_processing'
    os.makedirs(motion_processing_dir, exist_ok=True)


    @spaces.GPU(duration=100)
    def core_fn(image_upload, input_image_path, video_upload, target_motion_video_path, use_prefilter, use_pose_joint, use_trans_vel, pose_alpha, trans_alpha, enable_multi_keyframe_fusion, num_source_keyframes, source_frames_upload, project_source_path, enable_heldout_eval, heldout_gt_video_upload, heldout_gt_video_path, enable_lpips_eval, enable_diffusion_restoration, diffusion_method, diffusion_model_path, diffusion_strength, diffusion_steps, diffusion_max_frames, working_dir):
        resolved_image = resolve_image_input(
            image_upload=image_upload,
            path_text=input_image_path,
        )
        resolved_video_path = resolve_video_input(
            video_upload=video_upload,
            path_text=target_motion_video_path,
        )

        if resolved_image is None:
            raise gr.Error("Please upload an input image or provide a valid server image path.")
        if not resolved_video_path:
            raise gr.Error("Please upload a target motion video or provide a valid server video path.")

        image_raw = os.path.join(working_dir.name, "raw.png")
        with Image.fromarray(resolved_image) as img:
            img.save(image_raw)

        video_params = resolved_video_path

        print("[CORE] resolved input image from upload/path:", input_image_path if input_image_path else "upload/default")
        print("[CORE] resolved target motion video path:", video_params)

        resolved_source_path = resolve_source_input(
            source_frames_upload=source_frames_upload,
            source_video_upload=None,
            source_path_text=project_source_path,
        )
        resolved_heldout_gt_video_path = resolve_video_input(
            video_upload=heldout_gt_video_upload,
            path_text=heldout_gt_video_path,
        )

        print("[Project] multi-keyframe canonical GS fusion enabled:", bool(enable_multi_keyframe_fusion))
        print("[Project] requested number of source keyframes:", int(num_source_keyframes) if num_source_keyframes is not None else None)
        print("[Project] resolved source path:", resolved_source_path)
        selected_source_frame, source_selection_note = _select_best_source_frame(resolved_source_path, num_source_keyframes) if bool(enable_multi_keyframe_fusion) else (None, "multi-keyframe source manager disabled")
        print("[Project] selected best source frame prototype:", selected_source_frame)
        print("[Project] source-selection note:", source_selection_note)
        if selected_source_frame:
            resolved_image = resolve_image_input(path_text=selected_source_frame)
            Image.fromarray(resolved_image).save(image_raw)
        print("[Project] held-out view evaluation enabled:", bool(enable_heldout_eval))
        print("[Project] resolved held-out GT video path:", resolved_heldout_gt_video_path)

        # TODO for WK2:
        # - use resolved_source_path for keyframe selection / canonical GS fusion
        # - use resolved_heldout_gt_video_path for held-out PSNR/SSIM/LPIPS evaluation
        
        
        base_vid = os.path.basename(video_params).split(".")[0]
        smplx_params_dir = os.path.join("./train_data/motion_video/", base_vid, "smplx_params")

        baseline_smplx_dir = None
        improved_smplx_dir = None

        motion_processing_dir = "./train_data/users/motion_processing"
        video_hash = get_video_hash(video_params)
        output_path = os.path.join(motion_processing_dir, video_hash)

        # If smplx params already exist in repository, use them directly
        if os.path.exists(smplx_params_dir):
            baseline_smplx_dir = smplx_params_dir
            # derive improved variant from existing repo baseline (avoid double-gaussian)
            combo_dir = os.path.join(motion_processing_dir, video_hash + "_ours_from_repo")
            try:
                os.makedirs(combo_dir, exist_ok=True)
                make_variant_from_raw(
                    smplx_params_dir,
                    combo_dir,
                    fps=30,
                    compose_cfg={
                        "use_prefilter": False,
                        "use_pose_joint": bool(use_pose_joint),
                        "use_trans_vel": bool(use_trans_vel),
                        "joint_alpha": float(pose_alpha),
                        "vel_alpha_trans": float(trans_alpha),
                    },
                )
                improved_smplx_dir = combo_dir
            except Exception as e:
                print(f"Failed to derive improved variant from repo baseline: {e}")
                improved_smplx_dir = smplx_params_dir
        else:
            os.makedirs(motion_processing_dir, exist_ok=True)
            if not os.path.isdir(output_path + "_raw/smplx_params"):
                print("Loading Video2MotionPipeline (raw output)...")
                device = avaliable_device()
                motion_generation = Video2MotionPipeline(
                    "./pretrained_models/human_model_files",
                    fitting_steps=[30, 50],
                    device=device,
                    kp_mode="vitpose",
                    visualize=False,
                    pad_ratio=0.2,
                    fov=60,
                    smooth_mode="none",
                )
                print("Generating raw (unsmoothed) motion parameters (single V2M run)...")
                raw_out = motion_generation(video_params, output_path + "_raw", is_file_only=True, smooth_mode="none")
                raw_smplx_dir = raw_out
                torch.cuda.empty_cache()

                # derive variants on CPU using postprocess utilities
                fps = 30
                baseline_dir = output_path + "_baseline"
                pose_only_dir = output_path + "_pose_only"
                trans_only_dir = output_path + "_trans_only"
                combo_dir = output_path + "_ours_full"

                # baseline: gaussian prefilter only (smplx_smooth_compose applies gaussian by default)
                make_variant_from_raw(raw_smplx_dir, baseline_dir, fps=fps, compose_cfg={"use_prefilter": True})

                # pose-only ablation: joint-adaptive EMA on poses
                make_variant_from_raw(
                    raw_smplx_dir,
                    pose_only_dir,
                    fps=fps,
                    compose_cfg={"use_prefilter": True, "use_pose_joint": True, "use_trans_vel": False, "use_ema": False, "joint_alpha": float(pose_alpha)},
                )

                # trans-only ablation: velocity EMA on translation
                make_variant_from_raw(
                    raw_smplx_dir,
                    trans_only_dir,
                    fps=fps,
                    compose_cfg={"use_prefilter": True, "use_pose_joint": False, "use_trans_vel": True, "use_ema": False, "vel_alpha_trans": float(trans_alpha)},
                )

                # combo: composed ours_full
                make_variant_from_raw(
                    raw_smplx_dir,
                    combo_dir,
                    fps=fps,
                    compose_cfg={
                        "use_prefilter": bool(use_prefilter),
                        "use_pose_joint": bool(use_pose_joint),
                        "use_trans_vel": bool(use_trans_vel),
                        "use_ema": False,
                        "joint_alpha": float(pose_alpha),
                        "vel_alpha_trans": float(trans_alpha),
                    },
                )

                baseline_smplx_dir = baseline_dir
                improved_smplx_dir = combo_dir
            else:
                # if already processed earlier, use cached variants
                baseline_smplx_dir = os.path.join(output_path + "_baseline")
                improved_smplx_dir = os.path.join(output_path + "_ours_full")

        if os.environ.get("LHM_ASSET_DIR"):
            import json as _json
            os.makedirs(os.environ["LHM_ASSET_DIR"], exist_ok=True)
            with open(os.path.join(os.environ["LHM_ASSET_DIR"], "motion_directories.json"), "w") as _f:
                _json.dump({"baseline": baseline_smplx_dir, "stabilized": improved_smplx_dir,
                            "raw": output_path + "_raw/smplx_params"}, _f, indent=2)
        dump_video_path = os.path.join(working_dir.name, "output.mp4")
        dump_image_path = os.path.join(working_dir.name, "output.png")

        # prepare dump paths
        omit_prefix = os.path.dirname(image_raw)
        image_name = os.path.basename(image_raw)
        uid = image_name.split(".")[0]
        subdir_path = os.path.dirname(image_raw).replace(omit_prefix, "")
        subdir_path = (
            subdir_path[1:] if subdir_path.startswith("/") else subdir_path
        )
        print("subdir_path and uid:", subdir_path, uid)

        # pick dirs for baseline / improved (prefer generated dirs)
        motion_seqs_dir_baseline = baseline_smplx_dir if baseline_smplx_dir is not None else smplx_params_dir
        motion_seqs_dir_improved = improved_smplx_dir if improved_smplx_dir is not None else smplx_params_dir

        # Compute metrics: only two rows (baseline vs improved combo)
        metrics_df = pd.DataFrame()
        try:
            raw_candidate = raw_smplx_dir if 'raw_smplx_dir' in locals() else os.path.join(output_path + "_raw")
            base_dir = baseline_smplx_dir
            combo_dir = improved_smplx_dir

            if os.path.exists(raw_candidate):
                # Preferred path: use raw as reference
                _, raw_meta, raw_poses, raw_betas, raw_trans = load_smplx_dir(raw_candidate, device="cpu")
                print(f"[metrics] raw_candidate={raw_candidate}, frames={raw_poses.shape[0]}, poses.shape={raw_poses.shape}, transl.shape={raw_trans.shape}")

                def metrics_for(dirp):
                    files, meta, p_poses, p_betas, p_trans = load_smplx_dir(dirp, device="cpu")
                    print(f"[metrics] computing for {dirp}: files={len(files)}, poses.shape={p_poses.shape}, transl.shape={p_trans.shape}")
                    if p_poses.shape[0] < 3:
                        print(f"[metrics] WARNING: {dirp} has <3 frames ({p_poses.shape[0]}). jitter metrics will be zero.")
                    print("[metrics] sample transl[:3]", p_trans[:3].cpu().numpy() if isinstance(p_trans, torch.Tensor) else p_trans[:3])
                    # self metrics (intrinsic)
                    selfm = compute_self_metrics(p_poses, p_trans)
                    # deviation vs raw reference
                    devm = compute_metrics_vs(raw_poses, raw_trans, p_poses, p_trans)
                    combined = {"jitter_pose": float(selfm["jitter_pose"]), "jitter_trans": float(selfm["jitter_trans"]), "drift_trans": float(selfm["drift_trans"]), "deviation_pose": float(devm["deviation_pose"]), "deviation_trans": float(devm["deviation_trans"])}
                    print(f"[metrics] self for {dirp}: {selfm}, dev vs raw: {devm}")
                    return combined

                if base_dir is None or not os.path.exists(base_dir):
                    print(f"Baseline dir missing: {base_dir}")
                if combo_dir is None or not os.path.exists(combo_dir):
                    print(f"Combo dir missing: {combo_dir}")

                m_base = metrics_for(base_dir) if base_dir and os.path.exists(base_dir) else {k: 0.0 for k in ["jitter_pose","jitter_trans","drift_trans","deviation_pose","deviation_trans"]}
                m_impr = metrics_for(combo_dir) if combo_dir and os.path.exists(combo_dir) else {k: 0.0 for k in ["jitter_pose","jitter_trans","drift_trans","deviation_pose","deviation_trans"]}

                rows = [
                    {"variant": "baseline", **m_base},
                    {"variant": "improved", **m_impr},
                ]
            else:
                # Fallback: raw not available. Prefer using `smplx_params_dir` as reference (方案 A).
                print(f"Raw smplx not found: {raw_candidate}. Trying smplx_params_dir as reference for metrics (方案 A) if available.")
                ref_used = None
                if 'smplx_params_dir' in locals() and smplx_params_dir and os.path.exists(smplx_params_dir):
                    # Use smplx_params_dir as the raw reference
                    files_ref, meta_ref, raw_poses, raw_betas, raw_trans = load_smplx_dir(smplx_params_dir, device="cpu")
                    ref_used = smplx_params_dir
                    print(f"[metrics] using smplx_params_dir as reference: {smplx_params_dir}, frames={raw_poses.shape[0]}")

                    def metrics_for_ref(dirp):
                        files, meta, p_poses, p_betas, p_trans = load_smplx_dir(dirp, device="cpu")
                        print(f"[metrics] computing for {dirp}: files={len(files)}, poses.shape={p_poses.shape}, transl.shape={p_trans.shape}")
                        if p_poses.shape[0] < 3:
                            print(f"[metrics] WARNING: {dirp} has <3 frames ({p_poses.shape[0]}). jitter metrics will be zero.")
                        selfm = compute_self_metrics(p_poses, p_trans)
                        devm = compute_metrics_vs(raw_poses, raw_trans, p_poses, p_trans)
                        combined = {"jitter_pose": float(selfm["jitter_pose"]), "jitter_trans": float(selfm["jitter_trans"]), "drift_trans": float(selfm["drift_trans"]), "deviation_pose": float(devm["deviation_pose"]), "deviation_trans": float(devm["deviation_trans"])}
                        print(f"[metrics] self for {dirp}: {selfm}, dev vs ref({ref_used}): {devm}")
                        return combined

                    m_base = metrics_for_ref(base_dir) if base_dir and os.path.exists(base_dir) else {k: 0.0 for k in ["jitter_pose","jitter_trans","drift_trans","deviation_pose","deviation_trans"]}
                    m_impr = metrics_for_ref(combo_dir) if combo_dir and os.path.exists(combo_dir) else {k: 0.0 for k in ["jitter_pose","jitter_trans","drift_trans","deviation_pose","deviation_trans"]}

                    rows = [
                        {"variant": "baseline", **m_base},
                        {"variant": "improved", **m_impr},
                    ]
                elif base_dir and os.path.exists(base_dir):
                    # Last-resort fallback: use baseline itself as reference (legacy behavior)
                    print(f"Falling back to baseline-as-reference for metrics: {base_dir}")
                    files_b, meta_b, b_poses, b_betas, b_trans = load_smplx_dir(base_dir, device="cpu")
                    b_self = compute_self_metrics(b_poses, b_trans)
                    b_dev = {"deviation_pose": 0.0, "deviation_trans": 0.0}
                    m_base = {"jitter_pose": float(b_self["jitter_pose"]), "jitter_trans": float(b_self["jitter_trans"]), "drift_trans": float(b_self["drift_trans"]), **b_dev}

                    if combo_dir and os.path.exists(combo_dir):
                        files_i, meta_i, i_poses, i_betas, i_trans = load_smplx_dir(combo_dir, device="cpu")
                        i_self = compute_self_metrics(i_poses, i_trans)
                        i_dev = compute_metrics_vs(b_poses, b_trans, i_poses, i_trans)
                        m_impr = {"jitter_pose": float(i_self["jitter_pose"]), "jitter_trans": float(i_self["jitter_trans"]), "drift_trans": float(i_self["drift_trans"]), "deviation_pose": float(i_dev["deviation_pose"]), "deviation_trans": float(i_dev["deviation_trans"])}
                    else:
                        m_impr = {k: 0.0 for k in ["jitter_pose","jitter_trans","drift_trans","deviation_pose","deviation_trans"]}

                    rows = [
                        {"variant": "baseline", **m_base},
                        {"variant": "improved", **m_impr},
                    ]
                else:
                    print(f"Neither raw nor baseline smplx available for metrics: raw={raw_candidate}, baseline={base_dir}")
            # Build metrics_df if rows computed
            if "rows" in locals():
                metrics_df = pd.DataFrame(rows)
                # enforce column order
                metrics_df = metrics_df[["variant", "jitter_pose", "jitter_trans", "drift_trans", "deviation_pose", "deviation_trans"]]
                print("metrics_df type:", type(metrics_df), "shape:", metrics_df.shape)
                print(metrics_df.head())
        except Exception as e:
            print(f"Metrics computation error: {e}")

        # Always prepare a stable two-row metrics result (baseline, improved).
        cols = ["variant", "jitter_pose", "jitter_trans", "drift_trans", "deviation_pose", "deviation_trans"]
        # default rows with NaN to avoid Gradio showing undefined
        default_row = {"variant": "baseline", "jitter_pose": float("nan"), "jitter_trans": float("nan"), "drift_trans": float("nan"), "deviation_pose": float("nan"), "deviation_trans": float("nan")}
        default_row2 = {**default_row, "variant": "improved"}
        final_metrics = [default_row, default_row2]

        # If we successfully computed real rows, override defaults
        try:
            if "rows" in locals() and isinstance(rows, list) and len(rows) >= 2:
                # ensure keys are primitives
                cleaned = []
                for r in rows:
                    cr = {k: (float(v) if k != "variant" else str(v)) for k, v in r.items()}
                    cleaned.append(cr)
                final_metrics = cleaned
        except Exception:
            pass

        # Convert final_metrics (list-of-dicts) into list-of-lists matching the Dataframe headers
        final_metrics_for_gradio = []
        try:
            for r in final_metrics:
                final_metrics_for_gradio.append([
                    r.get("variant"),
                    float(r.get("jitter_pose", float("nan"))),
                    float(r.get("jitter_trans", float("nan"))),
                    float(r.get("drift_trans", float("nan"))),
                    float(r.get("deviation_pose", float("nan"))),
                    float(r.get("deviation_trans", float("nan"))),
                ])
        except Exception:
            # fallback: two NaN rows
            final_metrics_for_gradio = [
                ["baseline", float("nan"), float("nan"), float("nan"), float("nan"), float("nan")],
                ["improved", float("nan"), float("nan"), float("nan"), float("nan"), float("nan")],
            ]

        motion_name = os.path.dirname(
            motion_seqs_dir_baseline[:-1] if motion_seqs_dir_baseline[-1] == "/" else motion_seqs_dir_baseline
        )
        motion_name = os.path.basename(motion_name)

        dump_image_dir = os.path.dirname(dump_image_path)
        os.makedirs(dump_image_dir, exist_ok=True)

        # Print baseline/improved smplx dirs (fixed: previously referenced undefined variable `motion_seqs_dir`)
        print(image_raw, motion_seqs_dir_baseline, motion_seqs_dir_improved, dump_image_dir, dump_video_path)

        dump_tmp_dir = dump_image_dir


        source_size = cfg.source_size
        render_size = cfg.render_size
        render_fps = 30

        aspect_standard = 5.0 / 3
        motion_img_need_mask = cfg.get("motion_img_need_mask", False)  # False
        vis_motion = cfg.get("vis_motion", False)  # False

        # --- On-demand SAM2/rembg loading and execution ---
        parsing_mask = None
        parsingnet_ondemand = None
        try:
            # Try to load SAM2 on demand
            print("Attempting to load SAM2Seg...")
            parsingnet_ondemand = SAM2Seg()
            print("SAM2Seg loaded successfully.")
            with torch.no_grad():
                 parsing_out = parsingnet_ondemand(img_path=image_raw, bbox=None)
                 parsing_mask = (parsing_out.masks * 255).astype(np.uint8)
            print("Image parsed using SAM2.")
        except NameError: # Catches if SAM2Seg is not defined (import failed)
             print("SAM2Seg not available (NameError). Falling back to rembg.")
        except Exception as e: # Catches other potential errors during SAM2 loading/running
             print(f"Error during SAM2 processing: {e}. Falling back to rembg.")

        if parsing_mask is None: # If SAM2 failed or wasn't available
             try:
                 print("Using rembg for background removal...")
                 img_np = cv2.imread(image_raw)
                 remove_np = remove(img_np)
                 parsing_mask = remove_np[...,3]
                 print("Background removed using rembg.")
             except Exception as e:
                 print(f"Error during rembg processing: {e}. Cannot generate mask.")
                 # Handle error appropriately, maybe raise or return an error message
                 raise gr.Error("Failed to process image background removal.")

        # Unload SAM2 if it was loaded
        if parsingnet_ondemand is not None:
             print("Unloading SAM2Seg...")
             del parsingnet_ondemand
             torch.cuda.empty_cache()
             print("SAM2Seg unloaded.")
        # --- End of On-demand SAM2/rembg ---

        # Ensure parsing_mask is valid before proceeding
        if parsing_mask is None:
             raise gr.Error("Failed to generate parsing mask for the image.")

        # --- On-demand PoseEstimator loading and execution ---
        print("Loading PoseEstimator...")
        device = avaliable_device() # Ensure device is available
        pose_estimator_ondemand = PoseEstimator(
            "./pretrained_models/human_model_files/", device='cpu' # Load to CPU first potentially? Or directly to device? Let's use device.
        )
        pose_estimator_ondemand.to(device)
        pose_estimator_ondemand.device = device
        print("PoseEstimator loaded.")
        with torch.no_grad():
             shape_pose = pose_estimator_ondemand(image_raw)
        print("Pose estimated.")
        # Unload PoseEstimator
        shape_param_beta = shape_pose.beta # Store the result before deleting
        is_full_body = shape_pose.is_full_body # Store result
        msg = shape_pose.msg # Store result
        print("Unloading PoseEstimator...")
        del pose_estimator_ondemand
        del shape_pose # Delete intermediate variable too
        torch.cuda.empty_cache()
        print("PoseEstimator unloaded.")
        # --- End of On-demand PoseEstimator ---

        assert is_full_body, f"The input image is illegal, {msg}"

        # prepare reference image
        image, _, _ = infer_preprocess_image(
            image_raw,
            mask=parsing_mask,
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

        # --- On-demand VGGHeadDetector loading and execution ---
        src_head_rgb = None
        try:
            print("Loading VGGHeadDetector...")
            device = avaliable_device() # Ensure device is available
            facedetector_ondemand = VGGHeadDetector(
                "./pretrained_models/gagatracker/vgghead/vgg_heads_l.trcd",
                device=device,
            )
            # facedetector_ondemand.to(device) # Already done in constructor if device is cuda
            print("VGGHeadDetector loaded.")

            rgb_face = np.array(Image.open(image_raw))[...,:3]  # RGBA input
            rgb_face = torch.from_numpy(rgb_face).permute(2, 0, 1).to(device) # Move tensor to device
            bbox = facedetector_ondemand.detect_face(rgb_face)
            head_rgb = rgb_face[:, int(bbox[1]) : int(bbox[3]), int(bbox[0]) : int(bbox[2])]
            head_rgb = head_rgb.permute(1, 2, 0)
            src_head_rgb = head_rgb.cpu().numpy() # Get result as numpy array on CPU
            print("Face detected and head cropped.")

            # Unload VGGHeadDetector
            print("Unloading VGGHeadDetector...")
            del facedetector_ondemand
            del rgb_face # Delete intermediate tensor
            del head_rgb # Delete intermediate tensor
            torch.cuda.empty_cache()
            print("VGGHeadDetector unloaded.")

        except Exception as e:
            print(f"Face detection failed or w/o head input: {e}")
            src_head_rgb = np.zeros((112, 112, 3), dtype=np.uint8)
            # Ensure unloading even if detection fails but loading succeeded
            if 'facedetector_ondemand' in locals():
                 print("Unloading VGGHeadDetector after error...")
                 del facedetector_ondemand
                 torch.cuda.empty_cache()
                 print("VGGHeadDetector unloaded.")
        # --- End of On-demand VGGHeadDetector ---


        # resize to dino size (src_head_rgb is now defined)
        try:
            src_head_rgb = cv2.resize(
                src_head_rgb,
                dsize=(cfg.src_head_size, cfg.src_head_size),
                interpolation=cv2.INTER_AREA,
            )  # resize to dino size
        except:
            src_head_rgb = np.zeros(
                (cfg.src_head_size, cfg.src_head_size, 3), dtype=np.uint8
            )

        src_head_rgb = (
            torch.from_numpy(src_head_rgb / 255.0).float().permute(2, 0, 1).unsqueeze(0)
        )  # [1, 3, H, W]

        save_ref_img_path = os.path.join(
            dump_tmp_dir, "output.png"
        )
        vis_ref_img = (image[0].permute(1, 2, 0).cpu().detach().numpy() * 255).astype(
            np.uint8
        )
        Image.fromarray(vis_ref_img).save(save_ref_img_path)

        # read motion seq
        motion_name = os.path.dirname(
            motion_seqs_dir_baseline[:-1] if motion_seqs_dir_baseline[-1] == "/" else motion_seqs_dir_baseline
        )
        motion_name = os.path.basename(motion_name)

        motion_seq = prepare_motion_seqs(
            motion_seqs_dir_baseline,
            None,
            save_root=dump_tmp_dir,
            fps=30,
            bg_color=1.0,
            aspect_standard=aspect_standard,
            enlarge_ratio=[1.0, 1, 0],
            render_image_res=render_size,
            multiply=16,
            need_mask=motion_img_need_mask,
            vis_motion=vis_motion,
            motion_size=3000,
        )

        camera_size = len(motion_seq["motion_seqs"])
        # shape_param = shape_pose.beta # Use the stored value
        shape_param = shape_param_beta
        shared_betas = os.environ.get("LHM_SHARED_BETAS")
        if shared_betas:
            import json as _json
            if os.path.exists(shared_betas):
                shape_param = _json.load(open(shared_betas))
            else:
                os.makedirs(os.path.dirname(shared_betas), exist_ok=True)
                _json.dump(np.asarray(shape_param).tolist(), open(shared_betas, "w"))

        device = avaliable_device() # Use the helper function
        dtype = torch.float32
        shape_param = torch.tensor(shape_param, dtype=dtype).unsqueeze(0)

        lhm.to(dtype)

        smplx_params = motion_seq['smplx_params']
        smplx_params['betas'] = shape_param.to(device)

        gs_model_list, query_points, transform_mat_neutral_pose = lhm.infer_single_view(
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

        asset_dir = os.environ.get("LHM_ASSET_DIR")
        if asset_dir:
            from research_assets import save_bundle
            save_bundle(os.path.join(asset_dir, "canonical.pt"), gs_model_list,
                        query_points, transform_mat_neutral_pose, shape_param,
                        motion_seq, {"source_image": selected_source_frame or input_image_path,
                                     "motion_video": video_params,
                                     "camera_calibration": "LHM default camera; not ZJU calibrated"})

        # rendering !!!!
        start_time = time.time()

        batch_list = [] 

        batch_size = 5  # avoid memeory out!

        for batch_i in range(0, camera_size, batch_size):
            with torch.no_grad():
                # TODO check device and dtype
                # dict_keys(['comp_rgb', 'comp_rgb_bg', 'comp_mask', 'comp_depth', '3dgs'])

                print(f"batch: {batch_i}, total: {camera_size //batch_size +1} ")

                keys = [
                    "root_pose",
                    "body_pose",
                    "jaw_pose",
                    "leye_pose",
                    "reye_pose",
                    "lhand_pose",
                    "rhand_pose",
                    "trans",
                    "focal",
                    "princpt",
                    "img_size_wh",
                    "expr",
                ]


                batch_smplx_params = dict()
                batch_smplx_params["betas"] = shape_param.to(device)
                batch_smplx_params['transform_mat_neutral_pose'] = transform_mat_neutral_pose
                for key in keys:
                    batch_smplx_params[key] = motion_seq["smplx_params"][key][
                        :, batch_i : batch_i + batch_size
                    ].to(device)

                # def animation_infer(self, gs_model_list, query_points, smplx_params, render_c2ws, render_intrs, render_bg_colors, render_h, render_w):
                res = lhm.animation_infer(gs_model_list, query_points, batch_smplx_params,
                    render_c2ws=motion_seq["render_c2ws"][
                        :, batch_i : batch_i + batch_size
                    ].to(device),
                    render_intrs=motion_seq["render_intrs"][
                        :, batch_i : batch_i + batch_size
                    ].to(device),
                    render_bg_colors=motion_seq["render_bg_colors"][
                        :, batch_i : batch_i + batch_size
                    ].to(device),
                    )

            comp_rgb = res["comp_rgb"] # [Nv, H, W, 3], 0-1
            comp_mask = res["comp_mask"] # [Nv, H, W, 3], 0-1
            comp_mask[comp_mask < 0.5] = 0.0

            batch_rgb = comp_rgb * comp_mask + (1 - comp_mask) * 1
            batch_rgb = (batch_rgb.clamp(0,1) * 255).to(torch.uint8).detach().cpu().numpy()
            batch_list.append(batch_rgb)

            del res
            torch.cuda.empty_cache()
        
        rgb = np.concatenate(batch_list, axis=0)
        print(f"time elapsed: {time.time() - start_time}")

        if vis_motion:
            # print(rgb.shape, motion_seq["vis_motion_render"].shape)

            vis_ref_img = np.tile(
                cv2.resize(vis_ref_img, (rgb[0].shape[1], rgb[0].shape[0]))[
                    None, :, :, :
                ],
                (rgb.shape[0], 1, 1, 1),
            )
            rgb = np.concatenate(
                [rgb, motion_seq["vis_motion_render"], vis_ref_img], axis=2
            )

        os.makedirs(os.path.dirname(dump_video_path), exist_ok=True)

        images_to_video(
            rgb,
            output_path=dump_video_path,
            fps=render_fps,
            gradio_codec=False,
            verbose=True,
        )

        # --- render improved version ---
        dump_video_path_improved = os.path.join(working_dir.name, "output_improved.mp4")

        motion_seq2 = prepare_motion_seqs(
            motion_seqs_dir_improved,
            None,
            save_root=dump_tmp_dir,
            fps=30,
            bg_color=1.0,
            aspect_standard=aspect_standard,
            enlarge_ratio=[1.0, 1, 0],
            render_image_res=render_size,
            multiply=16,
            need_mask=motion_img_need_mask,
            vis_motion=vis_motion,
            motion_size=3000,
        )

        camera_size2 = len(motion_seq2["motion_seqs"])

        batch_list2 = []
        for batch_i in range(0, camera_size2, batch_size):
            with torch.no_grad():
                print(f"[improved] batch: {batch_i}, total: {camera_size2 //batch_size +1} ")

                batch_smplx_params = dict()
                batch_smplx_params["betas"] = shape_param.to(device)
                batch_smplx_params['transform_mat_neutral_pose'] = transform_mat_neutral_pose
                for key in keys:
                    batch_smplx_params[key] = motion_seq2["smplx_params"][key][
                        :, batch_i : batch_i + batch_size
                    ].to(device)

                res = lhm.animation_infer(gs_model_list, query_points, batch_smplx_params,
                    render_c2ws=motion_seq2["render_c2ws"][
                        :, batch_i : batch_i + batch_size
                    ].to(device),
                    render_intrs=motion_seq2["render_intrs"][
                        :, batch_i : batch_i + batch_size
                    ].to(device),
                    render_bg_colors=motion_seq2["render_bg_colors"][
                        :, batch_i : batch_i + batch_size
                    ].to(device),
                    )

                comp_rgb = res["comp_rgb"]
                comp_mask = res["comp_mask"]
                comp_mask[comp_mask < 0.5] = 0.0

                batch_rgb = comp_rgb * comp_mask + (1 - comp_mask) * 1
                batch_rgb = (batch_rgb.clamp(0,1) * 255).to(torch.uint8).detach().cpu().numpy()
                batch_list2.append(batch_rgb)

                del res
                torch.cuda.empty_cache()

        rgb2 = np.concatenate(batch_list2, axis=0)

        if vis_motion:
            vis_ref_img = np.tile(
                cv2.resize(vis_ref_img, (rgb2[0].shape[1], rgb2[0].shape[0]))[
                    None, :, :, :
                ],
                (rgb2.shape[0], 1, 1, 1),
            )
            rgb2 = np.concatenate(
                [rgb2, motion_seq2["vis_motion_render"], vis_ref_img], axis=2
            )

        images_to_video(
            rgb2,
            output_path=dump_video_path_improved,
            fps=render_fps,
            gradio_codec=False,
            verbose=True,
        )

        # Level-1.5 diffusion/detail restoration prototype. This uses the temporal-stabilized render as input.
        dump_video_path_diffusion = None
        diffusion_result = {"requested": bool(enable_diffusion_restoration), "active": False, "status": "disabled", "note": "disabled by checkbox"}
        try:
            results_root_for_diff = os.environ.get("LHM_PROJECT_RESULTS_ROOT") or os.environ.get("LHM_URECA_RESULTS_ROOT") or "/autodl-fs/data/lhm_shared/poster_results"
            diffusion_out = os.path.join(results_root_for_diff, "zju_313_wk2", "final", "level15_diffusion_restored.mp4")
            diffusion_result = _run_level15_diffusion_restoration(
                input_video_path=dump_video_path_improved,
                out_video_path=diffusion_out,
                enable=bool(enable_diffusion_restoration),
                method=diffusion_method,
                model_path=diffusion_model_path,
                strength=float(diffusion_strength),
                num_inference_steps=int(diffusion_steps),
                max_frames=int(diffusion_max_frames),
                fps=render_fps,
            )
            dump_video_path_diffusion = diffusion_result.get("output_path") if diffusion_result.get("active") else None
            print("[level1.5 diffusion result]", diffusion_result)
        except Exception as e:
            import traceback
            traceback.print_exc()
            diffusion_result = {"requested": bool(enable_diffusion_restoration), "active": False, "status": "error", "note": str(e)[:240]}
            dump_video_path_diffusion = None

        # Held-out evaluation: compute immediately after rendering and return rows to Gradio.
        # This block is intentionally robust: if something is missing, the table shows WHY instead of only "disabled_or_missing".
        heldout_metrics_for_gradio = [_heldout_status_row("disabled_by_checkbox")]
        if bool(enable_heldout_eval):
            try:
                inferred_mask_dir = None
                gt_for_metrics = _as_existing_path_for_metrics(resolved_heldout_gt_video_path)
                if gt_for_metrics:
                    gt_parent = os.path.dirname(str(gt_for_metrics))
                    candidate_mask_dir = os.path.join(gt_parent, "heldout_view_B_masks")
                    if os.path.isdir(candidate_mask_dir):
                        inferred_mask_dir = candidate_mask_dir

                heldout_metrics_for_gradio = compute_heldout_rows_safe(
                    dump_video_path,
                    dump_video_path_improved,
                    gt_for_metrics,
                    inferred_mask_dir,
                    enable_lpips=enable_lpips_eval,
                )
                if dump_video_path_diffusion and os.path.exists(str(dump_video_path_diffusion)):
                    heldout_metrics_for_gradio.append(
                        compute_heldout_video_metrics(
                            dump_video_path_diffusion,
                            gt_for_metrics,
                            inferred_mask_dir,
                            variant="level15_diffusion_restored_vs_heldout",
                            enable_lpips=enable_lpips_eval,
                        )
                    )
                print("[heldout metrics returned to gradio]", heldout_metrics_for_gradio)
            except Exception as e:
                import traceback
                traceback.print_exc()
                heldout_metrics_for_gradio = [_heldout_status_row("heldout_eval_error:" + str(e)[:80])]
        else:
            print("[heldout metrics] skipped because checkbox is disabled")

        # Four-way comparison video for poster/weekly visualization.
        comparison_video_path = None
        try:
            gt_for_comparison = _as_existing_path_for_metrics(resolved_heldout_gt_video_path)
            results_root = os.environ.get("LHM_PROJECT_RESULTS_ROOT") or os.environ.get("LHM_URECA_RESULTS_ROOT") or "/autodl-fs/data/lhm_shared/poster_results"
            comparison_out = os.path.join(results_root, "zju_313_wk2", "final", "four_way_source_gt_baseline_stabilized.mp4")
            comparison_video_path = make_four_way_comparison_video(
                source_video=resolved_video_path,
                gt_video=gt_for_comparison,
                baseline_video=dump_video_path,
                improved_video=dump_video_path_improved,
                out_path=comparison_out,
                fps=30,
            )
            print("[comparison video]", comparison_video_path)
        except Exception as e:
            print("[comparison video] failed:", e)
            comparison_video_path = None

        final_metrics_for_gradio = _rename_motion_rows_for_poster_ui(final_metrics_for_gradio)
        module_status_for_gradio = _build_poster_module_status(
            use_prefilter=use_prefilter,
            use_pose_joint=use_pose_joint,
            use_trans_vel=use_trans_vel,
            enable_multi_source_manager=enable_multi_keyframe_fusion,
            num_source_keyframes=num_source_keyframes,
            resolved_source_path=resolved_source_path,
            enable_heldout_eval=enable_heldout_eval,
            resolved_heldout_gt_video_path=resolved_heldout_gt_video_path,
            enable_lpips_eval=enable_lpips_eval,
            selected_source_frame=selected_source_frame,
            source_selection_note=source_selection_note,
            comparison_video_path=comparison_video_path,
            diffusion_result=diffusion_result,
        )
        return dump_image_path, dump_video_path, dump_video_path_improved, dump_video_path_diffusion, comparison_video_path, final_metrics_for_gradio, heldout_metrics_for_gradio, module_status_for_gradio

    if os.environ.get("LHM_HEADLESS") == "1":
        return core_fn

    _TITLE = "LHM: Large Animatable Human Model"

    _DESCRIPTION = '''
        <strong>Reconstruct a human avatar in 0.2 seconds with A100!</strong>
    '''

    with gr.Blocks(analytics_enabled=False) as demo:

        logo_url = "./assets/LHM_logo_parsing.png"
        logo_base64 = get_image_base64(logo_url)
        gr.HTML(
            f"""
            <div style="max-width: 1180px; margin: 0 auto 10px auto; text-align: center;">
                <h1 style="color:#c4002f; font-size:36px; line-height:1.12; margin: 8px 0 10px 0; font-weight: 800;">
                    CCDS25096 - Enhanced 4D Gaussian Splatting with Diffusion-based Dynamic Detail Restoration for Scene Understanding
                </h1>
                <div style="font-size:15px; color:#333; margin: 4px 0 14px 0;">
                    LHM-based reconstruction and evaluation system using the pretrained <b>Large Animatable Human Model</b> as the avatar reconstruction backbone.
                </div>
                <div style="display:flex; justify-content:center; align-items:center; gap:14px; flex-wrap:wrap; margin-top:6px; padding:8px 10px; border-top:1px solid #eee; border-bottom:1px solid #eee;">
                    <div style="display:flex; align-items:center; gap:8px;">
                        <img src="data:image/png;base64,{logo_base64}" style="height:34px; object-fit:contain;" alt="LHM logo">
                        <span style="font-size:18px; font-weight:700; color:#111;">Large Animatable Human Model</span>
                    </div>
                    <div style="display:flex; align-items:center; gap:4px; flex-wrap:wrap; justify-content:center;">
                        <a title="Project" href="https://aigc3d.github.io/LHM/" target="_blank" rel="noopener noreferrer" style="display:inline-block;">
                            <img src="https://img.shields.io/badge/Project-LHM-555?logo=heart" alt="Project">
                        </a>
                        <a title="Website" href="https://lingtengqiu.github.io/LHM/" target="_blank" rel="noopener noreferrer" style="display:inline-block;">
                            <img src="https://www.obukhov.ai/img/badges/badge-website.svg" alt="Website">
                        </a>
                        <a title="Read" href="https://arxiv.org/abs/2503.10625" target="_blank" rel="noopener noreferrer" style="display:inline-block;">
                            <img src="https://img.shields.io/badge/Read-arXiv-777?logo=readthedocs" alt="Read">
                        </a>
                        <a title="Paper" href="https://arxiv.org/pdf/2503.10625" target="_blank" rel="noopener noreferrer" style="display:inline-block;">
                            <img src="https://img.shields.io/badge/Paper-PDF-b31b1b" alt="Paper">
                        </a>
                        <a title="GitHub" href="https://github.com/aigc3d/LHM" target="_blank" rel="noopener noreferrer" style="display:inline-block;">
                            <img src="https://img.shields.io/github/stars/aigc3d/LHM?label=GitHub%20%E2%98%85&logo=github&color=C8C" alt="GitHub stars">
                        </a>
                        <a title="YouTube" href="https://www.youtube.com/watch?v=tivEpz_yiEo" target="_blank" rel="noopener noreferrer" style="display:inline-block;">
                            <img src="https://img.shields.io/badge/YouTube-QiuLingteng-red?logo=youtube" alt="YouTube">
                        </a>
                    </div>
                </div>
                <div style="font-size:12px; color:#666; margin-top:6px;">
                    Backbone attribution: LHM original project links are preserved above. The main interface below is aligned with the CCDS25096 poster pipeline.
                </div>
            </div>
            """
        )

        gr.HTML(
            """<p><h4 style="color: red;"> Notes: Please input human image. Currently, it only supports motion video input with a maximum of 500 frames. </h4></p>"""
            """<p><h4 style="color: red;"> For LHM-500M, we require at least 24 GB of GPU memory; for LHM-1B, we require at least 32 GB of memory.</h4></p>"""
        )

        # DISPLAY
        with gr.Row():

            with gr.Column(variant='panel', scale=1):
                with gr.Tabs(elem_id="openlrm_input_image"):
                    with gr.TabItem('Input Image'):
                        with gr.Row():
                            input_image = gr.Image(
                                label="Source Image — Drag / Click Upload",
                                value="./train_data/example_imgs/00000000_joker_2.jpg",
                                image_mode="RGBA",
                                height=480,
                                width=270,
                                sources="upload",
                                type="numpy",
                                elem_id="content_image",
                            )
                        input_image_path = gr.Textbox(
                            label="OR server source image path",
                            placeholder="/autodl-fs/data/lhm_shared/poster_inputs/zju_313_wk2/source_view_A_frames/keyframe_02_frame_000030.jpg",
                        )
                # EXAMPLES
                examples = os.listdir('./train_data/example_imgs/')
                with gr.Row():
                    examples = [os.path.join('./train_data/example_imgs/', example) for example in examples]
                    gr.Examples(
                        examples=examples,
                        inputs=[input_image], 
                        examples_per_page=9,
                    )

            examples_video =  os.listdir('./train_data/motion_video/')
            examples =[os.path.join('./train_data/motion_video/', example, 'samurai_visualize.mp4') for example in examples_video] 

            examples = sorted(examples)
            new_examples = []
            for example in examples:
                video_basename = os.path.basename(os.path.dirname(example))
                input_video = os.path.join(os.path.dirname(example), video_basename+'.mp4')
                if not os.path.exists(input_video):
                    shutil.copyfile(example, input_video)
                new_examples.append(input_video)

            with gr.Column(variant='panel', scale=1):
                with gr.Tabs(elem_id="openlrm_input_video"):
                    with gr.TabItem('Target Motion'):
                        with gr.Row():
                            video_input = gr.Video(
                                label="Target Motion Video — Drag / Click Upload",
                                sources='upload',
                                height=480,
                                width=270,
                                interactive=True,
                                value=new_examples[3],
                            )
                        target_motion_video_path = gr.Textbox(
                            label="OR server target motion video path",
                            placeholder="/autodl-fs/data/lhm_shared/poster_inputs/zju_313_wk2/source_view_A.mp4",
                        )
                with gr.Row():
                    gr.Examples(
                        examples=new_examples,
                        inputs=[video_input],
                        examples_per_page=9,
                    )

            with gr.Column(variant='panel', scale=1):
                with gr.Tabs(elem_id="openlrm_processed_image"):
                    with gr.TabItem('Processed Image'):
                        with gr.Row():
                            processed_image = gr.Image(label="Processed Image", image_mode="RGBA", type="filepath", elem_id="processed_image", height=480, width=270, interactive=False)

            with gr.Column(variant='panel', scale=1):
                with gr.Tabs(elem_id="openlrm_render_video"):
                    with gr.TabItem('Rendered Outputs - Baseline vs Temporal Stabilized'):
                        with gr.Row():
                            output_video_baseline = gr.Video(label="Single-keyframe LHM baseline render", format="mp4", height=480, width=270, autoplay=False)
                            output_video_improved = gr.Video(label="Temporal-stabilized render", format="mp4", height=480, width=270, autoplay=False)
                            output_video_diffusion = gr.Video(label="Level-1.5 diffusion-restored render", format="mp4", height=480, width=270, autoplay=False)
                    with gr.TabItem('Poster Comparison'):
                        output_video_comparison = gr.Video(label="Four-way comparison: source | held-out GT | baseline | temporal-stabilized", format="mp4", height=520, width=520, autoplay=False)

        # SETTING
        gr.Markdown("""
### Poster Module Controls
These controls separate implemented modules from poster-direction modules. The current active path supports single-keyframe LHM baseline rendering, temporal motion stabilization, multi-keyframe source input management, a best-source-frame selection prototype, optional LPIPS evaluation, four-way comparison video generation, a Level-1.5 pretrained diffusion/detail restoration prototype, and held-out video evaluation. True GS-level canonical fusion and camera-aligned rendering remain explicitly marked until their backend is implemented.
""")
        with gr.Row():
            with gr.Column(variant='panel', scale=1):
                use_prefilter_checkbox = gr.Checkbox(label="Gaussian prefilter (motion preprocessing)", value=True)
                use_pose_joint_checkbox = gr.Checkbox(label="Pose joint-adaptive EMA", value=True)
                use_trans_vel_checkbox = gr.Checkbox(label="Translation velocity EMA", value=True)
                pose_alpha_slider = gr.Slider(label="Pose EMA alpha", minimum=0.0, maximum=1.0, value=0.25, step=0.01)
                trans_alpha_slider = gr.Slider(label="Trans EMA alpha", minimum=0.0, maximum=1.0, value=0.25, step=0.01)

                enable_multi_keyframe_checkbox = gr.Checkbox(
                    label="Multi-keyframe source input manager",
                    value=True,
                )
                num_source_keyframes_slider = gr.Slider(
                    label="Number of source frames to use",
                    minimum=1,
                    maximum=8,
                    value=5,
                    step=1,
                )
                source_frames_upload = gr.File(
                    label="Source Frames — Drag / Click Upload folder",
                    file_count="directory",
                    type="filepath",
                )
                project_source_path = gr.Textbox(
                    label="OR server source frames/video path",
                    placeholder="/autodl-fs/data/lhm_shared/poster_inputs/zju_313_wk2/source_view_A_frames",
                )

                enable_heldout_eval_checkbox = gr.Checkbox(
                    label="Held-out view evaluation",
                    value=True,
                )
                heldout_gt_video_upload = gr.Video(
                    label="Held-out GT Video — Drag / Click Upload",
                    sources="upload",
                    interactive=True,
                )
                heldout_gt_video_path = gr.Textbox(
                    label="OR server held-out GT video path",
                    placeholder="/autodl-fs/data/lhm_shared/poster_inputs/zju_313_wk2/heldout_view_B.mp4",
                )
                enable_lpips_checkbox = gr.Checkbox(
                    label="Enable LPIPS perceptual metric (optional, slower)",
                    value=False,
                )

                enable_diffusion_restoration_checkbox = gr.Checkbox(
                    label="Enable Level-1.5 diffusion/detail restoration prototype",
                    value=False,
                )
                diffusion_method_dropdown = gr.Dropdown(
                    label="Restoration method",
                    choices=["diffusers_img2img", "unsharp_preview"],
                    value="diffusers_img2img",
                )
                diffusion_model_path = gr.Textbox(
                    label="Local pretrained diffusion model path",
                    placeholder="/autodl-fs/data/lhm_shared/shared_assets/diffusion_models/stable-diffusion-img2img-local",
                )
                diffusion_strength_slider = gr.Slider(
                    label="Diffusion strength",
                    minimum=0.0,
                    maximum=0.6,
                    value=0.15,
                    step=0.01,
                )
                diffusion_steps_slider = gr.Slider(
                    label="Diffusion inference steps",
                    minimum=4,
                    maximum=30,
                    value=8,
                    step=1,
                )
                diffusion_max_frames_slider = gr.Slider(
                    label="Max restored frames for prototype",
                    minimum=1,
                    maximum=60,
                    value=12,
                    step=1,
                )

                submit = gr.Button('Generate', elem_id="openlrm_generate", variant='primary')


        working_dir = gr.State()
        # status and metrics tables — arrows indicate direction: ↓ lower is better, ↑ higher is better.
        module_status_table = gr.Dataframe(
            headers=["module", "status", "note"],
            label="Poster module status",
            interactive=False,
        )
        metrics_table = gr.Dataframe(headers=["variant", "jitter_pose ↓", "jitter_trans ↓", "drift_trans ↓", "deviation_pose ↓", "deviation_trans ↓"], label="Motion stabilization metrics: single-keyframe baseline vs temporal-stabilized", interactive=False)
        heldout_metrics_table = gr.Dataframe(headers=["variant", "PSNR full ↑", "SSIM full ↑", "PSNR fg crop ↑", "SSIM fg crop ↑", "temporal delta error ↓", "LPIPS ↓"], label="Held-out evaluation metrics: rendered video vs GT video", interactive=False)

        submit.click(
            fn=assert_core_inputs,
            inputs=[input_image, input_image_path, video_input, target_motion_video_path],
            queue=False,
        ).success(
            fn=prepare_working_dir,
            outputs=[working_dir],
            queue=False,
        ).success(
            fn=core_fn,
            inputs=[input_image, input_image_path, video_input, target_motion_video_path, use_prefilter_checkbox, use_pose_joint_checkbox, use_trans_vel_checkbox, pose_alpha_slider, trans_alpha_slider, enable_multi_keyframe_checkbox, num_source_keyframes_slider, source_frames_upload, project_source_path, enable_heldout_eval_checkbox, heldout_gt_video_upload, heldout_gt_video_path, enable_lpips_checkbox, enable_diffusion_restoration_checkbox, diffusion_method_dropdown, diffusion_model_path, diffusion_strength_slider, diffusion_steps_slider, diffusion_max_frames_slider, working_dir],
            outputs=[processed_image, output_video_baseline, output_video_improved, output_video_diffusion, output_video_comparison, metrics_table, heldout_metrics_table, module_status_table],
        )

        demo.queue()
        demo.launch(server_name="0.0.0.0")

def get_parse():
    import argparse
    parser = argparse.ArgumentParser(description='LHM-gradio: Large Animatable Human Model')
    parser.add_argument('--model_name', default='LHM-1B-HF', type=str, choices=['LHM-500M', 'LHM-1B', 'LHM-500M-HF', 'LHM-1B-HF', "LHM-MINI"], help='Model name')
    args = parser.parse_args()
    return args


def launch_gradio_app():

    args = get_parse()
    model_name = args.model_name # Directly use the model name from args


    model_switcher = AutoModelSwitcher(MEMORY_MODEL_CARD, extra_memory=-2000)
    model_name =  model_switcher.query(model_name)

    print(f"Using specified model: {model_name}") # Add a print statement to confirm

    os.environ.update({
        "APP_ENABLED": "1",
        "APP_MODEL_NAME": model_name,  # choice from MODEL_CARD
        "APP_TYPE": "infer.human_lrm",
        "NUMBA_THREADING_LAYER": 'omp',
    })

    prior_check()


    download_geo_files()

    device= avaliable_device()


    accelerator = Accelerator()

    cfg, cfg_train = parse_configs()
    lhm = _build_model(cfg)
    lhm.to('cuda')

    demo_lhm(lhm, None, cfg) # Removed pose_estimator and facedetector arguments




if __name__ == '__main__':
    launch_gradio_app()

