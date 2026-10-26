import numpy as np
import os, json
import torch
from typing import List, Tuple, Dict, Optional
import torch.nn.functional as F
from pose_utils.rot6d import axis_angle_to_rotation_6d, rotation_6d_to_axis_angle


def get_gaussian_kernel_1d(kernel_size, sigma, device):
    x = torch.arange(kernel_size).float() - (kernel_size // 2)
    g = torch.exp(-((x**2) / (2 * sigma**2)))
    g /= g.sum()

    kernel_weight = g.view(1, 1, -1).to(device)

    return kernel_weight


def gaussian_filter_1d(data, kernel_size=3, sigma=1.0, weight=None):
    kernel_weight = (
        get_gaussian_kernel_1d(kernel_size, sigma, data.device)
        if weight is None
        else weight
    )
    data = F.pad(data, (kernel_size // 2, kernel_size // 2), mode="replicate")
    return F.conv1d(data, kernel_weight)


def exponential_smoothing(x, d_x, alpha=0.5):
    return d_x + alpha * (x - d_x)


@torch.no_grad()
def smplx_gs_smooth(poses, betas, transl, fps=30):
    poses = axis_angle_to_rotation_6d(poses)
    N, J, _ = poses.shape
    poses = (
        gaussian_filter_1d(
            poses.view(N, 1, -1).permute(2, 1, 0),
            kernel_size=9,
            sigma=1 * fps / 30,
        )
        .permute(2, 1, 0)
        .view(N, J, -1)
    )
    betas = (
        gaussian_filter_1d(
            betas.view(-1, 1, 10).permute(2, 1, 0),
            kernel_size=11,
            sigma=5.0 * fps / 30,
        )
        .permute(2, 1, 0)
        .view(-1, 10)
    )
    transl[1:-1] = (
        gaussian_filter_1d(
            transl.view(N, 1, -1).permute(2, 1, 0),
            kernel_size=9,
            sigma=1.0 * fps / 30,
        )
        .permute(2, 1, 0)
        .view(N, -1)[1:-1]
    )

    poses = rotation_6d_to_axis_angle(poses)
    return poses, betas, transl


@torch.no_grad()
def smplx_temporal_smooth(
    poses, betas, transl, fps=30, mode="gaussian", ema_alpha=0.8, **kwargs
):
    """Unified temporal smoothing for SMPL-X parameters.

    mode: "none" | "gaussian" | "gaussian_ema"
    ema_alpha: alpha for exponential moving average when using gaussian_ema
    Additional kwargs kept for future extensibility.
    """
    if mode is None or mode == "none":
        return poses, betas, transl

    # first apply gaussian smoothing (reusing existing implementation)
    poses_g, betas_g, transl_g = smplx_gs_smooth(poses, betas, transl, fps=fps)

    if mode == "gaussian":
        return poses_g, betas_g, transl_g

    if mode == "gaussian_ema":
        # apply EMA on top of gaussian results
        def _ema(tensor, alpha=ema_alpha):
            out = torch.empty_like(tensor)
            out[0] = tensor[0]
            for i in range(1, tensor.shape[0]):
                out[i] = alpha * tensor[i] + (1 - alpha) * out[i - 1]
            return out

        poses_ge = _ema(poses_g, alpha=ema_alpha)
        betas_ge = _ema(betas_g, alpha=ema_alpha)
        transl_ge = _ema(transl_g, alpha=ema_alpha)

        return poses_ge, betas_ge, transl_ge

    # fallback: return gaussian
    return poses_g, betas_g, transl_g


class OneEuroFilter:
    # param setting:
    #   realtime v2m: min_cutoff=1.0, beta=1.5
    #   motionshop 2d keypoint: min_cutoff=1.7, beta=0.3
    def __init__(
        self, min_cutoff=1.0, beta=0.0, sampling_rate=30, d_cutoff=1.0, device="cuda"
    ):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.sampling_rate = sampling_rate
        self.x_prev = None
        self.dx_prev = None
        self.d_cutoff = d_cutoff
        self.pi = torch.tensor(torch.pi, device=device)

    def smoothing_factor(self, cutoff):

        r = 2 * self.pi * cutoff / self.sampling_rate
        return r / (1 + r)

    def filter(self, x):
        if self.x_prev is None:
            self.x_prev = x
            self.dx_prev = torch.zeros_like(x)
            return x

        a_d = self.smoothing_factor(self.d_cutoff)
        # 计算当前的速度
        dx = (x - self.x_prev) * self.sampling_rate

        dx_hat = exponential_smoothing(dx, self.dx_prev, a_d)

        cutoff = self.min_cutoff + self.beta * torch.abs(dx_hat)
        a = self.smoothing_factor(cutoff)

        x_hat = exponential_smoothing(x, self.x_prev, a)

        self.x_prev = x_hat
        self.dx_prev = dx_hat

        return x_hat


# ----------------------------
# Composite smoothing, IO and metrics utilities
# ----------------------------


@torch.no_grad()
def _second_diff(x: torch.Tensor) -> torch.Tensor:
    # x: [T, ...]
    if x.shape[0] < 3:
        return torch.zeros((0,) + x.shape[1:], device=x.device)
    return x[2:] - 2 * x[1:-1] + x[:-2]


@torch.no_grad()
def ema_smooth(x: torch.Tensor, alpha: float) -> torch.Tensor:
    if x.numel() == 0 or x.shape[0] <= 1:
        return x
    out = x.clone()
    for t in range(1, x.shape[0]):
        out[t] = (1 - alpha) * out[t - 1] + alpha * x[t]
    return out


@torch.no_grad()
def ema_joint_adaptive(x: torch.Tensor, alpha_per_joint: torch.Tensor) -> torch.Tensor:
    # x: [T, J, D]
    if x.numel() == 0 or x.shape[0] <= 1:
        return x
    y = x.clone()
    J = x.shape[1]
    D = x.shape[2]
    alpha_per_joint = alpha_per_joint.view(J, 1).to(x.device).to(x.dtype)
    for t in range(1, x.shape[0]):
        y[t] = y[t - 1] + alpha_per_joint[None, :, :] * (x[t] - y[t - 1])
    return y


@torch.no_grad()
def vel_ema_integrate(x: torch.Tensor, alpha: float) -> torch.Tensor:
    # x: [T, C]
    if x.numel() == 0 or x.shape[0] <= 1:
        return x
    v = x[1:] - x[:-1]
    v_s = ema_smooth(v, alpha=alpha)
    y = x.clone()
    for t in range(1, x.shape[0]):
        y[t] = y[t - 1] + v_s[t - 1]
    return y


@torch.no_grad()
def vel_ema_integrate(x: torch.Tensor, alpha: float, drift_correction: bool = True) -> torch.Tensor:
    # x: [T, C]
    if x.numel() == 0 or x.shape[0] <= 1:
        return x
    v = x[1:] - x[:-1]
    v_s = ema_smooth(v, alpha=alpha)
    y = x.clone()
    for t in range(1, x.shape[0]):
        y[t] = y[t - 1] + v_s[t - 1]
    if drift_correction:
        # Correct accumulated endpoint drift by distributing the endpoint error linearly
        delta = (x[-1] - y[-1]).to(x.dtype)
        T = x.shape[0]
        corr = torch.linspace(0.0, 1.0, steps=T, device=x.device, dtype=x.dtype).view(-1, 1) * delta.view(1, -1)
        y = y + corr
    return y


@torch.no_grad()
def smplx_smooth_compose(
    poses: torch.Tensor,
    betas: torch.Tensor,
    transl: torch.Tensor,
    fps: float = 30.0,
    use_prefilter: bool = True,
    use_ema: bool = False,
    use_pose_joint: bool = False,
    use_trans_vel: bool = False,
    ema_alpha: float = 0.25,
    alpha_root: float = 0.45,
    alpha_body: float = 0.30,
    alpha_hand: float = 0.18,
    joint_alpha: Optional[float] = None,
    vel_alpha_trans: float = 0.25,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # baseline: optionally apply gaussian prefilter (to reproduce repo baseline)
    if use_prefilter:
        poses_b, betas_b, transl_b = smplx_gs_smooth(poses, betas, transl, fps=fps)
    else:
        poses_b, betas_b, transl_b = poses, betas, transl

    # convert poses to 6D
    poses6_b = axis_angle_to_rotation_6d(poses_b)
    T, J, D = poses6_b.shape

    # pose branch
    if use_pose_joint:
        # allow a single `joint_alpha` scalar from UI to control joint-adaptive smoothing
        if joint_alpha is not None:
            a_body = float(joint_alpha)
            a_root = min(0.95, a_body * 2.0)
            a_hand = max(0.02, a_body * 0.7)
        else:
            a_root = alpha_root
            a_body = alpha_body
            a_hand = alpha_hand

        alpha_j = torch.full((J,), a_body, device=poses6_b.device, dtype=poses6_b.dtype)
        alpha_j[0] = a_root
        if J > 25:
            alpha_j[25:55] = a_hand
        poses6_o = ema_joint_adaptive(poses6_b, alpha_j)
    elif use_ema:
        poses6_o = ema_smooth(poses6_b.reshape(T, -1), alpha=ema_alpha).reshape(T, J, D)
    else:
        poses6_o = poses6_b

    # transl branch
    if use_trans_vel:
        transl_o = vel_ema_integrate(transl_b, alpha=vel_alpha_trans)
    elif use_ema:
        transl_o = ema_smooth(transl_b, alpha=ema_alpha)
    else:
        transl_o = transl_b

    # betas
    if use_ema:
        betas_o = ema_smooth(betas_b, alpha=ema_alpha)
    else:
        betas_o = betas_b

    poses_o = rotation_6d_to_axis_angle(poses6_o)
    return poses_o, betas_o, transl_o


def load_smplx_dir(dir_path: str, device: str = "cpu") -> Tuple[List[str], Dict, torch.Tensor, torch.Tensor, torch.Tensor]:
    files = sorted([f for f in os.listdir(dir_path) if f.endswith(".json")])
    assert len(files) > 0, f"No json found in {dir_path}"

    meta0 = None
    poses_list, betas_list, transl_list = [], [], []

    for fn in files:
        fp = os.path.join(dir_path, fn)
        with open(fp, "r") as f:
            d = json.load(f)

        if meta0 is None:
            meta0 = {k: d.get(k) for k in ["focal", "princpt", "img_size_wh", "pad_ratio"]}

        pose = np.zeros((55, 3), dtype=np.float32)
        pose[0] = np.array(d.get("root_pose", [0, 0, 0]), dtype=np.float32)
        pose[1:22] = np.array(d.get("body_pose", [[0, 0, 0]] * 21), dtype=np.float32)
        pose[22] = np.array(d.get("jaw_pose", [0, 0, 0]), dtype=np.float32)
        pose[23] = np.array(d.get("leye_pose", [0, 0, 0]), dtype=np.float32)
        pose[24] = np.array(d.get("reye_pose", [0, 0, 0]), dtype=np.float32)
        pose[25:40] = np.array(d.get("lhand_pose", [[0, 0, 0]] * 15), dtype=np.float32)
        pose[40:55] = np.array(d.get("rhand_pose", [[0, 0, 0]] * 15), dtype=np.float32)

        poses_list.append(pose)
        betas_list.append(np.array(d.get("betas", [0.0] * 10), dtype=np.float32))
        transl_list.append(np.array(d.get("trans", [0.0, 0.0, 0.0]), dtype=np.float32))

    poses = torch.from_numpy(np.stack(poses_list, axis=0)).to(device)
    betas = torch.from_numpy(np.stack(betas_list, axis=0)).to(device)
    transl = torch.from_numpy(np.stack(transl_list, axis=0)).to(device)
    return files, meta0, poses, betas, transl


def save_smplx_dir(dir_path: str, files: List[str], meta: Dict,
                   poses: torch.Tensor, betas: torch.Tensor, transl: torch.Tensor) -> None:
    os.makedirs(dir_path, exist_ok=True)
    poses_np = poses.detach().cpu().numpy()
    betas_np = betas.detach().cpu().numpy()
    transl_np = transl.detach().cpu().numpy()

    for i, fn in enumerate(files):
        d = dict(meta) if meta is not None else {}
        d["betas"] = betas_np[i].tolist()
        d["root_pose"] = poses_np[i, 0].tolist()
        d["body_pose"] = poses_np[i, 1:22].tolist()
        d["jaw_pose"] = poses_np[i, 22].tolist()
        d["leye_pose"] = poses_np[i, 23].tolist()
        d["reye_pose"] = poses_np[i, 24].tolist()
        d["lhand_pose"] = poses_np[i, 25:40].tolist()
        d["rhand_pose"] = poses_np[i, 40:55].tolist()
        d["trans"] = transl_np[i].tolist()

        with open(os.path.join(dir_path, fn), "w") as f:
            json.dump(d, f)


@torch.no_grad()
def make_variant_from_raw(raw_dir: str, out_dir: str, fps: float, compose_cfg: Dict, overwrite: bool = True) -> str:
    if (not overwrite) and os.path.exists(out_dir) and len(os.listdir(out_dir)) > 0:
        return out_dir
    files, meta, poses, betas, transl = load_smplx_dir(raw_dir, device="cpu")
    poses_s, betas_s, transl_s = smplx_smooth_compose(poses, betas, transl, fps=fps, **compose_cfg)
    if os.path.exists(out_dir) and overwrite:
        for f in os.listdir(out_dir):
            if f.endswith(".json"):
                os.remove(os.path.join(out_dir, f))
    save_smplx_dir(out_dir, files, meta, poses_s, betas_s, transl_s)
    return out_dir


@torch.no_grad()
def compute_metrics_vs(raw_poses: torch.Tensor, raw_trans: torch.Tensor,
                       pred_poses: torch.Tensor, pred_trans: torch.Tensor) -> Dict[str, float]:
    try:
        raw_p6 = axis_angle_to_rotation_6d(raw_poses)
        prd_p6 = axis_angle_to_rotation_6d(pred_poses)
    except Exception:
        raw_p6 = axis_angle_to_rotation_6d(raw_poses)
        prd_p6 = axis_angle_to_rotation_6d(pred_poses)

    j_pose = _second_diff(prd_p6.reshape(prd_p6.shape[0], -1))
    j_trans = _second_diff(pred_trans)
    jitter_pose = float(j_pose.norm(dim=-1).mean().item()) if j_pose.numel() else 0.0
    jitter_trans = float(j_trans.norm(dim=-1).mean().item()) if j_trans.numel() else 0.0

    drift_trans = float((pred_trans[-1] - pred_trans[0]).norm().item()) if pred_trans.shape[0] > 1 else 0.0

    dev_pose = (prd_p6 - raw_p6).reshape(prd_p6.shape[0], -1).norm(dim=-1).mean()
    dev_trans = (pred_trans - raw_trans).norm(dim=-1).mean()
    deviation_pose = float(dev_pose.item())
    deviation_trans = float(dev_trans.item())

    return {
        "jitter_pose": jitter_pose,
        "jitter_trans": jitter_trans,
        "drift_trans": drift_trans,
        "deviation_pose": deviation_pose,
        "deviation_trans": deviation_trans,
    }


@torch.no_grad()
def compute_self_metrics(poses: torch.Tensor, transl: torch.Tensor) -> Dict[str, float]:
    """Compute self (intrinsic) metrics for a single SMPL-X sequence.

    Returns jitter_pose, jitter_trans, drift_trans.
    """
    try:
        p6 = axis_angle_to_rotation_6d(poses)
    except Exception:
        p6 = axis_angle_to_rotation_6d(poses)

    j_pose = _second_diff(p6.reshape(p6.shape[0], -1))
    j_trans = _second_diff(transl)
    jitter_pose = float(j_pose.norm(dim=-1).mean().item()) if j_pose.numel() else 0.0
    jitter_trans = float(j_trans.norm(dim=-1).mean().item()) if j_trans.numel() else 0.0
    drift_trans = float((transl[-1] - transl[0]).norm().item()) if transl.shape[0] > 1 else 0.0

    return {
        "jitter_pose": jitter_pose,
        "jitter_trans": jitter_trans,
        "drift_trans": drift_trans,
    }
