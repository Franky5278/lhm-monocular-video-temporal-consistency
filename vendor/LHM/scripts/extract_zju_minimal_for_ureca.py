import argparse
import io
import json
from pathlib import Path

import cv2
import numpy as np


def norm_name(name: str) -> str:
    return str(name).replace("\\", "/").lstrip("./")


def load_annots_from_tar(tar_path: Path, seq_name: str):
    import tarfile

    target_suffix = f"{seq_name}/annots.npy"

    with tarfile.open(tar_path, "r:gz") as tar:
        for member in tar:
            name = norm_name(member.name)
            if name.endswith(target_suffix):
                f = tar.extractfile(member)
                if f is None:
                    raise RuntimeError(f"Cannot extract {member.name}")
                data = f.read()
                ann = np.load(io.BytesIO(data), allow_pickle=True).item()
                return ann

    raise FileNotFoundError(f"Cannot find {target_suffix} inside {tar_path}")


def write_bytes(path: Path, data: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)


def extract_selected_members(tar_path: Path, wanted_map: dict):
    import tarfile

    remaining = set(wanted_map.keys())
    extracted = {}

    with tarfile.open(tar_path, "r:gz") as tar:
        for member in tar:
            name = norm_name(member.name)
            if name not in remaining:
                continue

            f = tar.extractfile(member)
            if f is None:
                continue

            data = f.read()
            for out_path in wanted_map[name]:
                write_bytes(Path(out_path), data)

            extracted[name] = True
            remaining.remove(name)

            if not remaining:
                break

    return extracted, remaining


def read_bgr(path: Path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Cannot read image: {path}")
    return img


def write_video(image_paths, out_path: Path, fps: int):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    first = read_bgr(Path(image_paths[0]))
    h, w = first.shape[:2]

    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )

    for p in image_paths:
        img = read_bgr(Path(p))
        if img.shape[:2] != (h, w):
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        writer.write(img)

    writer.release()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tar_path", required=True)
    parser.add_argument("--seq", default="313")
    parser.add_argument("--source_cam", type=int, default=0)
    parser.add_argument("--heldout_cam", type=int, default=1)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=59)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--num_keyframes", type=int, default=5)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--out_root", required=True)
    args = parser.parse_args()

    tar_path = Path(args.tar_path)
    out_root = Path(args.out_root)
    seq_name = f"CoreView_{args.seq}"

    if not tar_path.exists():
        raise FileNotFoundError(f"Missing tar file: {tar_path}")

    out_root.mkdir(parents=True, exist_ok=True)

    print("[1/5] Loading annots.npy from tar.gz...")
    ann = load_annots_from_tar(tar_path, seq_name)

    ims_records = ann["ims"]
    cams = ann["cams"]
    num_cams = len(cams["K"])

    print("[INFO] frames:", len(ims_records))
    print("[INFO] cameras:", num_cams)

    if args.source_cam >= num_cams or args.heldout_cam >= num_cams:
        raise ValueError(
            f"Camera index out of range: num_cams={num_cams}, "
            f"source_cam={args.source_cam}, heldout_cam={args.heldout_cam}"
        )

    last = min(args.end, len(ims_records) - 1)
    frame_ids = list(range(args.start, last + 1, args.step))

    if not frame_ids:
        raise RuntimeError("No frames selected. Check start/end/step.")

    source_all_dir = out_root / "source_view_A_all_frames"
    source_key_dir = out_root / "source_view_A_frames"
    heldout_frame_dir = out_root / "heldout_view_B_frames"
    heldout_mask_dir = out_root / "heldout_view_B_masks"

    for d in [source_all_dir, source_key_dir, heldout_frame_dir, heldout_mask_dir]:
        d.mkdir(parents=True, exist_ok=True)

    wanted = {}
    source_all_paths = []
    heldout_paths = []
    heldout_mask_paths = []

    print("[2/5] Building extraction list...")

    for local_idx, fid in enumerate(frame_ids):
        rels = ims_records[fid]["ims"]

        src_rel = norm_name(f"{seq_name}/{rels[args.source_cam]}")
        gt_rel = norm_name(f"{seq_name}/{rels[args.heldout_cam]}")

        src_suffix = Path(src_rel).suffix.lower() or ".jpg"
        gt_suffix = Path(gt_rel).suffix.lower() or ".jpg"

        src_out = source_all_dir / f"src_frame_{fid:06d}{src_suffix}"
        gt_out = heldout_frame_dir / f"gt_frame_{fid:06d}{gt_suffix}"

        wanted.setdefault(src_rel, []).append(str(src_out))
        wanted.setdefault(gt_rel, []).append(str(gt_out))

        source_all_paths.append(str(src_out))
        heldout_paths.append(str(gt_out))

        mask_rel = norm_name(f"{seq_name}/mask_cihp/{rels[args.heldout_cam]}")
        mask_rel = str(Path(mask_rel).with_suffix(".png")).replace("\\", "/")
        mask_out = heldout_mask_dir / f"mask_{fid:06d}.png"

        wanted.setdefault(mask_rel, []).append(str(mask_out))
        heldout_mask_paths.append(str(mask_out))

    print(f"[INFO] selected frames: {len(frame_ids)}")
    print(f"[INFO] wanted tar members including optional masks: {len(wanted)}")

    print("[3/5] Extracting selected frames/masks from tar.gz...")
    extracted, remaining = extract_selected_members(tar_path, wanted)

    missing_required = []
    for p in source_all_paths + heldout_paths:
        if not Path(p).exists():
            missing_required.append(p)

    if missing_required:
        print("[ERROR] Missing required extracted images:")
        for p in missing_required[:20]:
            print("  ", p)
        raise RuntimeError(f"Missing {len(missing_required)} required images.")

    actual_masks = [p for p in heldout_mask_paths if Path(p).exists()]
    if not actual_masks:
        print("[WARN] No held-out masks extracted. Evaluation can still run without masks.")
    else:
        print(f"[INFO] extracted held-out masks: {len(actual_masks)}")

    print("[4/5] Selecting keyframes...")
    if len(source_all_paths) <= args.num_keyframes:
        key_local_ids = list(range(len(source_all_paths)))
    else:
        key_local_ids = np.linspace(
            0, len(source_all_paths) - 1, args.num_keyframes
        ).round().astype(int).tolist()

    keyframes = []
    for k, local_idx in enumerate(key_local_ids):
        src = Path(source_all_paths[local_idx])
        fid = frame_ids[local_idx]
        dst = source_key_dir / f"keyframe_{k:02d}_frame_{fid:06d}{src.suffix.lower()}"
        dst.write_bytes(src.read_bytes())
        keyframes.append(str(dst))

    print("[5/5] Writing videos...")
    source_video = out_root / "source_view_A.mp4"
    heldout_video = out_root / "heldout_view_B.mp4"

    write_video(source_all_paths, source_video, args.fps)
    write_video(heldout_paths, heldout_video, args.fps)

    manifest = {
        "tar_path": str(tar_path),
        "seq": args.seq,
        "source_cam": args.source_cam,
        "heldout_cam": args.heldout_cam,
        "start": args.start,
        "end": last,
        "step": args.step,
        "num_frames": len(frame_ids),
        "num_keyframes": args.num_keyframes,
        "fps": args.fps,
        "out_root": str(out_root),
        "source_frames_dir": str(source_key_dir),
        "source_all_frames_dir": str(source_all_dir),
        "source_video": str(source_video),
        "heldout_video": str(heldout_video),
        "heldout_frames_dir": str(heldout_frame_dir),
        "heldout_masks_dir": str(heldout_mask_dir) if actual_masks else None,
        "keyframes": keyframes,
        "num_heldout_masks": len(actual_masks),
    }

    with open(out_root / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(json.dumps(manifest, indent=2))
    print("[DONE] Minimal ZJU-MoCap extraction completed.")


if __name__ == "__main__":
    main()
