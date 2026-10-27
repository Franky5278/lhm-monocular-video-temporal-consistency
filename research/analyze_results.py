"""Aggregate experiment results into tables, figures and a markdown report.

Runs on the local machine against downloaded results; it never touches the GPU
and never invents a number: every row comes from a stage.json / summary.json.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

METRIC_KEYS = ["psnr_full_db", "psnr_masked_db", "ssim_full", "lpips_full",
               "temporal_signed_delta_l1_0_1"]
SHORT = {"psnr_full_db": "PSNR full (dB)", "psnr_masked_db": "PSNR masked (dB)",
         "ssim_full": "SSIM", "lpips_full": "LPIPS",
         "temporal_signed_delta_l1_0_1": "Temporal delta"}


def load_stage(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf8"))
    except Exception:
        return {}


def collect(root: Path) -> list[dict]:
    rows = []
    for stage in sorted(root.rglob("stage.json")):
        rel = stage.parent.relative_to(root).as_posix()
        data = load_stage(stage)
        if data.get("status") != "success":
            rows.append({"stage": rel, "variant": rel.split("/")[-1], "status": data.get("status"),
                         "seconds": data.get("seconds")})
            continue
        row = {"stage": rel, "variant": rel.split("/")[-1], "status": "success",
               "seconds": round(data.get("seconds", 0), 1)}
        out = data.get("output") or {}
        if isinstance(out, dict):
            mm = out.get("motion_metrics") or {}
            for key in ("jitter_pose_6d_second_difference", "jitter_translation_second_difference",
                        "pose_deviation_6d_norm", "translation_deviation_m",
                        "pose_geodesic_deviation_mean_deg", "translation_speed_mean_m_per_s",
                        "endpoint_displacement_m"):
                if key in mm:
                    row[key] = mm[key]
        summary = stage.parent / "summary.json"
        if summary.exists():
            metrics = json.loads(summary.read_text(encoding="utf8")).get("metrics", {})
            for key in METRIC_KEYS:
                if key in metrics and metrics[key]:
                    row[key] = metrics[key]["mean"]
                    row[key + "_ci"] = metrics[key].get("block_bootstrap_95_interval")
        rows.append(row)
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="", encoding="utf8") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in r.items()})


def bar_plot(rows: list[dict], metric: str, title: str, path: Path, limit=None) -> None:
    data = [r for r in rows if metric in r and r.get("status") == "success"]
    data = [r for r in data if "/eval_" in r["stage"]]
    if limit:
        data = data[:limit]
    if not data:
        return
    labels = [r["stage"].split("/")[-1].replace("eval_", "") for r in data]
    values = [r[metric] for r in data]
    fig, ax = plt.subplots(figsize=(max(7, len(labels) * 0.45), 4.2))
    ax.bar(range(len(values)), values, color="#2f6feb")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=75, ha="right", fontsize=8)
    ax.set_ylabel(SHORT.get(metric, metric))
    ax.set_title(title)
    ax.grid(axis="y", alpha=.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def ablation_plot(rows: list[dict], path: Path) -> None:
    pts = [r for r in rows if "jitter_pose_6d_second_difference" in r and "pose_deviation_6d_norm" in r]
    if not pts:
        return
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for r in pts:
        ax.scatter(r["jitter_pose_6d_second_difference"], r["pose_deviation_6d_norm"], s=42)
        ax.annotate(r["variant"], (r["jitter_pose_6d_second_difference"], r["pose_deviation_6d_norm"]),
                    fontsize=7, xytext=(4, 3), textcoords="offset points")
    ax.set_xlabel("Pose jitter (second difference)")
    ax.set_ylabel("Pose deviation from raw fit")
    ax.set_title("Smoothing strength trade-off (measured on ZJU-313)")
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def markdown(rows: list[dict], path: Path, meta: dict) -> None:
    ok = [r for r in rows if r.get("status") == "success"]
    bad = [r for r in rows if r.get("status") not in ("success", None)]
    lines = ["# GPU experiment results", "",
             f"Generated from `stage.json` records under `{meta['root']}`. "
             f"{len(ok)} stages succeeded, {len(bad)} failed or were skipped.", ""]
    lines += ["## Reconstruction and evaluation", "",
              "| stage | PSNR full | PSNR masked | SSIM | LPIPS | temporal delta | seconds |",
              "|---|---|---|---|---|---|---|"]
    for r in ok:
        if not any(k in r for k in METRIC_KEYS):
            continue
        def fmt(k):
            v = r.get(k)
            return f"{v:.4f}" if isinstance(v, (int, float)) else "—"
        lines.append(f"| {r['stage']} | {fmt('psnr_full_db')} | {fmt('psnr_masked_db')} | "
                     f"{fmt('ssim_full')} | {fmt('lpips_full')} | {fmt('temporal_signed_delta_l1_0_1')} | "
                     f"{r.get('seconds', '—')} |")
    lines += ["", "## Motion metrics (no image metric attached)", "",
              "| stage | pose jitter | trans jitter | pose deviation | trans deviation |",
              "|---|---|---|---|---|"]
    for r in ok:
        if "jitter_pose_6d_second_difference" not in r:
            continue
        lines.append(f"| {r['stage']} | {r['jitter_pose_6d_second_difference']:.6f} | "
                     f"{r.get('jitter_translation_second_difference', float('nan')):.8f} | "
                     f"{r.get('pose_deviation_6d_norm', float('nan')):.6f} | "
                     f"{r.get('translation_deviation_m', float('nan')):.8f} |")
    if bad:
        lines += ["", "## Failed or skipped stages", ""]
        for r in bad:
            lines.append(f"- `{r['stage']}` — {r.get('status')}")
    lines += ["", "## Metric definitions", "",
              "- PSNR/SSIM/LPIPS are computed after resizing both videos to 512x512; equal frame "
              "count and equal fps are enforced, otherwise the evaluator raises.",
              "- `PSNR masked` uses the held-out view's GT foreground mask; `PSNR full` covers the whole frame.",
              "- Temporal delta is the mean absolute second difference difference in 0-1 units, i.e. a "
              "synchronized motion-consistency proxy, not a perceptual score.",
              "- Pose jitter is a per-frame second difference of the 6D rotation vector; the translation "
              "jitter uses the same operator on metric translations.", ""]
    path.write_text("\n".join(lines), encoding="utf8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    root, out = Path(args.root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = collect(root)
    write_csv(rows, out / "metrics_table.csv")
    bar_plot(rows, "psnr_full_db", "Full-frame PSNR by variant", out / "fig_psnr_full.png")
    bar_plot(rows, "ssim_full", "SSIM by variant", out / "fig_ssim.png")
    bar_plot(rows, "lpips_full", "LPIPS by variant (lower is better)", out / "fig_lpips.png")
    bar_plot(rows, "temporal_signed_delta_l1_0_1", "Synchronized temporal delta", out / "fig_temporal.png")
    ablation_plot(rows, out / "fig_smoothing_tradeoff.png")
    markdown(rows, out / "results_tables.md", {"root": str(root)})
    print(f"wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
