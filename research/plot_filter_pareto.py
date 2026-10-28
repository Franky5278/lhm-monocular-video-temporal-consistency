import csv
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rows = list(csv.DictReader(open(sys.argv[1], encoding="utf8")))
out = sys.argv[2]

groups = {
    "moving average": ("#1f77b4", "o"),
    "one-euro": ("#9467bd", "s"),
    "zero-phase EMA": ("#2ca02c", "^"),
}
fig, ax = plt.subplots(figsize=(6.6, 4.4))
for prefix, (color, marker) in groups.items():
    pts = [(float(r["deviation_pose"]), float(r["jitter_pose"]))
           for r in rows if r["method"].startswith(prefix)]
    pts.sort()
    if pts:
        xs, ys = zip(*pts)
        ax.plot(xs, ys, marker=marker, color=color, label=prefix, markersize=4, linewidth=1.2)

ours = [r for r in rows if r["method"].startswith("ours")][0]
raw = [r for r in rows if r["method"].startswith("raw")][0]
ax.scatter([float(ours["deviation_pose"])], [float(ours["jitter_pose"])],
           marker="*", s=220, color="#d62728", zorder=5, label="this pipeline (causal EMA)")
ax.scatter([float(raw["deviation_pose"])], [float(raw["jitter_pose"])],
           marker="x", s=90, color="black", zorder=5, label="raw SMPL-X motion")

ax.set_xlabel("pose deviation from raw motion (6D, lower is better)")
ax.set_ylabel("pose jitter, second difference (lower is better)")
ax.set_title("Matched-fidelity filter comparison, ZJU-313 (60 frames)")
ax.grid(alpha=.3)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(out, dpi=180)
print("wrote", out)
