"""Plot historical log measurements with explicit provenance and limitations."""
import ast
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

root = Path(__file__).resolve().parents[1]
log = root/'evidence/historical/poster_gradio_v7.log'
rows = []
for number, line in enumerate(log.read_bytes().decode('utf8').split('\n'), 1):
    if '[metrics] self for ' not in line or 'dev vs raw: ' not in line:
        continue
    name = 'Gaussian baseline' if '_baseline:' in line else 'Gaussian + EMA' if '_ours_full:' in line else None
    if name:
        rows.append({'variant': name, 'log_line': number,
                     **ast.literal_eval(line.split('dev vs raw: ', 1)[1])})
if len(rows) != 2:
    raise ValueError('Expected exactly one historical pair; inspect source log')
out = root/'evidence/historical'
(out/'motion_measurements.json').write_text(json.dumps({
    'source': 'poster_gradio_v7.log', 'rows': rows,
    'scope': 'Historical 120-frame run; duplicated 60-frame input. Not a new GPU result.',
    'deviation_reference': 'raw fitted motion, not ground-truth motion',
    'drift_trans_definition': 'endpoint displacement in meters, not error',
}, indent=2))
with (out/'motion_measurements.csv').open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
fig, axes = plt.subplots(1, 2, figsize=(10, 4.3))
for ax, fields, labels, title in [
    (axes[0], ['jitter_pose', 'jitter_trans'], ['Pose jitter', 'Translation jitter'], 'Second-difference magnitude'),
    (axes[1], ['deviation_pose', 'deviation_trans'], ['Pose deviation', 'Translation deviation'], 'Deviation from raw fitted motion'),
]:
    x = np.arange(2)
    for i, (row, color) in enumerate(zip(rows, ['#8997a6', '#126a8a'])):
        values = np.array([row[k]/rows[0][k] for k in fields])
        bars = ax.bar(x+(i-.5)*.35, values, .35, label=row['variant'], color=color)
        ax.bar_label(bars, labels=[f'{v:.2f}x' for v in values], padding=3)
    ax.set_xticks(x, labels)
    ax.set_ylabel('Ratio to historical Gaussian baseline')
    ax.set_title(title)
    ax.set_ylim(0, ax.get_ylim()[1]*1.15)
    ax.axhline(1, ls=':', color='#444444', lw=.8)
fig.legend(*axes[0].get_legend_handles_labels(), frameon=False,
           loc='upper center', bbox_to_anchor=(.5, .91), ncol=2, fontsize=9)
fig.suptitle('Historical evidence: lower jitter accompanies larger motion deviation', fontsize=13)
fig.text(.5, .02, 'Original log lines 80 / 85; 120 frames from duplicated input. Deviation is not GT accuracy. New GPU runs pending.',
         ha='center', fontsize=8)
fig.tight_layout(rect=(0, .06, 1, .86))
figures = root/'figures'
figures.mkdir(exist_ok=True)
fig.savefig(figures/'historical_motion_tradeoff.png', dpi=180)
fig.savefig(figures/'historical_motion_tradeoff.svg')
print(json.dumps(rows, indent=2))
