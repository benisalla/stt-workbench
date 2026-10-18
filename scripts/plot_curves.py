"""Plot the training curves of several runs on one figure.

    python scripts/plot_curves.py docs/images/synthetic_curves.png checkpoints/synthetic-*

Reads each run's log.csv (written by stt-train) and draws train loss, val WER and val CER.
"""

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

out = Path(sys.argv[1])
runs = [Path(p) for p in sys.argv[2:]]
fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))
for run in runs:
    rows = list(csv.DictReader((run / "log.csv").open()))
    train = [(int(r["step"]), float(r["loss"])) for r in rows if r["loss"]]
    val = [(int(r["step"]), float(r["wer"]), float(r["cer"])) for r in rows if r["wer"]]
    label = run.name
    axes[0].plot(*zip(*train), label=label)
    axes[1].plot([v[0] for v in val], [100 * v[1] for v in val], "o-", label=label)
    axes[2].plot([v[0] for v in val], [100 * v[2] for v in val], "s-", label=label)
for ax, title in zip(axes, ("train loss", "val WER (%)", "val CER (%)")):
    ax.set_title(title)
    ax.set_xlabel("step")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
axes[1].set_ylim(0, 105)
axes[2].set_ylim(0, 105)
plt.tight_layout()
out.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out, dpi=110)
print("wrote", out)
