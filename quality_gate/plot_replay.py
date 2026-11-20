"""Plot the four measurements from the archived KL gate replay."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from check import DEFAULT_INPUT, THRESHOLD, evaluate


report = evaluate(json.loads(DEFAULT_INPUT.read_text()))
rows = report["per_layer"]
figure, axes = plt.subplots(figsize=(7.6, 3.5), layout="constrained")
axes.axvspan(0.07, THRESHOLD, color="#e6f0ed", zorder=0)
axes.axvline(THRESHOLD, color="#287065", linewidth=1.5, linestyle="--", label="Cutoff (0.1)")
axes.scatter([row["kl"] for row in rows], range(len(rows)), color="#ad4132", s=58, zorder=3)
for index, row in enumerate(rows):
    axes.annotate(f'{row["kl"]:.4f}', (row["kl"], index), xytext=(8, 0),
                  textcoords="offset points", va="center", color="#342b34")
axes.set_xscale("log")
axes.set_xlim(0.07, 2.8)
axes.set_xticks([0.1, 0.2, 0.5, 1, 2], ["0.1", "0.2", "0.5", "1", "2"])
axes.set_yticks(range(len(rows)), [f'Layer {row["layer"]}' for row in rows])
axes.set_ylim(len(rows) - 0.45, -0.55)
axes.set_xlabel("KL divergence on benign prompts (log scale)")
axes.set_title("Retrospective KL gate · 0 of 4 archived candidates passed", loc="left", weight="bold")
axes.grid(axis="x", color="#dbe1e6", linewidth=0.8)
axes.spines[["top", "right", "left"]].set_visible(False)
axes.legend(loc="upper right", frameon=False)
figure.savefig(Path(__file__).with_name("archived_kl_gate.png"), dpi=170)
