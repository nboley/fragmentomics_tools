"""End-weight figure for the simulator spec, computed from fl_end_weight itself."""
import gzip
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from background_model.constants import L_MAX, L_MIN
from background_model.fragment_lengths import FragmentLengthDist, fl_end_weight

SIDECAR = "/efs/analytics/nathanboley/sim_d166_lowfix_w1/RD-56804.p.tsv.gz"
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(_REPO, "docs", "pending", "simulator_spec_figs", "end_weight.png")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

# A real length distribution: the lengths of the 200-region reference run.
lengths = []
with gzip.open(SIDECAR, "rt") as fh:
    for line in fh:
        if line.startswith("#"):
            continue
        f = line.split("\t")
        if f[0] == "contig":
            continue
        lengths.append(int(f[2]) - int(f[1]))
lengths = np.asarray(lengths, dtype=np.int64)
counts = np.bincount(lengths - L_MIN, minlength=L_MAX - L_MIN + 1).astype(np.int64)
fl = FragmentLengthDist(counts, L_MIN)

region_len = 1536
n_hex = region_len + fl.max_fl           # the padded track one tile reads
w = fl_end_weight(n_hex, region_len, fl)

plt.style.use("dark_background")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6.2), height_ratios=[1, 1])
fig.patch.set_facecolor("#111317")

CYAN, PINK, AMBER, GREY = "#4dd0e1", "#ff6ec7", "#ffc857", "#9aa0a6"

# --- top: one tile ---------------------------------------------------------
x = np.arange(n_hex)
ax1.plot(x, w, color=CYAN, lw=1.8)
ax1.axvspan(0, region_len, color=GREY, alpha=0.10)
ax1.axvline(region_len, color=AMBER, lw=1.2, ls="--")
ax1.text(region_len - 20, 1.14, "region ends", color=AMBER, ha="right", fontsize=9)
ax1.text(region_len / 2, 0.45, "every length reaches here\nweight = 1",
         color=GREY, ha="center", va="center", fontsize=9)
ax1.annotate("fewer lengths reach\nthe first bases",
             xy=(90, 0.42), xytext=(300, 0.18), color=PINK, fontsize=9,
             arrowprops=dict(arrowstyle="->", color=PINK, lw=1.3))
ax1.annotate("ends past the region\nbelong to the next tile",
             xy=(region_len + 90, 0.42), xytext=(region_len - 560, 0.12),
             color=PINK, fontsize=9,
             arrowprops=dict(arrowstyle="->", color=PINK, lw=1.3))
ax1.set_xlim(0, n_hex)
ax1.set_ylim(0, 1.25)
ax1.set_ylabel("end weight  w(i)")
ax1.set_title("One tile: how often each position can be a fragment end",
              color="w", fontsize=11)

# --- bottom: three adjacent tiles, and their sum ---------------------------
span = 3 * region_len
total = np.zeros(span + fl.max_fl, dtype=np.float64)
for t in range(3):
    o = t * region_len
    ax2.plot(np.arange(o, o + n_hex), w, color=CYAN, lw=1.2, alpha=0.75)
    total[o:o + n_hex] += w
ax2.plot(np.arange(len(total)), total, color=PINK, lw=2.0, label="sum over tiles")
for t in range(1, 3):
    ax2.axvline(t * region_len, color=AMBER, lw=1.0, ls="--")
interior = total[fl.max_fl:span]
ax2.set_xlim(0, span)
ax2.set_ylim(0, 1.25)
ax2.set_xlabel("position along the genome")
ax2.set_ylabel("end weight")
ax2.set_title(
    f"Adjacent tiles complement each other: interior sum = "
    f"{interior.min():.6f} to {interior.max():.6f}", color="w", fontsize=11)
ax2.legend(loc="lower center", frameon=False, fontsize=9)

for ax in (ax1, ax2):
    ax.set_facecolor("#111317")
    ax.grid(alpha=0.15)
    for s in ax.spines.values():
        s.set_color("#3a3f44")

fig.tight_layout()
fig.savefig(OUT, dpi=130, facecolor=fig.get_facecolor())
print("wrote", OUT)
print("interior sum min/max:", interior.min(), interior.max())
print("n fragments used for f(L):", len(lengths), "min_fl", fl.min_fl, "max_fl", fl.max_fl)
