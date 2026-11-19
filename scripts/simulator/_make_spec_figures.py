"""Figures for docs/pending/simulator_spec.md.

Both are computed from the real code and real measured tables, never drawn by
hand, so a change in behaviour shows up in the figure.

    propensity.png  -- the spread of r(h) over the 4096 hexamers, per table.
                       Input: an .npz from scripts/simulator/measure_cut_site_hexamers.py.
    end_weight.png  -- fl_end_weight across one tile and across three adjacent
                       tiles. Input: a run's p sidecar, for a real f(L).

Usage:
    python scripts/simulator/_make_spec_figures.py [--measure-npz PATH] [--sidecar PATH]
"""
import argparse
import gzip
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from background_model.constants import L_MAX, L_MIN
from background_model.fragment_lengths import FragmentLengthDist, fl_end_weight

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIGS = os.path.join(_REPO, "docs", "pending", "simulator_spec_figs")

DEFAULT_NPZ = "/tmp/em_measure_5k/RD-56804.cut_site_measure.npz"
DEFAULT_SIDECAR = "/efs/analytics/nathanboley/sim_d166_lowfix_w1/RD-56804.p.tsv.gz"

BG = "#111317"
CYAN, PINK, AMBER, GREEN, GREY = "#4dd0e1", "#ff6ec7", "#ffc857", "#7ee787", "#9aa0a6"
TABLES = [("start_fwd", CYAN), ("end_fwd", PINK),
          ("start_rev", AMBER), ("end_rev", GREEN)]


def _style(ax):
    ax.set_facecolor(BG)
    ax.grid(alpha=0.15)
    for s in ax.spines.values():
        s.set_color("#3a3f44")


def propensity_figure(npz_path):
    """How uneven cut-site preference is, over all 4096 hexamers."""
    z = np.load(npz_path)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))
    fig.patch.set_facecolor(BG)

    for name, colour in TABLES:
        r = np.asarray(z[f"r_{name}"], dtype=np.float64)
        rel = r / r.mean(dtype=np.float64)          # mean 1, so tables compare
        nz = rel[rel > 0]
        ax1.hist(np.log10(nz), bins=70, histtype="step", lw=1.8,
                 color=colour, label=name)
        ax2.plot(np.sort(rel)[::-1], color=colour, lw=1.6, label=name)

    ax1.set_xlabel("log10 propensity, relative to the mean")
    ax1.set_ylabel("hexamers")
    ax1.set_title("Cut-site preference is uneven", color="w", fontsize=11)
    ax1.legend(frameon=False, fontsize=8)

    ax2.set_yscale("log")
    ax2.set_xlabel("hexamers, most preferred first")
    ax2.set_ylabel("propensity, relative to the mean")
    ax2.set_title("Ranked, over all 4096", color="w", fontsize=11)

    r0 = np.asarray(z["r_start_fwd"], dtype=np.float64)
    rel0 = r0 / r0.mean(dtype=np.float64)
    hi, lo = np.quantile(rel0[rel0 > 0], [0.99, 0.01])
    ax2.annotate(f"top 1% / bottom 1% = {hi / lo:.0f}x  (start_fwd)",
                 xy=(0.5, 0.06), xycoords="axes fraction", color=GREY, fontsize=9)

    for ax in (ax1, ax2):
        _style(ax)
    fig.tight_layout()
    out = os.path.join(FIGS, "propensity.png")
    fig.savefig(out, dpi=130, facecolor=fig.get_facecolor())
    plt.close(fig)
    return out, hi / lo


def _fl_from_sidecar(path):
    lengths = []
    with gzip.open(path, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.split("\t")
            if f[0] == "contig":
                continue
            lengths.append(int(f[2]) - int(f[1]))
    lengths = np.asarray(lengths, dtype=np.int64)
    counts = np.bincount(lengths - L_MIN, minlength=L_MAX - L_MIN + 1).astype(np.int64)
    return FragmentLengthDist(counts, L_MIN), len(lengths)


def end_weight_figure(sidecar):
    """Why the end expectation ramps, and why adjacent tiles complete it."""
    fl, n = _fl_from_sidecar(sidecar)
    region_len = 1536
    n_hex = region_len + fl.max_fl
    w = fl_end_weight(n_hex, region_len, fl)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6.2))
    fig.patch.set_facecolor(BG)

    ax1.plot(np.arange(n_hex), w, color=CYAN, lw=1.8)
    ax1.axvspan(0, region_len, color=GREY, alpha=0.10)
    ax1.axvline(region_len, color=AMBER, lw=1.2, ls="--")
    ax1.text(region_len - 20, 1.14, "region stop", color=AMBER, ha="right", fontsize=9)
    ax1.text(region_len / 2, 0.45, "every length reaches here\nweight = 1",
             color=GREY, ha="center", va="center", fontsize=9)
    ax1.annotate("fewer lengths reach\nthe first bases", xy=(90, 0.42),
                 xytext=(300, 0.15), color=PINK, fontsize=9,
                 arrowprops=dict(arrowstyle="->", color=PINK, lw=1.3))
    ax1.annotate("ends past the region\nbelong to the next one",
                 xy=(region_len + 90, 0.42), xytext=(region_len - 600, 0.10),
                 color=PINK, fontsize=9,
                 arrowprops=dict(arrowstyle="->", color=PINK, lw=1.3))
    ax1.set_xlim(0, n_hex)
    ax1.set_ylim(0, 1.25)
    ax1.set_ylabel("end weight")
    ax1.set_title("How often each position can be a fragment end",
                  color="w", fontsize=11)

    span = 3 * region_len
    total = np.zeros(span + fl.max_fl, dtype=np.float64)
    for t in range(3):
        o = t * region_len
        ax2.plot(np.arange(o, o + n_hex), w, color=CYAN, lw=1.2, alpha=0.75)
        total[o:o + n_hex] += w
    ax2.plot(np.arange(len(total)), total, color=PINK, lw=2.0, label="sum over regions")
    for t in (1, 2):
        ax2.axvline(t * region_len, color=AMBER, lw=1.0, ls="--")
    interior = total[fl.max_fl:span]
    ax2.set_xlim(0, span)
    ax2.set_ylim(0, 1.25)
    ax2.set_xlabel("position along the genome")
    ax2.set_ylabel("end weight")
    ax2.set_title(f"Adjacent regions complete each other: interior sum = "
                  f"{interior.min():.6f} to {interior.max():.6f}",
                  color="w", fontsize=11)
    ax2.legend(loc="lower center", frameon=False, fontsize=9)

    for ax in (ax1, ax2):
        _style(ax)
    fig.tight_layout()
    out = os.path.join(FIGS, "end_weight.png")
    fig.savefig(out, dpi=130, facecolor=fig.get_facecolor())
    plt.close(fig)
    return out, n, (interior.min(), interior.max())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--measure-npz", default=DEFAULT_NPZ)
    p.add_argument("--sidecar", default=DEFAULT_SIDECAR)
    args = p.parse_args()

    os.makedirs(FIGS, exist_ok=True)
    plt.style.use("dark_background")

    out, ratio = propensity_figure(args.measure_npz)
    print(f"wrote {out}  (top1%/bottom1% = {ratio:.1f}x)")
    out, n, (lo, hi) = end_weight_figure(args.sidecar)
    print(f"wrote {out}  (f(L) from {n} fragments; interior sum {lo} to {hi})")


if __name__ == "__main__":
    main()
