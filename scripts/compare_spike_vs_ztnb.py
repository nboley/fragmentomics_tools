"""Heatmaps: spike-derived vs ZTNB recovery shape, and their difference.

Both surfaces are normalised at the same reference cell (52bp, 50% GC), so what
is compared is SHAPE. Neither panel is an absolute P(seen) -- see
spike_recovery.py for why the absolute scale is unobtainable from spikes.

Read the caveats printed at the end before drawing conclusions from the figure.
"""

import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, "/home/nathanboley/src/fragmentomics_tools/scripts")
from spike_recovery import spike_surface, ztnb_ratio_grid  # noqa: E402

from flgc.model import GCFlDistModel  # noqa: E402

ZTNB_PATH = (
    "/efs/analytics/nathanboley/workdir/e3/94a6396168af4a3b2af49a5b908272/"
    "flgc_model.ztnb.json"
)


def main(result_id=200000, out="/tmp/spike_vs_ztnb.png"):
    surf, diag = spike_surface(result_id)
    piv = surf.pivot_table(index="length", columns="model_gc_perc",
                           values="p_seen_ratio")
    lengths = list(piv.index)
    gcs = list(piv.columns)
    spike = piv.values

    ztnb_model = GCFlDistModel.load(ZTNB_PATH)
    ztnb, ztnb_ref = ztnb_ratio_grid(ztnb_model, lengths, gcs)

    # log2 so that "2x higher" and "2x lower" are symmetric about zero
    with np.errstate(divide="ignore", invalid="ignore"):
        diff = np.log2(spike) - np.log2(ztnb)

    n_both = int(np.isfinite(spike * ztnb).sum())

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    panels = [
        (spike, "spike-derived\nP(seen|L,G) / P(seen|ref)", "viridis", None),
        (ztnb, "ZTNB\np_seen(L,G) / p_seen(ref)", "viridis", None),
        (diff, "log2(spike / ZTNB)\npositive = spike higher", "RdBu_r", "sym"),
    ]
    for ax, (M, title, cmap, scale) in zip(axes, panels):
        if scale == "sym":
            lim = np.nanmax(np.abs(M)) if np.isfinite(M).any() else 1.0
            im = ax.imshow(M, cmap=cmap, aspect="auto", vmin=-lim, vmax=lim)
        else:
            im = ax.imshow(M, cmap=cmap, aspect="auto")
        ax.set_xticks(range(len(gcs)), [f"{g:g}" for g in gcs])
        ax.set_yticks(range(len(lengths)), [str(int(x)) for x in lengths])
        ax.set_xlabel("GC %")
        ax.set_ylabel("fragment length (bp)")
        ax.set_title(title, fontsize=10)
        for a in range(M.shape[0]):
            for b in range(M.shape[1]):
                v = M[a, b]
                ax.text(b, a, "--" if not np.isfinite(v) else f"{v:.2f}",
                        ha="center", va="center", fontsize=7,
                        color="white" if np.isfinite(v) and scale != "sym" else "black")
        fig.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle(
        f"result {result_id} — shape comparison, both normalised at "
        f"{diag['ref_cell'][0]}bp/{diag['ref_cell'][1]:g}% GC "
        f"('--' = no ZTNB fit, NOT zero)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")

    print(f"\nd(52)={diag['d_by_length'][52]:.4f}  d(75)={diag['d_by_length'][75]:.4f}"
          f"  ratio={diag['d_ratio']:.4f}")
    print(f"ZTNB reference p_seen = {ztnb_ref:.4f}")
    print(f"cells with BOTH surfaces: {n_both} / {spike.size}")
    if n_both:
        d = diff[np.isfinite(diff)]
        print(f"log2 ratio: mean={d.mean():+.3f} sd={d.sd() if hasattr(d,'sd') else d.std():.3f}"
              f"  range=[{d.min():+.3f}, {d.max():+.3f}]")
        print(f"  -> spike surface is {2**d.mean():.2f}x the ZTNB surface on average")

    print("\nCAVEATS")
    print(" 1. ZTNB model is from a pipeline TEST sample (8012 reads), not the")
    print("    same sample as the spike counts. This is a mechanism demo, NOT a")
    print("    real comparison -- the two panels are different samples.")
    print(" 2. ZTNB fits only 18 of 64 of its own cells; 12 of those have")
    print("    p_seen < 0.333, i.e. they clamp at MAX_WEIGHT in normal use.")
    print(" 3. The spike surface is confounded with oligo identity: one oligo")
    print("    per cell, so sequence effects are inseparable from (length, GC).")
    print(" 4. Neither panel is an absolute P(seen).")


if __name__ == "__main__":
    main(*(int(a) if a.isdigit() else a for a in sys.argv[1:]))
