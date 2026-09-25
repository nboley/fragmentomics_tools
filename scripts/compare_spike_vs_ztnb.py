"""Three heatmaps: spike surface, ZTNB surface, and their log2 difference.

SAME SAMPLE on both sides. The pairing comes from the IBD manifest, which
carries sample_id -> result_id:

    RD-56670  ->  result 318863   (in the DEV results bucket, not prod)

ZTNB is fitted from the precomputed PAIRED-END duplicate histograms at
/efs/analytics/nathanboley/ctcf_fa_cache/duphist_merged/. The per-result BAMs
retained in the results bucket are SINGLE-END, where the observable is read
length (capped ~64bp by chemistry) rather than fragment length, so they cannot
share a length axis with the spike panel's true oligo lengths. The duphist data
is paired-end -- its mononucleosome mode at 166bp confirms it.

Both surfaces are normalised at (52bp, 50% GC) and rendered on the SPIKE grid,
so what is compared is SHAPE. Neither panel is an absolute P(seen); see
spike_recovery.py for why the absolute scale is out of reach.

Caveats are printed after the figure. Read them before drawing conclusions.
"""

import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

sys.path.insert(0, "/home/nathanboley/src/fragmentomics_tools/scripts")
from spike_recovery import spike_surface  # noqa: E402
from ztnb_from_duphist import (  # noqa: E402
    SPIKE_GCS,
    SPIKE_LENGTHS,
    fit_ztnb,
    p_seen_on_spike_grid,
)

# sample -> (result_id, results-bucket env), from the IBD manifest
PAIRS = {"RD-56670": (318863, "dev")}

REF_L, REF_G = 52, 50


def build(sample="RD-56670", profile="SNMv4C"):
    result_id, env = PAIRS[sample]

    surf, sdiag = spike_surface(result_id, profile=profile, env=env)
    spike = (
        surf.pivot_table(index="length", columns="model_gc_perc",
                         values="p_seen_ratio")
        .reindex(index=SPIKE_LENGTHS, columns=SPIKE_GCS)
        .values
    )

    model, zdiag = fit_ztnb(sample)
    pz = p_seen_on_spike_grid(model)
    ref = pz[SPIKE_LENGTHS.index(REF_L), SPIKE_GCS.index(REF_G)]
    ztnb = pz / ref

    with np.errstate(divide="ignore", invalid="ignore"):
        diff = np.log2(spike) - np.log2(ztnb)

    return spike, ztnb, diff, sdiag, zdiag, ref


def plot(spike, ztnb, diff, sample, result_id, out):
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.9))
    panels = [
        (spike, "spike-derived\nP(seen|L,G) / P(seen|ref)", "viridis", False),
        (ztnb, "ZTNB (paired-end duphist)\np_seen(L,G) / p_seen(ref)", "viridis", False),
        (diff, "log2(spike / ZTNB)\npositive = spike higher", "RdBu_r", True),
    ]
    for ax, (M, title, cmap, sym) in zip(axes, panels):
        if sym:
            lim = np.nanmax(np.abs(M))
            im = ax.imshow(M, cmap=cmap, aspect="auto", vmin=-lim, vmax=lim)
        else:
            im = ax.imshow(M, cmap=cmap, aspect="auto")
        ax.set_xticks(range(len(SPIKE_GCS)), [str(g) for g in SPIKE_GCS])
        ax.set_yticks(range(len(SPIKE_LENGTHS)), [str(x) for x in SPIKE_LENGTHS])
        ax.set_xlabel("GC %")
        ax.set_ylabel("fragment length (bp)")
        ax.set_title(title, fontsize=10)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                v = M[i, j]
                ax.text(j, i, "--" if not np.isfinite(v) else f"{v:.2f}",
                        ha="center", va="center", fontsize=7,
                        color="black" if sym else "white")
        fig.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle(
        f"{sample} / result {result_id} — SAME SAMPLE both sides, "
        f"true PE fragment lengths  ('--' = no data, NOT zero)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    return out


PLOT_DIR = "/home/nathanboley/src/fragmentomics_tools/docs/pending/spike_vs_ztnb_plots"


def main(sample="RD-56670", out=None):
    # write alongside the design doc, not /tmp -- figures under /tmp do not
    # render in review and are lost on reboot
    out = out or f"{PLOT_DIR}/spike_vs_ztnb_{sample}.png"
    spike, ztnb, diff, sdiag, zdiag, zref = build(sample)
    result_id, _ = PAIRS[sample]
    print("wrote", plot(spike, ztnb, diff, sample, result_id, out))

    both = np.isfinite(spike) & np.isfinite(ztnb)
    d = diff[np.isfinite(diff)]
    print(f"\nZTNB : {zdiag['molecules']:,} molecules, duplication="
          f"{zdiag['duplication']:.4f}, cells fitted "
          f"{zdiag['cells_fitted']}/{zdiag['cells_possible']}")
    print(f"spike: d(52)={sdiag['d_by_length'].get(52, float('nan'))}, "
          f"d(75)={sdiag['d_by_length'].get(75, float('nan')):.4f}")
    print(f"\ndynamic range  spike: {np.nanmin(spike):.2f}-{np.nanmax(spike):.2f}"
          f"   ZTNB: {np.nanmin(ztnb):.2f}-{np.nanmax(ztnb):.2f}")
    print(f"log2 ratio: mean={d.mean():+.3f} sd={d.std():.3f} "
          f"range=[{d.min():+.2f}, {d.max():+.2f}]")
    print(f"Spearman(spike, ZTNB) = "
          f"{spearmanr(spike[both], ztnb[both]).statistic:.3f}  (n={both.sum()})")

    print("\nCAVEATS")
    print(" 1. The spike surface is CONFOUNDED with oligo identity: one oligo per")
    print("    (length, GC) cell, so sequence effects cannot be separated from the")
    print("    cell. Adjacent GC cells differ several-fold, which a smooth bias")
    print("    cannot do. ZTNB averages many native fragments per cell instead.")
    print(" 2. SPANK-52C is ABSENT from this sample (0 reads), so d rests on a")
    print("    single length and the d(52) vs d(75) check cannot run here.")
    print(" 3. This sample carries BOTH v3 and v4 spike panels ('SNM.v3, SNM.v4'),")
    print("    so the SNMv4C molarity assumption may be wrong for some cells.")
    print(" 4. Duplication is ABUNDANCE-DEPENDENT, not a library constant: SPANK")
    print("    gives d=3.76 here while the native population gives 1.70. The")
    print("    SPANK -> GC-dSpark transfer stays plausible because their")
    print("    abundances are comparable, but d does NOT transfer to native.")
    print(" 5. Neither panel is an absolute P(seen).")


if __name__ == "__main__":
    main(*sys.argv[1:])
