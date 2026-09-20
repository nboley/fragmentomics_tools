"""Re-plot the CTCF pileups zoomed to a narrow window around the motif center.

Reads the arrays produced by the full run (pileups.npz) so no GPU/recompute is
needed.  Default zoom is +-128 bp, where the footprint and the model's
sequence-bias correction both live.
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NPZ = "/efs/analytics/nathanboley/background_model/ctcf_pileup/pileups.npz"
# the four most interpretable tracks, per strand
SHOW = ["short_first", "short_last", "mono_first", "mono_last"]
# CTCF motifs are 17bp, aligned on their center => they occupy [-8, +8]
MOTIF_LO, MOTIF_HI = -8, 8


def shade_motif(ax):
    ax.axvspan(MOTIF_LO, MOTIF_HI, color="steelblue", alpha=0.15, zorder=0)


def track_indices(tracks):
    """Map the readable names above onto indices into the 12 canonical tracks."""
    idx = {}
    for i, t in enumerate(tracks):
        t = str(t)
        band = "short" if "40_65" in t else ("mono" if "120_175" in t else None)
        cov = "first" if t.endswith("first") else ("last" if t.endswith("last") else None)
        strand = "+" if "strand_+" in t else "-"
        if band and cov:
            idx[(strand, f"{band}_{cov}")] = i
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default=NPZ)
    ap.add_argument("--zoom", type=int, default=128, help="half-width in bp")
    ap.add_argument("--model", default="multinomial")
    ap.add_argument("--clamp", default="bounded_0.2_5.0")
    ap.add_argument("--outdir", default="/home/nathanboley/src/fragmentomics_tools/data/ctcf_plots")
    args = ap.parse_args()

    d = np.load(args.npz, allow_pickle=True)
    pos = d["positions"]
    tracks = d["tracks"]
    samples = [str(s) for s in d["sample_names"]]
    ti = track_indices(tracks)

    sel = np.abs(pos) <= args.zoom
    x = pos[sel]
    os.makedirs(args.outdir, exist_ok=True)

    # ---- panel 1: uncorrected vs corrected, zoomed -----------------------
    unc = sum(d[f"unc__{s}"] for s in samples) / len(samples)
    cor = sum(d[f"corr__{args.model}__{args.clamp}__{s}"] for s in samples) / len(samples)

    fig, ax = plt.subplots(2, 4, figsize=(20, 8))
    for r, strand in enumerate(["+", "-"]):
        for c, name in enumerate(SHOW):
            i = ti[(strand, name)]
            a = ax[r, c]
            shade_motif(a)
            a.plot(x, unc[i][sel], color="0.45", lw=1.2, label="uncorrected")
            a.plot(x, cor[i][sel], color="crimson", lw=1.2, label="corrected")
            a.axvline(0, color="k", ls=":", lw=0.8)
            a.set_title(f"{strand} {name.replace('_', ' ')}")
            a.grid(alpha=0.3)
            if r == 1:
                a.set_xlabel("position relative to motif center (oriented, bp)")
            if (r, c) == (0, 0):
                a.legend(fontsize=9)
    fig.suptitle(
        f"CTCF endpoint pileup ZOOM +-{args.zoom}bp — {args.model} — clamp={args.clamp}\n"
        f"uncorrected vs corrected (n_sites={int(d['n_sites'])}, n_samples={len(samples)})"
    )
    fig.tight_layout()
    p1 = os.path.join(args.outdir, f"zoom{args.zoom}__uncorr_vs_corr__{args.model}__{args.clamp}.png")
    fig.savefig(p1, dpi=130)
    plt.close(fig)

    # ---- panel 2: three-model correction ratio, zoomed --------------------
    fig, ax = plt.subplots(2, 4, figsize=(20, 8))
    models = [m for m in (str(x) for x in d["model_names"]) if m != "uniform"]
    for r, strand in enumerate(["+", "-"]):
        for c, name in enumerate(SHOW):
            i = ti[(strand, name)]
            a = ax[r, c]
            shade_motif(a)
            for m in models:
                cm = sum(d[f"corr__{m}__{args.clamp}__{s}"] for s in samples) / len(samples)
                with np.errstate(divide="ignore", invalid="ignore"):
                    ratio = np.where(unc[i] > 0, cm[i] / np.maximum(unc[i], 1e-9), np.nan)
                a.plot(x, ratio[sel], lw=1.1, label=m)
            a.axhline(1.0, color="k", ls="-", lw=0.6)
            a.axvline(0, color="k", ls=":", lw=0.8)
            a.set_title(f"{strand} {name.replace('_', ' ')} (corrected/uncorrected)")
            a.grid(alpha=0.3)
            if r == 1:
                a.set_xlabel("position relative to motif center (oriented, bp)")
            if (r, c) == (0, 0):
                a.legend(fontsize=8)
    fig.suptitle(f"3-model correction ratio ZOOM +-{args.zoom}bp — clamp={args.clamp}")
    fig.tight_layout()
    p2 = os.path.join(args.outdir, f"zoom{args.zoom}__three_model_ratio__{args.clamp}.png")
    fig.savefig(p2, dpi=130)
    plt.close(fig)

    # ---- panel 3: expected (predicted) shape, zoomed ----------------------
    fig, ax = plt.subplots(2, 4, figsize=(20, 8))
    for r, strand in enumerate(["+", "-"]):
        for c, name in enumerate(SHOW):
            i = ti[(strand, name)]
            a = ax[r, c]
            shade_motif(a)
            for m in models:
                a.plot(x, d[f"probs__{m}"][i][sel], lw=1.1, label=m)
            a.axvline(0, color="k", ls=":", lw=0.8)
            a.set_title(f"{strand} {name.replace('_', ' ')} (predicted shape)")
            a.grid(alpha=0.3)
            if r == 1:
                a.set_xlabel("position relative to motif center (oriented, bp)")
            if (r, c) == (0, 0):
                a.legend(fontsize=8)
    fig.suptitle(f"Expected sequence-bias shape ZOOM +-{args.zoom}bp — the component divided out")
    fig.tight_layout()
    p3 = os.path.join(args.outdir, f"zoom{args.zoom}__expected_shape.png")
    fig.savefig(p3, dpi=130)
    plt.close(fig)

    # ---- panel 4: RAW (observed) vs PREDICTED (model expected counts) -----
    # expected = N_w * probs, so it is already on the same count scale as raw.
    fig, ax = plt.subplots(2, 4, figsize=(20, 8))
    for r, strand in enumerate(["+", "-"]):
        for c, name in enumerate(SHOW):
            i = ti[(strand, name)]
            a = ax[r, c]
            shade_motif(a)
            shade_motif(a)
            a.plot(x, unc[i][sel], color="0.25", lw=1.4, label="raw (observed)")
            for m in models:
                em = sum(d[f"exp__{m}__{s}"] for s in samples) / len(samples)
                a.plot(x, em[i][sel], lw=1.1, alpha=0.85, label=f"predicted ({m})")
            a.axvline(0, color="k", ls=":", lw=0.8)
            a.set_title(f"{strand} {name.replace('_', ' ')}")
            a.grid(alpha=0.3)
            if r == 1:
                a.set_xlabel("position relative to motif center (oriented, bp)")
            if (r, c) == (0, 0):
                a.legend(fontsize=8)
    fig.suptitle(
        f"RAW vs PREDICTED ZOOM +-{args.zoom}bp — shaded = 17bp CTCF motif [-8,+8]\n"
        "predicted = model expected counts (N_w x probs); raw/predicted IS the correction"
    )
    fig.tight_layout()
    p4 = os.path.join(args.outdir, f"zoom{args.zoom}__raw_vs_predicted.png")
    fig.savefig(p4, dpi=130)
    plt.close(fig)

    # ---- panel 5: raw and predicted on SEPARATE y-axes --------------------
    # predicted is much flatter than raw, so on a shared axis its structure is
    # invisible.  Twin axes show both shapes at once.
    fig, ax = plt.subplots(2, 4, figsize=(20, 8))
    for r, strand in enumerate(["+", "-"]):
        for c, name in enumerate(SHOW):
            i = ti[(strand, name)]
            a = ax[r, c]
            shade_motif(a)
            a.plot(x, unc[i][sel], color="0.15", lw=1.5, label="raw (left axis)")
            a.set_ylabel("raw counts", color="0.15")
            a.tick_params(axis="y", labelcolor="0.15")
            b = a.twinx()
            em = sum(d[f"exp__multinomial__{s}"] for s in samples) / len(samples)
            b.plot(x, em[i][sel], color="crimson", lw=1.5, label="predicted (right axis)")
            b.set_ylabel("predicted counts", color="crimson")
            b.tick_params(axis="y", labelcolor="crimson")
            a.axvline(0, color="k", ls=":", lw=0.8)
            a.set_title(f"{strand} {name.replace('_', ' ')}")
            a.grid(alpha=0.3)
            if r == 1:
                a.set_xlabel("position relative to motif center (oriented, bp)")
    fig.suptitle(
        f"RAW (black, left) vs PREDICTED (red, right) — separate scales — ZOOM +-{args.zoom}bp\n"
        "shaded = 17bp CTCF motif [-8,+8]"
    )
    fig.tight_layout()
    p5 = os.path.join(args.outdir, f"zoom{args.zoom}__raw_vs_predicted_twin.png")
    fig.savefig(p5, dpi=130)
    plt.close(fig)

    # ---- panel 6: OBSERVED shape vs PREDICTED shape -----------------------
    # Both normalised to sum to 1 over the full aggregation window (+-1000),
    # so depth is divided out and the two are directly comparable.  This is
    # the like-for-like view: the model predicts a SHAPE, so compare shapes.
    fig, ax = plt.subplots(2, 4, figsize=(20, 8))
    for r, strand in enumerate(["+", "-"]):
        for c, name in enumerate(SHOW):
            i = ti[(strand, name)]
            a = ax[r, c]
            shade_motif(a)
            obs_shape = unc[i] / unc[i].sum()
            a.plot(x, obs_shape[sel], color="0.15", lw=1.5, label="observed shape")
            for m in models:
                pm = d[f"probs__{m}"][i]
                a.plot(x, (pm / pm.sum())[sel], lw=1.2, alpha=0.85,
                       label=f"predicted shape ({m})")
            a.axvline(0, color="k", ls=":", lw=0.8)
            a.set_title(f"{strand} {name.replace('_', ' ')}")
            a.grid(alpha=0.3)
            if r == 1:
                a.set_xlabel("position relative to motif center (oriented, bp)")
            if (r, c) == (0, 0):
                a.legend(fontsize=8)
    fig.suptitle(
        f"OBSERVED shape vs PREDICTED shape — both normalised over +-1000bp — ZOOM +-{args.zoom}bp\n"
        "shaded = 17bp CTCF motif [-8,+8]; the gap between them is what correction acts on"
    )
    fig.tight_layout()
    p6 = os.path.join(args.outdir, f"zoom{args.zoom}__observed_vs_predicted_shape.png")
    fig.savefig(p6, dpi=130)
    plt.close(fig)

    for p in (p1, p2, p3, p4, p5, p6):
        print("wrote", p)


if __name__ == "__main__":
    main()
