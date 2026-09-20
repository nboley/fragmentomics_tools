"""CTCF meta-profile plots built with the fragmentomics_tools track plotting library.

Uses `fragmentomics_tools.plot.tracks` (Tracks / VectorTracks / VLine) rather than
hand-rolled matplotlib, so these plots share styling and behaviour with the rest of
the codebase.

Two things this fixes relative to the ad-hoc version:

  * y-limits are computed from the data OUTSIDE the motif, so the few-bp prediction
    spikes inside the 17bp motif clip instead of compressing everything else into
    the bottom of the panel;
  * the motif interval and centre are drawn as VLines.

The profiles are meta-profiles (aggregated over thousands of sites in coordinates
relative to the motif centre), not a genomic locus, so they are plotted against a
synthetic `meta` region; VLines mark the motif.  Reads pileups.npz -- no GPU.
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")

from fragmentomics_tools.plot.tracks import Tracks, VectorTracks, VLine, smooth1d
from fragmentomics_tools.region import Region

NPZ = "/efs/analytics/nathanboley/background_model/ctcf_pileup/pileups.npz"
SHOW = ["short_first", "short_last", "mono_first", "mono_last"]
MOTIF_LO, MOTIF_HI = -8, 8          # 17bp CTCF motif, aligned on centre


def masked_smooth(arr, pos, mask_half, window_len):
    """Smooth with positions |pos| <= mask_half excluded.

    The library's smooth1d is not NaN-aware, so smoothing a masked profile with
    it would let the masked (motif) values bleed into neighbours -- a 100bp
    window would contaminate +-50bp and defeat the mask entirely.  We therefore
    use a normalised convolution: convolve the values and the validity mask
    separately and divide, so each output point is the mean of the VALID inputs
    in its window.  Masked positions are returned as NaN (drawn as gaps).
    """
    valid = (np.abs(pos) > mask_half).astype(float)
    vals = np.where(valid > 0, arr, 0.0)
    if window_len and window_len > 1:
        k = np.ones(int(window_len)) / float(window_len)
        num = np.convolve(vals, k, mode="same")
        den = np.convolve(valid, k, mode="same")
        out = np.divide(num, den, out=np.full_like(num, np.nan), where=den > 1e-9)
    else:
        out = np.where(valid > 0, arr, np.nan)
    return np.where(valid > 0, out, np.nan)


def track_indices(tracks):
    idx = {}
    for i, t in enumerate(tracks):
        t = str(t)
        band = "short" if "40_65" in t else ("mono" if "120_175" in t else None)
        cov = "first" if t.endswith("first") else ("last" if t.endswith("last") else None)
        strand = "+" if "strand_+" in t else "-"
        if band and cov:
            idx[(strand, f"{band}_{cov}")] = i
    return idx


def compute_ylim(arrays, pos, mode="auto", pad=0.08):
    """Y-limits for a panel.

    'flank' scales to data OUTSIDE the motif, so the few-bp in-motif prediction
    spikes clip rather than compressing the flanking structure.  That is the
    right choice for wide views, but it fails for narrow ones: at +-16bp barely
    any non-motif positions remain, so the spikes run off the panel.  'auto'
    therefore falls back to scaling over everything visible once the motif
    dominates the window.  'full' always uses all visible data.
    """
    off = np.abs(pos) > MOTIF_HI
    if mode == "auto":
        # Decide on how much of the VISIBLE window the motif occupies.  When it
        # is a small feature (wide view) scale to the flanks so its few-bp
        # spikes clip; once it fills a sizeable part of the window it IS the
        # subject of the plot, so scale to everything.
        motif_frac = float((~off).mean())
        mode = "full" if motif_frac > 0.15 else "flank"
    if mode == "flank":
        # deliberately scaling to the flanks so the in-motif spikes clip; use
        # percentiles so one extreme base does not dictate the whole panel
        lo = min(float(np.nanpercentile(a[off], 0.5)) for a in arrays)
        hi = max(float(np.nanpercentile(a[off], 99.5)) for a in arrays)
    else:
        # 'full' must COVER EVERY SERIES -- true min/max over all models, no
        # percentile clipping, so nothing runs off the panel in any curve
        lo = min(float(np.nanmin(a)) for a in arrays)
        hi = max(float(np.nanmax(a)) for a in arrays)
    if hi <= lo:
        lo, hi = float(np.nanmin(arrays[0])), float(np.nanmax(arrays[0])) + 1e-12
    span = hi - lo
    return lo - pad * span, hi + pad * span


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default=NPZ)
    ap.add_argument("--zoom", type=int, default=128)
    ap.add_argument("--model", default="multinomial",
                    help="model used for the 'counts' expected profile")
    ap.add_argument("--yscale", choices=["auto", "flank", "full"], default="auto",
                    help="auto: flank-scaled when wide, full when the motif dominates")
    ap.add_argument("--models", default="all",
                    help="comma-separated models to overlay, or 'all'")
    ap.add_argument("--mask", type=int, default=0,
                    help="mask +-N bp around the motif centre (excluded from the "
                         "plot, from smoothing and from the y-scale)")
    ap.add_argument("--smooth", type=int, default=0,
                    help="smoothing window in bp (0 = none); uses the library's "
                         "smooth1d so smoothing matches the rest of the codebase")
    ap.add_argument("--tracks", default="all",
                    help="comma-separated subset of "
                         "short_first,short_last,mono_first,mono_last (or 'all')")
    ap.add_argument("--mode", choices=["shape", "counts"], default="shape",
                    help="shape: observed vs predicted probability shape; "
                         "counts: raw vs expected counts")
    ap.add_argument("--outdir",
                    default="/home/nathanboley/src/fragmentomics_tools/data/ctcf_plots")
    args = ap.parse_args()

    d = np.load(args.npz, allow_pickle=True)
    pos = d["positions"]
    ti = track_indices(d["tracks"])
    samples = [str(s) for s in d["sample_names"]]

    show = SHOW if args.tracks == "all" else [t.strip() for t in args.tracks.split(",")]
    bad = [t for t in show if t not in SHOW]
    if bad:
        raise SystemExit(f"unknown track(s) {bad}; choose from {SHOW}")

    sel = np.abs(pos) <= args.zoom
    zpos = pos[sel]
    # synthetic region: meta-profile coordinates, motif centre at index `zoom`
    region = Region(chrom="meta", start=0, stop=int(sel.sum()), strand=".")
    centre = int(np.where(zpos == 0)[0][0])
    vlines = [
        VLine(x=centre, color="k", alpha=0.8, linestyle=":"),
        VLine(x=centre + MOTIF_LO, color="steelblue", alpha=0.7, linestyle="--"),
        VLine(x=centre + MOTIF_HI, color="steelblue", alpha=0.7, linestyle="--"),
    ]

    unc = sum(d[f"unc__{s}"] for s in samples) / len(samples)
    all_models = [m for m in (str(x) for x in d["model_names"]) if m != "uniform"]
    models = all_models if args.models == "all" else args.models.split(",")
    # one distinct colour per model; observed is always black
    palette = ["crimson", "tab:blue", "tab:green", "tab:orange", "tab:purple"]
    mcolors = {m: palette[k % len(palette)] for k, m in enumerate(models)}

    tracks = Tracks()
    for strand in ["+", "-"]:
        for name in show:
            i = ti[(strand, name)]
            if args.mode == "shape":
                obs = unc[i] / unc[i].sum()
                preds = {m: d[f"probs__{m}"][i] / d[f"probs__{m}"][i].sum()
                         for m in models}
                ylab = "shape"
            else:
                obs = unc[i]
                preds = {
                    m: sum(d[f"exp__{m}__{s}"] for s in samples) / len(samples)
                    for m in models
                }
                preds = {m: v[i] for m, v in preds.items()}
                ylab = "counts"
            full = [obs] + [preds[m] for m in models]
            if args.mask > 0:
                # mask-aware: exclude the motif region BEFORE smoothing so it
                # cannot bleed into the neighbouring window
                series = [masked_smooth(a, pos, args.mask, args.smooth)[sel]
                          for a in full]
            elif args.smooth > 1:
                # smooth the FULL profile then crop, so the window is not
                # distorted by the panel edges
                series = [smooth1d(a, window_len=args.smooth)[sel] for a in full]
            else:
                series = [a[sel] for a in full]
            ymin, ymax = compute_ylim(series, zpos, mode=args.yscale)
            tracks.append(
                VectorTracks(
                    input=series,
                    region=region,
                    name=f"{strand} {name.replace('_', ' ')}  ({ylab})",
                    labels=["observed"] + [f"pred: {m}" for m in models],
                    colors=["0.15"] + [mcolors[m] for m in models],
                    alphas=[1.0] + [0.8] * len(models),
                    ymin=ymin, ymax=ymax,
                    vlines=vlines,
                    height=2,
                )
            )

    motif_frac = float((np.abs(zpos) <= MOTIF_HI).mean())
    yscale_used = (args.yscale if args.yscale != "auto"
                   else ("full" if motif_frac > 0.15 else "flank"))

    os.makedirs(args.outdir, exist_ok=True)
    out = os.path.join(args.outdir,
                       f"tracks_zoom{args.zoom}__{args.mode}"
                       f"__{'-'.join(models) if args.models != 'all' else 'allmodels'}"
                       f"{'' if args.tracks == 'all' else '__' + '-'.join(show)}"
                       f"{'' if args.smooth <= 1 else f'__smooth{args.smooth}'}"
                       f"{'' if args.mask <= 0 else f'__mask{args.mask}'}.png")
    fig, axes = tracks.plot(
        width=16,
        height_multiplier=1.1,
        title=(f"CTCF meta-profile +-{args.zoom}bp — observed vs predicted ({args.mode})\n"
               f"dashed = 17bp motif [-8,+8]; y-scale={yscale_used}; "
               f"smoothing={'none' if args.smooth <= 1 else str(args.smooth) + 'bp'}"
               f"{'' if args.mask <= 0 else f'; MASKED +-{args.mask}bp'}  "
               f"(n_sites={int(d['n_sites'])}, n_samples={len(samples)})"),
    )
    # The meta-profile is in coordinates relative to the motif centre, but the
    # synthetic region is 0-based; relabel the x ticks back to relative bp.
    # `axes` is the list of main axes, one per track.  VectorTracks does not set
    # xlim, and Tracks.plot strips x ticks from every panel except the bottom one.
    n = int(sel.sum())
    ticks = np.linspace(0, n - 1, 9)
    labels = [f"{int(round(t - centre)):+d}" for t in ticks]
    for j, ax in enumerate(axes):
        ax.set_xlim(0, n - 1)
        if j == len(axes) - 1:                   # bottom panel keeps the ticks
            ax.set_xticks(ticks)
            ax.set_xticklabels(labels)
            ax.set_xlabel("position relative to motif centre (oriented, bp)")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
