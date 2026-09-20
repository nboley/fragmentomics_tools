"""Side-by-side comparison: BLOOD (accessible) vs NO-BLOOD (control) CTCF pileups.

The discriminating question (Phase 4 control):
  - If the model's correction is real sequence-bias removal, then at motifs NOT
    accessible in blood there is no biological footprint, the model still predicts
    its sequence-bias dip/helical structure, and the corrected profile stays flat
    (the correction does NOT manufacture a peak).
  - If the correction is an artifact, dividing the (flat) control raw data by the
    model's predicted dip manufactures a spurious peak in the corrected profile at
    the same [-50,-10] short-band location where the blood run showed amplification.

This reads both pileups.npz and reports, for the short band (40-65bp endpoints,
summed over first/last and both strands) and mono band (120-175bp):
  * raw footprint amplitude   = mean(raw   in FP) / mean(raw   in FLANK)
  * predicted footprint level = mean(probs in FP) / mean(probs in FLANK)
  * corrected footprint amp.  = mean(corr  in FP) / mean(corr  in FLANK)
  * correction factor in FP   = mean(corr in FP)  / mean(raw in FP)     (the "1.78x")
  * correction factor in FLANK
Plus the flank helical periodicity (FFT dominant period in the 5-20bp band, folded
flanks, |pos|>=150) of both the raw short band and the model's predicted short-band
shape -- it should stay ~10.2bp if the model behaves consistently on control sequence.

Writes a metrics JSON and a side-by-side comparison PNG.  No GPU.
"""
import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLOOD = "/efs/analytics/nathanboley/background_model/ctcf_pileup/pileups.npz"
NOBLOOD = "/efs/analytics/nathanboley/background_model/ctcf_pileup_noblood/pileups.npz"

FP_LO, FP_HI = -50, -10          # short-band footprint band (per task)
FLANK_LO, FLANK_HI = 150, 500    # reference flank band (|pos| in [150,500])
HELIX_MINP, HELIX_MAXP = 5.0, 20.0  # helical period search band (bp)
HELIX_FLANK_MIN = 150            # |pos| >= 150 for the FFT


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


def band_profile(arr12, ti, band):
    """Sum the 4 (strand x first/last) tracks of a band -> (W,) endpoint profile."""
    keys = [("+", f"{band}_first"), ("+", f"{band}_last"),
            ("-", f"{band}_first"), ("-", f"{band}_last")]
    return sum(arr12[ti[k]] for k in keys)


def mean_in(profile, pos, lo, hi):
    m = (pos >= lo) & (pos <= hi)
    return float(profile[m].mean())


def flank_mean(profile, pos, lo, hi):
    m = ((pos >= lo) & (pos <= hi)) | ((pos <= -lo) & (pos >= -hi))
    return float(profile[m].mean())


def fp_amplitude(profile, pos):
    fp = mean_in(profile, pos, FP_LO, FP_HI)
    fl = flank_mean(profile, pos, FLANK_LO, FLANK_HI)
    return fp, fl, (fp / fl if fl else float("nan"))


def helical_period(profile, pos):
    """Helical diagnostics on folded flanks (|pos|>=150), detrended.

    Returns dict: dominant period (bp) in [5,20], the relative power at 10.2bp
    (normalised to the in-band max), and the relative oscillation amplitude
    (std of detrended residual / mean level) -- a confidence proxy: a weak,
    diffuse flank oscillation makes the dominant-period pick unreliable.
    """
    m = np.abs(pos) >= HELIX_FLANK_MIN
    p = pos[m]
    v = profile[m].astype(float)
    # fold left+right flank about center onto |pos| axis, average duplicates
    ap = np.abs(p)
    uniq = np.unique(ap)
    folded = np.array([v[ap == u].mean() for u in uniq])
    x = uniq.astype(float)
    # detrend (remove linear trend)
    resid = folded - np.polyval(np.polyfit(x, folded, 1), x)
    resid = resid - resid.mean()
    osc_amp = float(np.std(resid) / np.mean(folded)) if np.mean(folded) else float("nan")
    resid = resid * np.hanning(len(resid))
    # zero-pad for frequency resolution; unit spacing = 1 bp
    n = 1 << (int(np.log2(len(resid))) + 4)
    spec = np.abs(np.fft.rfft(resid, n=n))
    freqs = np.fft.rfftfreq(n, d=1.0)
    with np.errstate(divide="ignore"):
        periods = 1.0 / freqs
    band = (periods >= HELIX_MINP) & (periods <= HELIX_MAXP)
    if not band.any():
        return {"dominant_period_bp": float("nan"),
                "relpower_at_10.2bp": float("nan"), "osc_amp_rel": osc_amp}
    P, S = periods[band], spec[band]
    dom_period = float(P[np.argmax(S)])
    relpow_102 = float(S[np.argmin(np.abs(P - 10.2))] / S.max())
    return {"dominant_period_bp": dom_period,
            "relpower_at_10.2bp": relpow_102, "osc_amp_rel": osc_amp}


def analyze(npz_path, model, clamp):
    d = np.load(npz_path, allow_pickle=True)
    pos = d["positions"]
    ti = track_indices(d["tracks"])
    samples = [str(s) for s in d["sample_names"]]
    unc = sum(d[f"unc__{s}"] for s in samples) / len(samples)
    cor = sum(d[f"corr__{model}__{clamp}__{s}"] for s in samples) / len(samples)
    probs = d[f"probs__{model}"]

    out = {"n_sites": int(d["n_sites"]),
           "n_plus": int(d["n_plus"]), "n_minus": int(d["n_minus"])}
    curves = {"pos": pos}
    for band in ("short", "mono"):
        raw_p = band_profile(unc, ti, band)
        cor_p = band_profile(cor, ti, band)
        prb_p = band_profile(probs, ti, band)
        raw_fp, raw_fl, raw_amp = fp_amplitude(raw_p, pos)
        cor_fp, cor_fl, cor_amp = fp_amplitude(cor_p, pos)
        prb_fp, prb_fl, prb_amp = fp_amplitude(prb_p, pos)
        out[band] = {
            "raw_fp_mean": raw_fp, "raw_flank_mean": raw_fl,
            "raw_fp_amplitude": raw_amp,
            "predicted_fp_amplitude": prb_amp,
            "corrected_fp_mean": cor_fp, "corrected_flank_mean": cor_fl,
            "corrected_fp_amplitude": cor_amp,
            "correction_factor_fp": (cor_fp / raw_fp if raw_fp else float("nan")),
            "correction_factor_flank": (cor_fl / raw_fl if raw_fl else float("nan")),
        }
        if band == "short":
            out["helical_raw_short"] = helical_period(raw_p, pos)
            out["helical_predicted_short"] = helical_period(prb_p, pos)
        curves[f"{band}_raw"] = raw_p
        curves[f"{band}_cor"] = cor_p
        curves[f"{band}_prb"] = prb_p
    return out, curves


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blood", default=BLOOD)
    ap.add_argument("--noblood", default=NOBLOOD)
    ap.add_argument("--model", default="multinomial")
    ap.add_argument("--clamp", default="bounded_0.2_5.0")
    ap.add_argument("--zoom", type=int, default=128)
    ap.add_argument("--outdir",
                    default="/efs/analytics/nathanboley/background_model/ctcf_pileup_noblood")
    args = ap.parse_args()

    blood, bc = analyze(args.blood, args.model, args.clamp)
    noblood, nc = analyze(args.noblood, args.model, args.clamp)

    metrics = {"model": args.model, "clamp": args.clamp,
               "footprint_band": [FP_LO, FP_HI],
               "flank_band": [FLANK_LO, FLANK_HI],
               "blood": blood, "noblood": noblood}

    os.makedirs(args.outdir, exist_ok=True)
    with open(os.path.join(args.outdir, "compare_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # ---- side-by-side figure: raw / predicted-shape / corrected ----------
    fig, ax = plt.subplots(2, 3, figsize=(19, 9))
    datasets = [("BLOOD (accessible)", bc, "tab:blue"),
                ("NO-BLOOD (control)", nc, "tab:red")]
    for r, band in enumerate(("short", "mono")):
        pos = bc["pos"]
        sel = np.abs(pos) <= args.zoom
        x = pos[sel]
        # normalise each curve to its flank mean so shapes are comparable
        def norm(p):
            fl = flank_mean(p, pos, FLANK_LO, FLANK_HI)
            return p / fl if fl else p
        # panel col 0: raw
        a = ax[r, 0]
        for name, c, col in datasets:
            a.plot(x, norm(c[f"{band}_raw"])[sel], color=col, lw=1.4, label=name)
        a.set_title(f"{band} band — RAW (flank-normalised)")
        # panel col 1: predicted shape
        a = ax[r, 1]
        for name, c, col in datasets:
            a.plot(x, norm(c[f"{band}_prb"])[sel], color=col, lw=1.4, label=name)
        a.set_title(f"{band} band — PREDICTED shape (flank-normalised)")
        # panel col 2: corrected
        a = ax[r, 2]
        for name, c, col in datasets:
            a.plot(x, norm(c[f"{band}_cor"])[sel], color=col, lw=1.4, label=name)
        a.set_title(f"{band} band — CORRECTED (flank-normalised)")
        for cidx in range(3):
            a = ax[r, cidx]
            a.axvspan(FP_LO, FP_HI, color="gold", alpha=0.20, zorder=0,
                      label="footprint [-50,-10]")
            a.axvline(0, color="k", ls=":", lw=0.8)
            a.axhline(1.0, color="0.6", ls="-", lw=0.6)
            a.grid(alpha=0.3)
            if r == 1:
                a.set_xlabel("position rel. motif centre (oriented, bp)")
            if (r, cidx) == (0, 0):
                a.legend(fontsize=8)
    fig.suptitle(
        f"BLOOD vs NO-BLOOD CTCF meta-profile — {args.model} — clamp={args.clamp}\n"
        "gold = short-band footprint [-50,-10]; each curve normalised to its own flank mean"
    )
    fig.tight_layout()
    p = os.path.join(args.outdir, "compare_blood_vs_noblood.png")
    fig.savefig(p, dpi=130)
    plt.close(fig)

    print(json.dumps(metrics, indent=2))
    print("wrote", p)
    print("wrote", os.path.join(args.outdir, "compare_metrics.json"))


if __name__ == "__main__":
    main()
