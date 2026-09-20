"""CTCF motif-aligned endpoint pileups: UNCORRECTED vs CORRECTED vs EXPECTED,
across the three trained background models + a uniform-forced control (Phase 4).

Per (model, sample) motif-aligned pileups over +-AGG_HALF bp around the motif
CENTER for the 12 canonical tracks:
  1. UNCORRECTED : raw endpoint counts per position, summed across sites.
  2. CORRECTED   : endpoint counts weighted by apply_fragment_weights
                   (both identity and bounded [0.2, 5.0] clamps).
  3. EXPECTED    : expected_profile (N_w * probs) + the shape (probs) itself.
Plus a uniform-forced CONTROL model (shape_head zeroed) whose identity-clamp
corrected pileup MUST equal the uncorrected pileup.

BINDING contracts honoured (docs/pending/correction_outputs_design.md):
  - strandless query regions (strand '.'); orientation happens HERE, at the
    aggregation layer: for a '-' motif reverse the position axis about the
    center AND permute tracks via reverse_complement_track_permutation (§0.3, F1).
  - single, untrimmed, grid-aligned window per site: region is exactly one TILE
    (16384 bp), motif center at local 8192, so there is exactly one window and no
    seam / no trim (§3.3, §4 HARD REQUIREMENT).
  - clamp is a required arg; both identity() and bounded [0.2, 5.0] are run and
    the raw (identity) weight distribution is reported per model (§5.2, §7).

Efficiency: model probs depend only on sequence, so per-(site, model) window
predictions are cached (via inference._predict_window) and reused across all
samples/clamps/interfaces; fragments are loaded ONCE per (site, sample) and
reused across all models.
"""

import argparse
import json
import sys
import time

import numpy as np
import pysam

sys.path.insert(0, "/home/nathanboley/src/fragmentomics_tools")

import torch  # noqa: E402

from background_model_core import (  # noqa: E402
    BackgroundModel,
    DEFAULT_OUTPUT_TRACKS,
    reverse_complement_track_permutation,
)
from background_model.preprocess import TRACK_INDEX, FL_BANDS  # noqa: E402
from background_model import inference as _inf  # noqa: E402
from background_model.correction import (  # noqa: E402
    apply_fragment_weights,
    expected_profile,
    WeightClampConfig,
)
from fragmentomics_tools.contig import CONTIG_LENGTHS  # noqa: E402
from fragmentomics_tools.dataframe import RegionDataFrame  # noqa: E402
from fragmentomics_tools.fragment_array.fragment_array import (  # noqa: E402
    RegionFragmentArray,
)
from fragmentomics_tools.region import Region  # noqa: E402
from fragments_h5 import FragmentsH5  # noqa: E402

# ---- fixed inputs -------------------------------------------------------
FASTA = "/efs/analytics/nathanboley/data_resources/genome/hg38.fa"
BLACKLIST_BED = (
    "/home/nathanboley/src/fragmentomics_tools/data/region_sets/"
    "exclusion_4_blacklist_encode_v2.bed"
)
RUNS = "/efs/analytics/nathanboley/background_model/runs"
CKPTS = {
    "multinomial": f"{RUNS}/bakeoff_multinomial_20260918/checkpoints/"
    "epoch=4-step=16435-val_loss=9.3243.ckpt",
    "dirichlet_multinomial": f"{RUNS}/bakeoff_dirichlet_multinomial_20260918/"
    "checkpoints/epoch=4-step=16435-val_loss=9.3191.ckpt",
    "nb_offset": f"{RUNS}/bakeoff_nb_offset_20260918/checkpoints/"
    "epoch=4-step=16435-val_loss=5.3247.ckpt",
}
SAMPLES = {
    "BDS-159060_prod-Lib1": "/efs/analytics/nathanboley/biomarker-projects/"
    "data_cache/DC4-17506/c96e13325defe7963255ef902e465fac-28-"
    "BDS-159060_prod-Lib1.hg38.fragments.h5",
    "RD-50548-Lib1": "/efs/analytics/nathanboley/biomarker-projects/data_cache/"
    "NC-13909/b0315d6c52ac7f4f8744b4dafb6661ca-63-RD-50548-Lib1.hg38.fragments.h5",
    "RD-50551-Lib1": "/efs/analytics/nathanboley/biomarker-projects/data_cache/"
    "NC-13909/e4172484cb6b3379bf6ed14405dc2f58-138-RD-50551-Lib1.hg38.fragments.h5",
}
MIN_MAPQ = 10
MAX_FRAG_LEN = 175  # config.max_frag_len = max(hi over fl_bands)
BLACKLIST_EXPANSION = 120

# weight-distribution histogram (identity clamp), log-spaced
WBINS = np.logspace(-4, 6, 2001)
CLIP_LO, CLIP_HI = 0.2, 5.0

# ---- window prediction cache (probs are sample-independent) --------------
_orig_predict_window = _inf._predict_window
_pw_cache = {}


def _cached_predict_window(model, fasta, contig, win_start, win_stop, geom,
                           bl, ble, cl):
    key = (id(model), contig, int(win_start))
    hit = _pw_cache.get(key)
    if hit is None:
        hit = _orig_predict_window(
            model, fasta, contig, win_start, win_stop, geom, bl, ble, cl
        )
        _pw_cache[key] = hit
    return hit


_inf._predict_window = _cached_predict_window


def counts_matrix(rfa, tile):
    """(12, tile) dense per-track counts (weighted sum) from an rfa."""
    cov = rfa.build_coverage_counts(
        fl_bands=list(FL_BANDS), split_strand=True, return_sparse=False
    )
    M = np.zeros((12, tile), dtype=np.float64)
    for key, vec in cov.items():
        strand, fl, covtype = key
        M[TRACK_INDEX[(strand, fl, covtype)]] = vec
    return M


def orient_slice(M, strand, center_local, half, perm):
    """Motif-oriented (12, 2*half+1) slice about center_local.

    '+': out[c, half+d] = M[c, center_local+d]
    '-': reverse position about center AND permute tracks (RC).
    """
    lo, hi = center_local - half, center_local + half + 1
    sl = M[:, lo:hi]
    if strand == "-":
        sl = sl[perm][:, ::-1]
    return sl


def load_model(ckpt, device, uniform=False):
    m = BackgroundModel.load_from_checkpoint(ckpt, map_location=device)
    if uniform:
        with torch.no_grad():
            m.shape_head.weight.zero_()
            m.shape_head.bias.zero_()
    m.to(device)
    m.eval()
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--sites", required=True)
    ap.add_argument("--limit-sites", type=int, default=0,
                    help="0 = all; else first N sites (smoke)")
    ap.add_argument("--samples-limit", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() or args.device == "cpu" \
        else "cpu"
    print(f"device={device} cuda_avail={torch.cuda.is_available()}", flush=True)

    S = np.load(args.sites, allow_pickle=False)
    contigs = S["contig"].astype(str)
    centers = S["center"].astype(int)
    strands = S["strand"].astype(str)
    tile = int(S["TILE"])
    half = int(S["AGG_HALF"])
    if args.limit_sites:
        contigs = contigs[: args.limit_sites]
        centers = centers[: args.limit_sites]
        strands = strands[: args.limit_sites]
    n_sites = len(centers)

    sample_items = list(SAMPLES.items())
    if args.samples_limit:
        sample_items = sample_items[: args.samples_limit]
    sample_names = [s for s, _ in sample_items]

    perm = np.array(reverse_complement_track_permutation(list(DEFAULT_OUTPUT_TRACKS)))
    hg = CONTIG_LENGTHS["hg38"]
    fasta = pysam.FastaFile(FASTA)
    blacklist_rdf = RegionDataFrame.from_bed(BLACKLIST_BED, ref="hg38")

    # models: uniform control + 3 trained
    models = {"uniform": load_model(CKPTS["multinomial"], device, uniform=True)}
    for name, ck in CKPTS.items():
        models[name] = load_model(ck, device)
    model_names = list(models.keys())

    clamps = {"identity": WeightClampConfig.identity(),
              "bounded_0.2_5.0": WeightClampConfig(CLIP_LO, CLIP_HI)}

    W = 2 * half + 1
    zc = lambda: np.zeros((12, W), dtype=np.float64)  # noqa: E731
    # accumulators
    unc = {s: zc() for s in sample_names}
    unc_plus = {s: zc() for s in sample_names}
    unc_minus = {s: zc() for s in sample_names}
    corr = {(m, c, s): zc()
            for m in model_names for c in clamps for s in sample_names}
    exp = {(m, s): zc() for m in model_names for s in sample_names}
    probs_sum = {m: zc() for m in model_names}  # sample-independent shape
    n_plus = int((strands == "+").sum())
    n_minus = int((strands == "-").sum())

    # weight-distribution accumulators (identity clamp)
    whist = {m: np.zeros(len(WBINS) - 1, dtype=np.int64) for m in model_names}
    wmin = {m: np.inf for m in model_names}
    wmax = {m: 0.0 for m in model_names}
    wtot = {m: 0 for m in model_names}
    wlo = {m: 0 for m in model_names}
    whi = {m: 0 for m in model_names}

    fh = {s: FragmentsH5(p, cache_pointers=False) for s, p in sample_items}
    t0 = time.time()
    n_ok = 0
    for i in range(n_sites):
        contig, center, strand = contigs[i], int(centers[i]), strands[i]
        start = center - tile // 2
        stop = start + tile
        cl = hg[contig]
        region = Region(chrom=contig, start=start, stop=stop, strand=".")
        center_local = center - start  # == tile//2

        # load fragments once per (site, sample); reuse across models
        raw = {}
        rawM = {}
        for s in sample_names:
            rfa = RegionFragmentArray.from_fragments_h5(
                fh[s], region, min_mapq=MIN_MAPQ, max_frag_len=MAX_FRAG_LEN,
            ).drop_duplicate_fragments()
            raw[s] = rfa
            rawM[s] = counts_matrix(rfa, tile)
            o = orient_slice(rawM[s], strand, center_local, half, perm)
            unc[s] += o
            if strand == "+":
                unc_plus[s] += o
            else:
                unc_minus[s] += o

        for m in model_names:
            model = models[m]
            # EXPECTED (probs sample-independent -> accumulate once; expected
            # per sample via that sample's observed N)
            for si, s in enumerate(sample_names):
                ep = expected_profile(
                    model, fasta, contig, start, stop, rawM[s],
                    blacklist_rdf=blacklist_rdf,
                    blacklist_expansion=BLACKLIST_EXPANSION, contig_len=cl,
                )
                exp_o = orient_slice(np.nan_to_num(ep.expected, nan=0.0),
                                     strand, center_local, half, perm)
                exp[(m, s)] += exp_o
                if si == 0:
                    probs_sum[m] += orient_slice(
                        ep.probs, strand, center_local, half, perm)

            # CORRECTED (both clamps)
            for cname, clamp in clamps.items():
                for s in sample_names:
                    cr = apply_fragment_weights(
                        raw[s], model, fasta, clamp=clamp,
                        blacklist_rdf=blacklist_rdf,
                        blacklist_expansion=BLACKLIST_EXPANSION, contig_len=cl,
                        drop_uncorrectable=True,
                    )
                    cM = counts_matrix(cr, tile)
                    corr[(m, cname, s)] += orient_slice(
                        cM, strand, center_local, half, perm)
                    # weight distribution from identity clamp only
                    if cname == "identity":
                        for attr in ("first_covered_base_weights",
                                     "last_covered_base_weights", "weights"):
                            w = getattr(cr, attr)
                            w = w[w > 1e-12]
                            if w.size:
                                whist[m] += np.histogram(w, bins=WBINS)[0]
                                wmin[m] = min(wmin[m], float(w.min()))
                                wmax[m] = max(wmax[m], float(w.max()))
                                wtot[m] += int(w.size)
                                wlo[m] += int((w < CLIP_LO).sum())
                                whi[m] += int((w > CLIP_HI).sum())
        n_ok += 1
        _pw_cache.clear()  # bound memory: predictions reused within a site only
        if (i + 1) % 100 == 0:
            dt = time.time() - t0
            print(f"[{i+1}/{n_sites}] {dt:.1f}s "
                  f"({dt/(i+1)*1000:.1f} ms/site)", flush=True)

    for s in fh.values():
        s.close()

    # ---- weight stats ---------------------------------------------------
    def quantile_from_hist(h, q):
        c = np.cumsum(h)
        if c[-1] == 0:
            return float("nan")
        target = q * c[-1]
        k = int(np.searchsorted(c, target))
        k = min(k, len(WBINS) - 2)
        return float(np.sqrt(WBINS[k] * WBINS[k + 1]))  # geo-mid of bin

    wstats = {}
    for m in model_names:
        tot = wtot[m]
        wstats[m] = {
            "n_nonzero_weights": tot,
            "min": (wmin[m] if tot else None),
            "median": quantile_from_hist(whist[m], 0.5) if tot else None,
            "p99": quantile_from_hist(whist[m], 0.99) if tot else None,
            "max": (wmax[m] if tot else None),
            "frac_below_0.2": (wlo[m] / tot if tot else None),
            "frac_above_5.0": (whi[m] / tot if tot else None),
            "frac_clipped_by_bounded": ((wlo[m] + whi[m]) / tot if tot else None),
        }

    # ---- uniform control check (central-zone equality) ------------------
    # corrected(uniform, identity) must equal uncorrected across all tracks.
    ctrl_maxabs = 0.0
    for s in sample_names:
        d = np.abs(corr[("uniform", "identity", s)] - unc[s])
        ctrl_maxabs = max(ctrl_maxabs, float(d.max()))
    ctrl_pass = bool(ctrl_maxabs < 1e-6)

    # ---- mirror check (oriented +-only vs --only shapes agree) ----------
    # sum over samples, use short-first track (0) + mono-first (3) as probes;
    # report normalized-shape correlation of total endpoints per position.
    def norm_shape(a):
        t = a.sum(axis=0)  # sum over tracks -> (W,)
        s = t.sum()
        return t / s if s > 0 else t
    up = sum(unc_plus[s] for s in sample_names)
    um = sum(unc_minus[s] for s in sample_names)
    mp, mm = norm_shape(up), norm_shape(um)
    mirror_corr = float(np.corrcoef(mp, mm)[0, 1])
    # smoothed (21 bp) correlation — robust to per-bp counting noise
    _k = np.ones(21) / 21.0
    mirror_corr_smooth = float(np.corrcoef(
        np.convolve(mp, _k, mode="same"), np.convolve(mm, _k, mode="same"))[0, 1])
    # control: reversing the minus set again should be WORSE if orientation ok
    mmr = norm_shape(um[:, ::-1])
    mirror_corr_reversed = float(np.corrcoef(mp, mmr)[0, 1])

    # ---- save -----------------------------------------------------------
    save = {
        "tracks": np.array(DEFAULT_OUTPUT_TRACKS),
        "positions": np.arange(-half, half + 1),
        "n_sites": np.int64(n_ok),
        "n_plus": np.int64(n_plus),
        "n_minus": np.int64(n_minus),
        "sample_names": np.array(sample_names),
        "model_names": np.array(model_names),
        "clamp_names": np.array(list(clamps.keys())),
    }
    for s in sample_names:
        save[f"unc__{s}"] = unc[s]
        save[f"uncplus__{s}"] = unc_plus[s]
        save[f"uncminus__{s}"] = unc_minus[s]
    for m in model_names:
        save[f"probs__{m}"] = probs_sum[m]
        for s in sample_names:
            save[f"exp__{m}__{s}"] = exp[(m, s)]
            for c in clamps:
                save[f"corr__{m}__{c}__{s}"] = corr[(m, c, s)]
    np.savez_compressed(f"{args.out}/pileups.npz", **save)

    meta = {
        "n_sites": n_ok, "n_plus": n_plus, "n_minus": n_minus,
        "n_samples": len(sample_names), "sample_names": sample_names,
        "model_names": model_names, "clamps": list(clamps.keys()),
        "weight_stats": wstats,
        "uniform_control_maxabs_diff": ctrl_maxabs,
        "uniform_control_pass": ctrl_pass,
        "mirror_plus_minus_corr": mirror_corr,
        "mirror_plus_minus_corr_smooth21": mirror_corr_smooth,
        "mirror_plus_minus_corr_reversed": mirror_corr_reversed,
        "elapsed_s": time.time() - t0,
        "min_mapq": MIN_MAPQ, "max_frag_len": MAX_FRAG_LEN,
        "blacklist_expansion": BLACKLIST_EXPANSION,
        "tile": tile, "agg_half": half,
    }
    with open(f"{args.out}/run_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
