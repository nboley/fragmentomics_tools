#!/usr/bin/env python
"""cfDNA fragment simulator for the simulation study (docs/pending/simulation_study_plan.md).

SIMULATOR + VALIDATION only -- this script emits a fragment set with a KNOWN
cut-site + GC/length bias and then verifies the bias was actually imposed.  It
does NOT train any model.

Two ground-truth bias mechanisms act on every proposed fragment and MULTIPLY
(plan sec 3 / sec 3.1):

1. Hexamer cut-site bias ``w6`` -- a synthetic 4096-entry log-normal weight
   table (seeded, normalised to max 1).  It is applied at BOTH ends of a
   fragment; the FAR (right) end uses the REVERSE-COMPLEMENT hexamer because it
   is a cut on the opposite strand (plan sec 3 step 3).  ``w6`` is structurally
   unlike the dilated-CNN estimator, so recovery is a real test.

2. GC x LENGTH bias -- a 2-D SURFACE (``bias_grid_2d_row_centred`` from
   ``gc_bias_grid.json``, fit by scripts/sim_fit_gc_bias.py from 30 real SPARK
   samples).  This grid is ALREADY the bias (= exp(+cf) = 1/production_weight);
   it is NOT inverted again here (sign convention, plan sec 3.1).  The GC
   dependence INTERACTS with length and FLIPS SIGN: the GC slope (70%/30%) runs
   0.654 (24bp, GC-disfavouring) -> 4.080 (75bp, strongly GC-favouring).  It is
   NOT collapsed to a 1-D GC marginal.  Interpolation is bilinear within the
   grid; extrapolation HOLDS at the edges (L<24 uses the 24bp row, L>75 uses the
   75bp row, GC clamps to [30,70]) -- exactly the ``extrapolation_policy``
   recorded in the JSON.  There is no data above 75bp and the slope is still
   rising, so extrapolating would invent unsupported bias.

Generative procedure, per region, propose-and-reject (plan sec 3):
  1. start ~ Uniform(region); length ~ empirical cfDNA length distribution taken
     from a REAL held-out h5 (FragmentsH5.fragment_length_counts).
  2. left hexamer (forward, 3-in/3-out spanning the cut) and right hexamer
     (reverse-complement).
  3. GC over the WHOLE fragment, in percent.
  4. accept w.p.  w6(left) * w6(right) * gc_bias(length, gc), normalised so the
     max achievable acceptance weight is 1 (per-sample normaliser, because
     regime-B jitter can push a w6 entry above 1).
  5. per-(sample, region) target counts sampled independently from the
     empirical per-tile count distribution of either the production zarr store
     (--real-store, recommended) or the heldout h5.

Regimes (plan sec 4):
  A  every sample shares one bias table.
  B  per-sample log-normal jitter of ``w6``:  w6_s(h) = w6(h) * exp(eps_{s,h}),
     eps ~ Normal(0, sd=0.286) i.i.d. per (sample, hexamer).  sd is the
     EMPIRICAL median across-sample SD of log-bias from the SPARK fit.  The true
     per-sample jitter is recorded as ground truth for later dispersion scoring.

Paired design (plan sec 5): one fragment set per regime, deterministic given the
seed, shared by every downstream model.

Output layout  <out-root>/<regime>/ :
  sample_<i>.npz     region_idx, start, stop, strand  (start/stop REGION-LOCAL,
                     0-based half-open, valid RegionFragmentArray coords)
  region_table.npz   contig, gstart, gstop, region_len  (shared across samples)
  ground_truth.npz   w6, bias surface used, per-sample jitter (regime B), targets,
                     length pmf, all params, seed
  ground_truth.json  scalar params + realised w6 stats + paths
  plots/*.png        the four validation-gate figures (also copied to
                     data/sim_plots/ for editor viewing)

Representation note (verified against background_model/preprocess.py and the
``_rfa`` helper in tests/test_bg_correction.py): a fragment as
(region_idx, region-local start, region-local stop, strand) plus a region table
is sufficient to (a) build store coverage counts and (b) construct
RegionFragmentArray objects for apply_fragment_weights.  Preprocess builds an
RFA from exactly (starts_0, stops_0, region, max_frag_len, fragment_strands) and
calls build_coverage_counts(fl_bands, split_strand=True, return_sparse=True);
apply_fragment_weights requires a strandless ('.'/None), non-flipped region --
both hold here (regions are Region(strand='.') and fragments are never flipped).
tests/test_sim_fragments.py exercises both round-trips on the emitted npz.

Env: /home/nathanboley/miniconda3/envs/biomarker_env/bin/python
Run: PYTHONPATH=. .../python scripts/sim_fragments.py --regime A --validate
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time

import numpy as np

# ── paths (on-host, matching scripts/build_inactive_regions.py + sim_fit_gc_bias.py) ──
REPO = "/home/nathanboley/src/fragmentomics_tools"
FASTA = "/efs/analytics/nathanboley/data_resources/genome/hg38.fa"
GC_BIAS_JSON = ("/efs/analytics/nathanboley/background_model/simulation/"
                "gc_bias_grid.json")
TRAINING_TILES = os.path.join(REPO, "data", "region_sets", "training_tiles.bed")
OUT_ROOT = "/efs/analytics/nathanboley/background_model/simulation"
SIM_PLOTS = os.path.join(REPO, "data", "sim_plots")  # gitignored editor copy
# held-out (role=1) library from data/sample_sheets/draw50.tsv -- a normal cfDNA
# length profile (median ~128bp, 46% in the 120-175 band) so both fl bands are
# well populated for the per-length-stratum validation.
DEFAULT_HELDOUT_H5 = ("/efs/analytics/nathanboley/biomarker-projects/data_cache/"
                      "NC-13909/b0315d6c52ac7f4f8744b4dafb6661ca-63-"
                      "RD-50548-Lib1.hg38.fragments.h5")

# ── geometry / knobs ──────────────────────────────────────────────────────
REGION_LEN = 2_048        # plan sec 4: ~2,048 bp tiles
N_REGIONS = 1_000         # plan sec 4: 1,000 real regions
HEX_HALF = 3              # 3-in / 3-out hexamer (plan sec 9): 6 bp spanning the cut
KMER = 2 * HEX_HALF       # 6
NHEX = 4 ** KMER          # 4096
MAX_LEN = 500             # cap the empirical length draw (>99% of real mass;
                          # drops the noisy multi-kb tail that never fits a 2kb region)
JITTER_SD = 0.286         # regime B: EMPIRICAL across-sample SD of log-bias (SPARK fit)

# base -> 2-bit code; anything else (N) -> 255 sentinel
_BASE_LUT = np.full(256, 255, dtype=np.uint8)
_BASE_LUT[ord("A")] = 0
_BASE_LUT[ord("C")] = 1
_BASE_LUT[ord("G")] = 2
_BASE_LUT[ord("T")] = 3
_POW = (4 ** np.arange(KMER - 1, -1, -1)).astype(np.int64)  # big-endian weights


# ── hexamer machinery ─────────────────────────────────────────────────────

def rc_hexamer_permutation() -> np.ndarray:
    """Permutation p such that p[i] = index of the reverse-complement of hexamer i.

    Complement in 2-bit code is (3 - code) (A0<->T3, C1<->G2); RC also reverses
    base order.  Involution: p[p[i]] == i.
    """
    idx = np.arange(NHEX, dtype=np.int64)
    # decode each index to its 6 base codes (big-endian)
    codes = np.empty((NHEX, KMER), dtype=np.int64)
    rem = idx.copy()
    for k in range(KMER):
        codes[:, KMER - 1 - k] = rem % 4
        rem //= 4
    rc_codes = (3 - codes)[:, ::-1]
    return (rc_codes @ _POW).astype(np.int64)


RC_PERM = rc_hexamer_permutation()


def hexamer_indices(seq_bytes: np.ndarray):
    """Sliding 6-mer indices over an ASCII-uint8 sequence.

    Returns (fwd_idx, rc_idx, valid) each length ``len(seq)-5``.  ``fwd_idx[c]``
    is the forward hexamer starting at ``seq[c:c+6]``; ``rc_idx[c]`` its
    reverse-complement index; ``valid[c]`` is False if the window contains a
    non-ACGT base.  Invalid windows carry fwd/rc index 0 (callers must gate on
    ``valid``).
    """
    codes = _BASE_LUT[seq_bytes].astype(np.int64)
    win = np.lib.stride_tricks.sliding_window_view(codes, KMER)  # (L-5, 6)
    valid = ~(win == 255).any(axis=1)
    safe = np.where(win == 255, 0, win)
    fwd = safe @ _POW
    rc = ((3 - safe)[:, ::-1] @ _POW)
    return fwd.astype(np.int64), rc.astype(np.int64), valid


# ── ground-truth w6 ───────────────────────────────────────────────────────

def build_w6(seed: int, dynamic_range: float) -> np.ndarray:
    """4096 synthetic log-normal hexamer weights, normalised to max 1.

    ``dynamic_range`` is the TARGET favoured/disfavoured ratio, defined as the
    p95/p5 spread of the table: sigma = log(dynamic_range) / (2 * 1.645) so the
    5th..95th percentile of the log-normal spans a factor ``dynamic_range`` (the
    plan's "roughly 2-5x favoured vs disfavoured").  The realised p95/p5 and the
    full min/max range are reported (the 4096-draw extremes exceed p95/p5).
    """
    rng = np.random.default_rng(seed)
    sigma = np.log(dynamic_range) / (2 * 1.6448536269514722)  # z_0.95
    w = np.exp(rng.normal(0.0, sigma, size=NHEX))
    return w / w.max()


# ── 2-D GC x length bias surface ──────────────────────────────────────────

class GCBias2D:
    """Bilinear interpolation of the row-centred GC x length bias with HOLD-at-edges.

    The grid is ``bias_grid_2d_row_centred`` from gc_bias_grid.json (rows =
    length in ``lengths``, cols = GC% in ``gc_percents``).  It is ALREADY the
    bias (exp(+cf)); it is not inverted.  Extrapolation holds the edge rows/cols
    (achieved by clipping the query into the grid range before interpolation),
    exactly the recorded ``extrapolation_policy``.
    """

    def __init__(self, lengths, gc_percents, grid, extrapolation_policy=""):
        self.lengths = np.asarray(lengths, dtype=np.float64)
        self.gc_percents = np.asarray(gc_percents, dtype=np.float64)
        self.grid = np.asarray(grid, dtype=np.float64)
        self.extrapolation_policy = extrapolation_policy
        assert self.grid.shape == (len(self.lengths), len(self.gc_percents))

    @classmethod
    def from_json(cls, path: str) -> "GCBias2D":
        d = json.load(open(path))
        key = d.get("surface_for_simulation", "bias_grid_2d_row_centred")
        return cls(d["lengths"], d["gc_percents"], d[key],
                   d.get("extrapolation_policy", ""))

    @property
    def max_bias(self) -> float:
        # bilinear interpolation never exceeds a vertex value, and HOLD-at-edges
        # keeps out-of-range queries within the grid range, so the domain max is
        # the grid max.  Used as the acceptance normaliser.
        return float(self.grid.max())

    def __call__(self, length, gc_percent) -> np.ndarray:
        L = np.clip(np.asarray(length, dtype=np.float64), self.lengths[0], self.lengths[-1])
        G = np.clip(np.asarray(gc_percent, dtype=np.float64),
                    self.gc_percents[0], self.gc_percents[-1])
        li = np.clip(np.searchsorted(self.lengths, L, side="right") - 1,
                     0, len(self.lengths) - 2)
        gj = np.clip(np.searchsorted(self.gc_percents, G, side="right") - 1,
                     0, len(self.gc_percents) - 2)
        L0 = self.lengths[li]; L1 = self.lengths[li + 1]
        G0 = self.gc_percents[gj]; G1 = self.gc_percents[gj + 1]
        tl = (L - L0) / (L1 - L0)
        tg = (G - G0) / (G1 - G0)
        b = self.grid
        v00 = b[li, gj]; v01 = b[li, gj + 1]
        v10 = b[li + 1, gj]; v11 = b[li + 1, gj + 1]
        top = v00 * (1 - tg) + v01 * tg
        bot = v10 * (1 - tg) + v11 * tg
        return top * (1 - tl) + bot * tl


# ── region set + per-region precompute ────────────────────────────────────

def load_regions(bed: str, n_regions: int, region_len: int, ref: str = "hg38"):
    """Carve ``n_regions`` central ``region_len``-bp windows from the training tiles.

    Uses the library (RegionDataFrame.from_bed) per CLAUDE.md rather than parsing
    the BED by hand -- the same loader background_model/preprocess.build_tiles
    uses, so region coordinates are identical to the plumbing path.  Central
    windows avoid the 16,384bp tile edges.  Tiles whose central window carries
    >1% N are skipped (a hexamer over N has no defined w6 weight); scanning
    continues until ``n_regions`` clean windows are found.
    """
    import pysam
    from fragmentomics_tools.dataframe import RegionDataFrame

    rdf = RegionDataFrame.from_bed(bed, ref=ref)
    fa = pysam.FastaFile(FASTA)
    off = (16_384 - region_len) // 2
    regions = []
    for row in rdf.itertuples():
        if len(regions) >= n_regions:
            break
        contig = str(row.contig)
        gstart = int(row.start) + off
        gstop = gstart + region_len
        seq = fa.fetch(contig, gstart, gstop).upper()
        if seq.count("N") / region_len > 0.01:
            continue
        regions.append({"contig": contig, "gstart": gstart, "gstop": gstop})
    fa.close()
    if len(regions) < n_regions:
        raise RuntimeError(
            f"only {len(regions)} clean regions found in {bed} (need {n_regions})"
        )
    return regions


def precompute_region(region: dict, region_len: int):
    """Per-region hexamer-index and cumulative-GC arrays (independent of w6).

    Fetches the region sequence padded by HEX_HALF on each side so hexamer
    windows for cuts at region-local 0..region_len resolve.  Returns:
      fwd_cut  (region_len+1,)  forward hexamer index for a cut at region-local c
      rc_cut   (region_len+1,)  reverse-complement hexamer index for that cut
      valid    (region_len+1,)  window all-ACGT
      cum_gc   (region_len+1,)  cumulative (#G+#C) over region-local [0, x)
    A fragment (start p, stop q) uses the left cut at c=p (forward hexamer) and
    the right/far cut at c=q (reverse-complement hexamer).
    """
    import pysam
    fa = pysam.FastaFile(FASTA)
    seq = fa.fetch(region["contig"], region["gstart"] - HEX_HALF,
                   region["gstop"] + HEX_HALF).upper()
    fa.close()
    seq_bytes = np.frombuffer(seq.encode("ascii"), dtype=np.uint8)
    # seq_bytes[i] is region-local position (i - HEX_HALF).  A cut at region-local
    # c spans bases [c-3, c+3) = seq_bytes[c : c+6], so fwd_cut[c] = window at c.
    fwd_cut, rc_cut, valid = hexamer_indices(seq_bytes)  # length region_len+1
    assert len(fwd_cut) == region_len + 1, (len(fwd_cut), region_len)
    core = seq_bytes[HEX_HALF:HEX_HALF + region_len]
    is_gc = (core == ord("G")) | (core == ord("C"))
    cum_gc = np.concatenate([[0], np.cumsum(is_gc)]).astype(np.int64)
    return {"fwd_cut": fwd_cut, "rc_cut": rc_cut, "valid": valid, "cum_gc": cum_gc}


# ── empirical inputs from the real h5 ─────────────────────────────────────

def empirical_length_pmf(h5_path: str):
    """Empirical fragment-length pmf over [1, MAX_LEN] from fragment_length_counts."""
    from fragments_h5 import FragmentsH5
    h5 = FragmentsH5(h5_path, cache_pointers=False)
    flc = np.asarray(h5.fragment_length_counts, dtype=np.float64)
    h5.close()
    vals = np.arange(1, MAX_LEN + 1)
    p = flc[1:MAX_LEN + 1].copy()
    p = p / p.sum()
    return vals, p


def per_region_real_counts(h5_path: str, regions: list, min_mapq: int = 10):
    """Real deduped fragment count per region (the empirical per-tile distribution)."""
    from fragments_h5 import FragmentsH5
    from fragmentomics_tools.fragment_array.fragment_array import RegionFragmentArray
    from fragmentomics_tools.region import Region

    h5 = FragmentsH5(h5_path, cache_pointers=False)
    counts = np.zeros(len(regions), dtype=np.int64)
    for i, r in enumerate(regions):
        region = Region(chrom=r["contig"], start=r["gstart"], stop=r["gstop"], strand=".")
        rfa = RegionFragmentArray.from_fragments_h5(
            h5, region, min_mapq=min_mapq, max_frag_len=MAX_LEN
        ).drop_duplicate_fragments()
        counts[i] = int(rfa.n_fragments)
    h5.close()
    return counts


# ── the sampler ───────────────────────────────────────────────────────────

def simulate_sample(region_pre, target_counts, w6, gcbias, len_vals, len_p,
                    region_len, rng):
    """Rejection-sample fragments for one sample across all regions.

    Acceptance prob = w6(left) * w6(right) * gc_bias(length, gc) / normaliser,
    normaliser = max(w6)**2 * gcbias.max_bias so the max achievable weight is 1.
    Strand is assigned 50/50 independently (the acceptance product is symmetric
    in the two ends, so strand carries no bias -- see the module docstring's
    strand-symmetry note).
    """
    norm = (w6.max() ** 2) * gcbias.max_bias
    all_ridx, all_start, all_stop, all_strand = [], [], [], []
    for ridx, (pre, target) in enumerate(zip(region_pre, target_counts)):
        if target <= 0:
            continue
        fwd_cut = pre["fwd_cut"]; rc_cut = pre["rc_cut"]
        valid = pre["valid"]; cum_gc = pre["cum_gc"]
        starts_acc = []
        stops_acc = []
        n_have = 0
        # oversample per batch; grow batch if acceptance is low
        batch = max(256, int(target * 2))
        while n_have < target:
            p = rng.integers(0, region_len, size=batch)          # region-local start
            length = rng.choice(len_vals, size=batch, p=len_p)
            q = p + length                                        # region-local stop
            fit = q <= region_len
            if not fit.any():
                continue
            p_f = p[fit]; q_f = q[fit]; len_f = length[fit]
            ok = valid[p_f] & valid[q_f]
            if not ok.any():
                continue
            p_f = p_f[ok]; q_f = q_f[ok]; len_f = len_f[ok]
            lw = w6[fwd_cut[p_f]]                 # left cut, forward hexamer
            rw = w6[rc_cut[q_f]]                  # right/far cut, reverse-complement
            gc_pct = 100.0 * (cum_gc[q_f] - cum_gc[p_f]) / len_f
            gb = gcbias(len_f, gc_pct)
            prob = (lw * rw * gb) / norm
            acc = rng.random(size=len(prob)) < prob
            if acc.any():
                starts_acc.append(p_f[acc])
                stops_acc.append(q_f[acc])
                n_have += int(acc.sum())
        s = np.concatenate(starts_acc)[:target]
        e = np.concatenate(stops_acc)[:target]
        strand = np.where(rng.random(size=target) < 0.5, "+", "-")
        all_ridx.append(np.full(target, ridx, dtype=np.int32))
        all_start.append(s.astype(np.int32))
        all_stop.append(e.astype(np.int32))
        all_strand.append(strand)
    return (np.concatenate(all_ridx), np.concatenate(all_start),
            np.concatenate(all_stop), np.concatenate(all_strand).astype("U1"))


# ── driver ────────────────────────────────────────────────────────────────

def run(regime, n_samples, seed, w6_dynamic_range, n_regions, out_root,
        heldout_h5, region_len=REGION_LEN, real_store=None):
    t0 = time.time()
    rng = np.random.default_rng(seed)

    print(f"[sim] regime={regime} n_samples={n_samples} seed={seed} "
          f"n_regions={n_regions} region_len={region_len}", flush=True)

    regions = load_regions(TRAINING_TILES, n_regions, region_len)
    print(f"[sim] loaded {len(regions)} regions ({time.time()-t0:.1f}s)", flush=True)

    gcbias = GCBias2D.from_json(GC_BIAS_JSON)
    w6 = build_w6(seed, w6_dynamic_range)
    print(f"[sim] w6 realised range {w6.min():.4f}..{w6.max():.4f} "
          f"(favoured/disfavoured p95/p5 = "
          f"{np.percentile(w6,95)/np.percentile(w6,5):.2f}x)", flush=True)

    len_vals, len_p = empirical_length_pmf(heldout_h5)
    real_counts = per_region_real_counts(heldout_h5, regions)

    # Build the pool of per-tile fragment counts to sample from.
    # --real-store: use the production store's empirical N distribution
    #   (totals/N has shape (samples, tiles, tracks); sum tracks, flatten).
    # Otherwise: fall back to the heldout-h5 per-region counts.
    if real_store is not None:
        import zarr
        store = zarr.open(real_store, mode="r")
        N = np.asarray(store["totals/N"])       # (samples, tiles, tracks)
        count_pool = N.sum(axis=2).ravel()       # total frags per (sample, tile)
        count_pool = count_pool[count_pool > 0]  # drop empty tiles
        print(f"[sim] real-store N pool: {len(count_pool)} entries, "
              f"median={np.median(count_pool):.0f} ({time.time()-t0:.1f}s)",
              flush=True)
    else:
        count_pool = real_counts

    # Each (sample, region) gets an independently sampled target count.
    target_counts = rng.choice(count_pool, size=(n_samples, n_regions),
                               replace=True)
    print(f"[sim] real per-tile counts median={np.median(count_pool):.0f} "
          f"targets total={int(target_counts.sum()):,} ({time.time()-t0:.1f}s)",
          flush=True)

    region_pre = [precompute_region(r, region_len) for r in regions]
    print(f"[sim] precomputed region arrays ({time.time()-t0:.1f}s)", flush=True)

    # regime-B per-sample jitter of w6 (recorded as ground truth)
    if regime == "B":
        jitter_rng = np.random.default_rng(seed + 1)
        log_jitter = jitter_rng.normal(0.0, JITTER_SD, size=(n_samples, NHEX))
    else:
        log_jitter = np.zeros((n_samples, NHEX))

    out_dir = os.path.join(out_root, regime)
    os.makedirs(out_dir, exist_ok=True)
    plots_dir = os.path.join(out_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    per_sample_seeds = rng.integers(0, 2 ** 31 - 1, size=n_samples)
    total_frags = 0
    for s in range(n_samples):
        w6_s = w6 * np.exp(log_jitter[s])
        srng = np.random.default_rng(int(per_sample_seeds[s]))
        ridx, start, stop, strand = simulate_sample(
            region_pre, target_counts[s], w6_s, gcbias, len_vals, len_p,
            region_len, srng
        )
        np.savez(os.path.join(out_dir, f"sample_{s:03d}.npz"),
                 region_idx=ridx, start=start, stop=stop, strand=strand)
        total_frags += len(ridx)
        print(f"[sim]   sample {s:03d}: {len(ridx):,} fragments "
              f"({time.time()-t0:.1f}s)", flush=True)

    # region table (shared)
    np.savez(os.path.join(out_dir, "region_table.npz"),
             contig=np.array([r["contig"] for r in regions], dtype="U6"),
             gstart=np.array([r["gstart"] for r in regions], dtype=np.int64),
             gstop=np.array([r["gstop"] for r in regions], dtype=np.int64),
             region_len=np.int64(region_len))

    # ground truth
    np.savez(os.path.join(out_dir, "ground_truth.npz"),
             w6=w6, log_jitter=log_jitter, target_counts=target_counts,
             real_counts=real_counts, len_vals=len_vals, len_p=len_p,
             bias_grid=gcbias.grid, bias_lengths=gcbias.lengths,
             bias_gc_percents=gcbias.gc_percents, rc_perm=RC_PERM)
    gt_json = dict(
        regime=regime, n_samples=n_samples, seed=seed,
        w6_dynamic_range=w6_dynamic_range,
        w6_realised_min=float(w6.min()), w6_realised_max=float(w6.max()),
        w6_p95_over_p5=float(np.percentile(w6, 95) / np.percentile(w6, 5)),
        n_regions=n_regions, region_len=region_len, hex_half=HEX_HALF,
        max_len=MAX_LEN, jitter_sd=(JITTER_SD if regime == "B" else 0.0),
        heldout_h5=heldout_h5, fasta=FASTA, gc_bias_json=GC_BIAS_JSON,
        training_tiles=TRAINING_TILES,
        surface_for_simulation="bias_grid_2d_row_centred",
        extrapolation_policy=gcbias.extrapolation_policy,
        acceptance_normaliser=float((w6.max() ** 2) * gcbias.max_bias),
        total_fragments=int(total_frags),
        runtime_s=round(time.time() - t0, 1),
    )
    json.dump(gt_json, open(os.path.join(out_dir, "ground_truth.json"), "w"),
              indent=2)
    print(f"[sim] wrote {out_dir}  total {total_frags:,} fragments "
          f"in {time.time()-t0:.1f}s", flush=True)
    return out_dir


# ── validation gate (plan sec 6 step 4) ───────────────────────────────────

def _load_all_fragments(out_dir):
    rt = np.load(os.path.join(out_dir, "region_table.npz"), allow_pickle=True)
    region_len = int(rt["region_len"])
    regions = [{"contig": str(c), "gstart": int(a), "gstop": int(b)}
               for c, a, b in zip(rt["contig"], rt["gstart"], rt["gstop"])]
    frames = sorted(f for f in os.listdir(out_dir)
                    if f.startswith("sample_") and f.endswith(".npz"))
    R, S, E, ST = [], [], [], []
    for fn in frames:
        d = np.load(os.path.join(out_dir, fn))
        R.append(d["region_idx"]); S.append(d["start"])
        E.append(d["stop"]); ST.append(d["strand"])
    return (regions, region_len,
            np.concatenate(R), np.concatenate(S),
            np.concatenate(E), np.concatenate(ST))


def validate(out_dir):
    """Run the four validation gates.  Returns a dict of numbers + writes plots."""
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig_sim")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import pearsonr, spearmanr

    gt = np.load(os.path.join(out_dir, "ground_truth.npz"), allow_pickle=True)
    w6 = gt["w6"]
    rc_perm = gt["rc_perm"]
    gcbias = GCBias2D(gt["bias_lengths"], gt["bias_gc_percents"], gt["bias_grid"])
    len_vals = gt["len_vals"]; len_p = gt["len_p"]

    regions, region_len, R, S, E, ST = _load_all_fragments(out_dir)
    region_pre = [precompute_region(r, region_len) for r in regions]
    plots_dir = os.path.join(out_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    res = {"n_fragments": int(len(R))}

    # per-fragment quantities via region-local lookups
    fwd_start = np.empty(len(R), dtype=np.int64)   # forward hexamer at left cut (start)
    rc_stop = np.empty(len(R), dtype=np.int64)     # RC hexamer at right/far cut (stop)
    fwd_stop = np.empty(len(R), dtype=np.int64)    # forward *reference* hexamer at stop
    gc_pct = np.empty(len(R), dtype=np.float64)
    length = (E - S).astype(np.int64)
    for ridx, pre in enumerate(region_pre):
        m = R == ridx
        if not m.any():
            continue
        p = S[m]; q = E[m]
        fwd_start[m] = pre["fwd_cut"][p]
        rc_stop[m] = pre["rc_cut"][q]
        fwd_stop[m] = pre["fwd_cut"][q]
        gc_pct[m] = 100.0 * (pre["cum_gc"][q] - pre["cum_gc"][p]) / (q - p)

    # background hexamer composition (uniform over candidate cut positions)
    bg_fwd = np.zeros(NHEX); bg_rc = np.zeros(NHEX)
    for pre in region_pre:
        v = pre["valid"]
        np.add.at(bg_fwd, pre["fwd_cut"][v], 1.0)
        np.add.at(bg_rc, pre["rc_cut"][v], 1.0)
    bg_fwd /= bg_fwd.sum(); bg_rc /= bg_rc.sum()

    # ── Gate 1: per-hexamer endpoint frequency vs true w6 ──────────────────
    obs_left = np.bincount(fwd_start, minlength=NHEX).astype(np.float64)
    obs_left /= obs_left.sum()
    obs_right = np.bincount(rc_stop, minlength=NHEX).astype(np.float64)
    obs_right /= obs_right.sum()
    # enrichment = observed / background  ~  w6  (up to a constant)
    enr_left = obs_left / np.where(bg_fwd > 0, bg_fwd, np.nan)
    enr_right = obs_right / np.where(bg_rc > 0, bg_rc, np.nan)
    good = np.isfinite(enr_left) & np.isfinite(enr_right) & (bg_fwd > 0) & (bg_rc > 0)
    pear_l = pearsonr(enr_left[good], w6[good])[0]
    spear_l = spearmanr(enr_left[good], w6[good])[0]
    pear_r = pearsonr(enr_right[good], w6[good])[0]
    spear_r = spearmanr(enr_right[good], w6[good])[0]
    res["gate1_left_pearson"] = float(pear_l)
    res["gate1_left_spearman"] = float(spear_l)
    res["gate1_right_pearson"] = float(pear_r)
    res["gate1_right_spearman"] = float(spear_r)

    fig, ax = plt.subplots(1, 2, figsize=(11, 5))
    for a, enr, pe, sp, ttl in [
        (ax[0], enr_left, pear_l, spear_l, "left cut (forward hexamer)"),
        (ax[1], enr_right, pear_r, spear_r, "right cut (reverse-complement)")]:
        a.scatter(w6[good], enr[good], s=3, alpha=0.25)
        a.set_xlabel("true w6"); a.set_ylabel("observed endpoint enrichment")
        a.set_title(f"{ttl}\nPearson={pe:.3f} Spearman={sp:.3f}")
    fig.suptitle("Gate 1: recovered hexamer cut-site bias vs ground truth")
    fig.tight_layout(); fig.savefig(os.path.join(plots_dir, "gate1_hexamer.png"), dpi=110)
    plt.close(fig)

    # ── Gate 2: realised GC bias PER LENGTH STRATUM vs true surface ────────
    grid_lengths = gcbias.lengths
    # assign each fragment to the nearest grid-length stratum
    edges = np.concatenate([[0],
                            (grid_lengths[:-1] + grid_lengths[1:]) / 2,
                            [np.inf]])
    strat = np.clip(np.searchsorted(edges, length, side="right") - 1,
                    0, len(grid_lengths) - 1)
    # background (length, gc) from a large no-acceptance proposal pool
    bg_len, bg_gc = _background_len_gc(region_pre, len_vals, len_p, region_len,
                                       n_per_region=400,
                                       rng=np.random.default_rng(12345))
    bg_strat = np.clip(np.searchsorted(edges, bg_len, side="right") - 1,
                       0, len(grid_lengths) - 1)
    gc_lo, gc_hi = (27.5, 32.5), (67.5, 72.5)   # around GC 30 and 70
    true_slopes = gcbias.grid[:, -1] / gcbias.grid[:, 0]
    realised_slopes = []
    for k in range(len(grid_lengths)):
        realised_slopes.append(_gc_slope(gc_pct[strat == k], bg_gc[bg_strat == k],
                                         gc_lo, gc_hi))
    res["gate2_true_gc_slopes"] = [float(x) for x in true_slopes]
    res["gate2_realised_gc_slopes"] = [float(x) for x in realised_slopes]
    res["gate2_grid_lengths"] = [int(x) for x in grid_lengths]

    # realised surface (enrichment on a length x gc grid) vs true
    realised_surface = _realised_surface(strat, gc_pct, bg_strat, bg_gc,
                                         grid_lengths, gcbias.gc_percents)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
    im0 = ax[0].imshow(gcbias.grid, aspect="auto", origin="lower", cmap="viridis")
    ax[0].set_title("true bias surface"); _label_axes(ax[0], gcbias)
    fig.colorbar(im0, ax=ax[0])
    im1 = ax[1].imshow(realised_surface, aspect="auto", origin="lower", cmap="viridis")
    ax[1].set_title("realised bias surface"); _label_axes(ax[1], gcbias)
    fig.colorbar(im1, ax=ax[1])
    ax[2].plot(grid_lengths, true_slopes, "o-", label="true")
    ax[2].plot(grid_lengths, realised_slopes, "s--", label="realised")
    ax[2].axhline(1.0, color="grey", lw=0.7)
    ax[2].set_xlabel("fragment length (bp)")
    ax[2].set_ylabel("GC slope (70% / 30%)")
    ax[2].set_title("GC slope per length stratum\n(sign flip: short<1<long)")
    ax[2].legend()
    fig.suptitle("Gate 2: realised GC x length interaction vs ground truth")
    fig.tight_layout(); fig.savefig(os.path.join(plots_dir, "gate2_gc_surface.png"), dpi=110)
    plt.close(fig)

    # ── Gate 3: realised length distribution vs empirical input ────────────
    obs_len_hist = np.bincount(length, minlength=MAX_LEN + 1)[1:MAX_LEN + 1].astype(np.float64)
    obs_len_p = obs_len_hist / obs_len_hist.sum()
    # total-variation distance + mean shift
    tvd = 0.5 * np.abs(obs_len_p - len_p).sum()
    mean_in = float((len_vals * len_p).sum())
    mean_out = float((len_vals * obs_len_p).sum())
    res["gate3_length_tvd"] = float(tvd)
    res["gate3_input_mean_len"] = mean_in
    res["gate3_realised_mean_len"] = mean_out
    res["gate3_mean_len_shift"] = mean_out - mean_in
    # long-tail depletion (edge rejection hits the longest fragments)
    long_in = float(len_p[len_vals >= 150].sum())
    long_out = float(obs_len_p[len_vals >= 150].sum())
    res["gate3_frac_ge150_input"] = long_in
    res["gate3_frac_ge150_realised"] = long_out
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(len_vals, len_p, label="empirical input", lw=1)
    ax.plot(len_vals, obs_len_p, label="realised (accepted)", lw=1, alpha=0.8)
    ax.set_xlabel("fragment length (bp)"); ax.set_ylabel("density")
    ax.set_title(f"Gate 3: length distortion  TVD={tvd:.4f}  "
                 f"mean {mean_in:.1f}->{mean_out:.1f}bp")
    ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "gate3_length.png"), dpi=110)
    plt.close(fig)

    # ── Gate 4: strand symmetry (RC image) ─────────────────────────────────
    is_plus = ST == "+"
    plus_ref = np.bincount(fwd_start[is_plus], minlength=NHEX).astype(np.float64)
    minus_ref = np.bincount(fwd_stop[~is_plus], minlength=NHEX).astype(np.float64)
    plus_ref /= plus_ref.sum(); minus_ref /= minus_ref.sum()
    # RC-image expectation: freq_plus(H) ~ freq_minus(RC(H))
    r_rc = pearsonr(plus_ref, minus_ref[rc_perm])[0]
    r_id = pearsonr(plus_ref, minus_ref)[0]
    res["gate4_rc_image_pearson"] = float(r_rc)
    res["gate4_identity_pearson"] = float(r_id)
    fig, ax = plt.subplots(1, 2, figsize=(11, 5))
    ax[0].scatter(plus_ref, minus_ref[rc_perm], s=3, alpha=0.25)
    ax[0].set_xlabel("+ strand 5' ref-hexamer freq")
    ax[0].set_ylabel("- strand 5' ref-hexamer freq [RC-indexed]")
    ax[0].set_title(f"RC image: Pearson={r_rc:.3f}")
    ax[1].scatter(plus_ref, minus_ref, s=3, alpha=0.25)
    ax[1].set_xlabel("+ strand 5' ref-hexamer freq")
    ax[1].set_ylabel("- strand 5' ref-hexamer freq [same index]")
    ax[1].set_title(f"identity (should be lower): Pearson={r_id:.3f}")
    fig.suptitle("Gate 4: + / - endpoint hexamer preferences are RC images")
    fig.tight_layout(); fig.savefig(os.path.join(plots_dir, "gate4_strand.png"), dpi=110)
    plt.close(fig)

    json.dump(res, open(os.path.join(out_dir, "validation.json"), "w"), indent=2)

    # copy plots to the gitignored editor-viewable dir
    os.makedirs(SIM_PLOTS, exist_ok=True)
    regime = os.path.basename(os.path.normpath(out_dir))
    for fn in os.listdir(plots_dir):
        shutil.copy(os.path.join(plots_dir, fn),
                    os.path.join(SIM_PLOTS, f"{regime}_{fn}"))
    return res


def _background_len_gc(region_pre, len_vals, len_p, region_len, n_per_region, rng):
    """Draw (length, gc) for proposals WITHOUT acceptance -- the null (length,gc) density."""
    lens, gcs = [], []
    for pre in region_pre:
        p = rng.integers(0, region_len, size=n_per_region)
        length = rng.choice(len_vals, size=n_per_region, p=len_p)
        q = p + length
        fit = (q <= region_len) & pre["valid"][p] & pre["valid"][np.minimum(q, region_len)]
        p = p[fit]; q = q[fit]; length = length[fit]
        gc = 100.0 * (pre["cum_gc"][q] - pre["cum_gc"][p]) / length
        lens.append(length); gcs.append(gc)
    return np.concatenate(lens), np.concatenate(gcs)


def _gc_slope(obs_gc, bg_gc, gc_lo, gc_hi):
    """Realised GC slope = enrichment(high GC) / enrichment(low GC) in a stratum."""
    def enr(lo, hi):
        o = np.mean((obs_gc >= lo) & (obs_gc < hi))
        b = np.mean((bg_gc >= lo) & (bg_gc < hi))
        return o / b if b > 0 else np.nan
    return enr(*gc_hi) / enr(*gc_lo)


def _realised_surface(strat, obs_gc, bg_strat, bg_gc, grid_lengths, grid_gc):
    """Realised enrichment surface aligned to the true grid (length x GC)."""
    gc_edges = np.concatenate([[-np.inf],
                               (grid_gc[:-1] + grid_gc[1:]) / 2, [np.inf]])
    surf = np.full((len(grid_lengths), len(grid_gc)), np.nan)
    for i in range(len(grid_lengths)):
        o = obs_gc[strat == i]; b = bg_gc[bg_strat == i]
        for j in range(len(grid_gc)):
            lo, hi = gc_edges[j], gc_edges[j + 1]
            oc = np.mean((o >= lo) & (o < hi)) if len(o) else np.nan
            bc = np.mean((b >= lo) & (b < hi)) if len(b) else np.nan
            surf[i, j] = oc / bc if (bc and bc > 0) else np.nan
        # row-centre to match the row-centred true grid
        row = surf[i]
        if np.isfinite(row).any():
            surf[i] = row / np.nanmean(row)
    return surf


def _label_axes(ax, gcbias):
    ax.set_xticks(range(len(gcbias.gc_percents)))
    ax.set_xticklabels([int(g) for g in gcbias.gc_percents])
    ax.set_yticks(range(len(gcbias.lengths)))
    ax.set_yticklabels([int(l) for l in gcbias.lengths])
    ax.set_xlabel("GC %"); ax.set_ylabel("length (bp)")


# ── CLI ────────────────────────────────────────────────────────────────────

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--regime", choices=["A", "B"], default="A")
    ap.add_argument("--n-samples", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--w6-dynamic-range", type=float, default=4.0)
    ap.add_argument("--n-regions", type=int, default=N_REGIONS)
    ap.add_argument("--out-root", default=OUT_ROOT)
    ap.add_argument("--heldout-h5", default=DEFAULT_HELDOUT_H5)
    ap.add_argument("--real-store", default=None,
                    help="path to production zarr store; sample target counts "
                         "from its totals/N distribution instead of the "
                         "heldout h5 (recommended: bg_store_b67d7c95.zarr)")
    ap.add_argument("--validate", action="store_true",
                    help="run the four validation gates after simulating")
    ap.add_argument("--validate-only", metavar="DIR",
                    help="skip simulation; run validation on an existing regime dir")
    args = ap.parse_args(argv)

    if args.validate_only:
        res = validate(args.validate_only)
        print(json.dumps(res, indent=2))
        return

    out_dir = run(args.regime, args.n_samples, args.seed, args.w6_dynamic_range,
                  args.n_regions, args.out_root, args.heldout_h5,
                  real_store=args.real_store)
    if args.validate:
        res = validate(out_dir)
        print(json.dumps(res, indent=2))


if __name__ == "__main__":
    sys.exit(main())
