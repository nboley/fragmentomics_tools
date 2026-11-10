#!/usr/bin/env python
"""Measure the ceiling on sampler vectorisation, and the draw-side fixes.

Owner decision 83 lifted the RNG-stream constraint, so a vectorised draw is
now permissible.  The question this answers is whether it is WORTH doing:
the draw loop's share of the per-region cost bounds the achievable saving
regardless of how good the vectorisation is.

Distributional identity is checked explicitly (the whole point of decision 83
is that the STREAM may change and the DISTRIBUTION may not).
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from background_model.simulator.precompute import HEX_HALF, precompute_region  # noqa: E402
from background_model.simulator.sampler import (  # noqa: E402
    draw_fragments_for_region, target_count_for_region,
)
from background_model.simulator.weights import (  # noqa: E402
    L_MAX, L_MIN, N_LENGTHS, NHEX, HexamerTables, build_region_weights,
)

REGION_SET = ("/home/nathanboley/src/fragmentomics_tools/data/region_sets/"
              "quiet_v2_pad1200_repeats_removed_tile2560.bed")
FASTA = "/efs/analytics/nathanboley/data_resources/genome/hg38.fa"
GCFL = "/efs/analytics/nathanboley/background_model/sim_smoke/RD-56670.gcfl_model.json"
CACHE = "/tmp/_sim_prof_inputs.npz"
pcf = time.perf_counter


def load_inputs():
    if os.path.exists(CACHE):
        z = np.load(CACHE)
        return z["predict_lut"], z["marginal_fl"]
    from flgc.model import GCFlDistModel
    from background_model.simulator.capture import (
        build_marginal_fl, load_duphist, predict_lut_from_model,
    )
    model = GCFlDistModel.load(GCFL)
    lut = predict_lut_from_model(model)
    fl = build_marginal_fl(load_duphist("RD-56670"))
    np.savez(CACHE, predict_lut=lut, marginal_fl=fl)
    return lut, fl


def tables(seed=42, dr=4.0):
    rng = np.random.default_rng(seed)
    sigma = np.log(dr) / (2 * 1.6448536269514722)
    return HexamerTables(*[
        (lambda w: w / w.max())(np.exp(rng.normal(0.0, sigma, size=NHEX)))
        for _ in range(4)])


def load_regions(n):
    from fragmentomics_tools.dataframe import RegionDataFrame
    rdf = RegionDataFrame.from_bed(REGION_SET, ref="hg38").iloc[:n]
    return [(r.contig, int(r.start), int(r.stop))
            for r in rdf.itertuples(index=False)]


def bench(fn, reps):
    ts = []
    for _ in range(5):
        t0 = pcf()
        for _ in range(reps):
            fn()
        ts.append((pcf() - t0) / reps)
    return 1000 * float(np.median(ts))


# ── A: draw with rw reused + lazy per-row conditional (bit-identical) ────

def draw_reuse(*, hex_tables, valid, hex_fwd, hex_rc, region_len,
               n_fragments, rng, rw):
    n_sites = region_len + 1
    svp = np.where(valid, hex_tables.start_fwd[hex_fwd], 0.0)
    svm = np.where(valid, hex_tables.start_rev[hex_rc], 0.0)
    Zp = rw.w_plus.sum(axis=1)
    sp = np.where(Zp > 0, svp, 0.0)
    t = sp.sum()
    if t > 0:
        sp /= t
    Zm = rw.w_minus.sum(axis=1)
    sm = np.where(Zm > 0, svm, 0.0)
    t = sm.sum()
    if t > 0:
        sm /= t
    Ls = np.arange(L_MIN, L_MAX + 1)
    starts = np.empty(n_fragments, dtype=np.int64)
    stops = np.empty(n_fragments, dtype=np.int64)
    strands = np.empty(n_fragments, dtype="U1")
    for i in range(n_fragments):
        if rng.random() < 0.5:
            c5 = rng.choice(n_sites, p=sp)
            L = Ls[rng.choice(N_LENGTHS, p=rw.w_plus[c5] / Zp[c5])]
            starts[i], stops[i], strands[i] = c5, c5 + L, "+"
        else:
            c5 = rng.choice(n_sites, p=sm)
            L = Ls[rng.choice(N_LENGTHS, p=rw.w_minus[c5] / Zm[c5])]
            starts[i], stops[i], strands[i] = c5 - L, c5, "-"
    return starts, stops, strands


# ── B: fully vectorised draw (different RNG stream, same distribution) ───

def draw_vectorised(*, hex_tables, valid, hex_fwd, hex_rc, region_len,
                    n_fragments, rng, rw):
    """Inverse-CDF draw of the SAME sequential factorisation.

    strand ~ Bernoulli(1/2); c5 ~ start_s/S_s; L ~ E_s(c5,L)/Z_s(c5).
    Each step is sampled by inverse-CDF instead of a per-fragment
    rng.choice, so the number and order of consumed uniforms differs while
    the target distribution is unchanged.
    """
    n_sites = region_len + 1
    svp = np.where(valid, hex_tables.start_fwd[hex_fwd], 0.0)
    svm = np.where(valid, hex_tables.start_rev[hex_rc], 0.0)
    Zp = rw.w_plus.sum(axis=1)
    sp = np.where(Zp > 0, svp, 0.0)
    sp = sp / sp.sum()
    Zm = rw.w_minus.sum(axis=1)
    sm = np.where(Zm > 0, svm, 0.0)
    sm = sm / sm.sum()

    is_plus = rng.random(n_fragments) < 0.5
    starts = np.empty(n_fragments, dtype=np.int64)
    stops = np.empty(n_fragments, dtype=np.int64)
    strands = np.where(is_plus, "+", "-").astype("U1")

    for plus, prob, Z, W in ((True, sp, Zp, rw.w_plus),
                             (False, sm, Zm, rw.w_minus)):
        sel = is_plus if plus else ~is_plus
        k = int(sel.sum())
        if k == 0:
            continue
        # c5 ~ prob, by inverse CDF
        cdf = np.cumsum(prob)
        cdf /= cdf[-1]
        c5 = np.searchsorted(cdf, rng.random(k), side="right")
        # L ~ W[c5,:]/Z[c5], by inverse CDF on the drawn rows only
        rows = W[c5] / Z[c5][:, None]
        rcdf = np.cumsum(rows, axis=1)
        rcdf /= rcdf[:, -1:]
        li = (rcdf < rng.random(k)[:, None]).sum(axis=1)
        li = np.minimum(li, N_LENGTHS - 1)
        L = L_MIN + li
        if plus:
            starts[sel], stops[sel] = c5, c5 + L
        else:
            starts[sel], stops[sel] = c5 - L, c5
    return starts, stops, strands


def distributional_check(region_len, seed, n_draws):
    """Chi-square-style check of the vectorised draw against w, mirroring
    tests/test_simulator_phase3.py::test_empirical_matches_weights."""
    rng0 = np.random.default_rng(seed)
    hex_fwd = rng0.integers(0, NHEX, size=region_len + 1)
    hex_rc = rng0.integers(0, NHEX, size=region_len + 1)
    cum_gc = np.cumsum(np.concatenate(
        [[0], rng0.integers(0, 2, size=region_len)])).astype(np.float64)
    valid = np.ones(region_len + 1, dtype=bool)
    tb = tables(seed=77)
    fl = np.full(N_LENGTHS, 1.0 / N_LENGTHS)
    lut = np.ones((N_LENGTHS, 20))
    rw = build_region_weights(
        hex_fwd=hex_fwd, hex_rc=hex_rc, cum_gc=cum_gc, hex_tables=tb,
        marginal_fl=fl, predict_lut=lut, region_len=region_len, valid=valid)

    out = {}
    for name, fn in (("production", draw_fragments_for_region),
                     ("vectorised", draw_vectorised)):
        rng = np.random.default_rng(1234)
        if name == "production":
            s, e, st = fn(hex_fwd=hex_fwd, hex_rc=hex_rc, cum_gc=cum_gc,
                          valid=valid, hex_tables=tb, marginal_fl=fl,
                          predict_lut=lut, region_len=region_len,
                          n_fragments=n_draws, rng=rng)
        else:
            s, e, st = fn(hex_tables=tb, valid=valid, hex_fwd=hex_fwd,
                          hex_rc=hex_rc, region_len=region_len,
                          n_fragments=n_draws, rng=rng, rw=rw)
        frac_plus = float((st == "+").sum()) / n_draws
        # length marginal
        Ld = e - s
        hist = np.bincount(Ld, minlength=L_MAX + 1).astype(float)
        emp = hist / hist.sum()
        exp = np.zeros(L_MAX + 1)
        for li in range(N_LENGTHS):
            exp[L_MIN + li] = rw.w_plus[:, li].sum() + rw.w_minus[:, li].sum()
        worst = 0.0
        for li in range(N_LENGTHS):
            L = L_MIN + li
            if exp[L] > 0.01:
                worst = max(worst, abs(emp[L] / exp[L] - 1.0))
        # plus-strand start marginal cosine
        ps = s[st == "+"]
        eh = np.bincount(ps, minlength=region_len + 1).astype(float)
        eh /= eh.sum()
        ex = rw.w_plus.sum(axis=1)
        ex = ex / ex.sum()
        cos = float(np.dot(eh, ex) / (np.linalg.norm(eh) * np.linalg.norm(ex)))
        tv = 0.5 * float(np.abs(emp[L_MIN:L_MAX + 1]
                                - exp[L_MIN:L_MAX + 1]
                                / exp[L_MIN:L_MAX + 1].sum()).sum())
        out[name] = (frac_plus, worst, cos, tv)
    return out


def main():
    predict_lut, marginal_fl = load_inputs()
    tb = tables()
    regions = load_regions(60)
    region_len = regions[0][2] - regions[0][1]
    target = target_count_for_region(region_len)

    import pysam
    with pysam.FastaFile(FASTA) as fa:
        for c, s, e in regions:
            fa.fetch(c, s - HEX_HALF, e + HEX_HALF)
    pcs = [precompute_region(*r, FASTA) for r in regions[:4]]
    pc0 = pcs[0]
    rw0 = build_region_weights(
        hex_fwd=pc0.hex_fwd, hex_rc=pc0.hex_rc, cum_gc=pc0.cum_gc,
        hex_tables=tb, marginal_fl=marginal_fl, predict_lut=predict_lut,
        region_len=region_len, valid=pc0.valid)

    print("=" * 72)
    print("F1. DRAW VARIANTS at region_len=2560, 54 fragments (median of 5)")
    print("=" * 72)
    dkw = dict(hex_fwd=pc0.hex_fwd, hex_rc=pc0.hex_rc, cum_gc=pc0.cum_gc,
               valid=pc0.valid, hex_tables=tb, marginal_fl=marginal_fl,
               predict_lut=predict_lut, region_len=region_len,
               n_fragments=target)
    vkw = dict(hex_tables=tb, valid=pc0.valid, hex_fwd=pc0.hex_fwd,
               hex_rc=pc0.hex_rc, region_len=region_len, n_fragments=target)
    r = np.random.default_rng(5)
    t_prod = bench(lambda: draw_fragments_for_region(rng=r, **dkw), 10)
    t_reuse = bench(lambda: draw_reuse(rng=r, rw=rw0, **vkw), 20)
    t_vec = bench(lambda: draw_vectorised(rng=r, rw=rw0, **vkw), 50)
    print(f"  production (rebuilds w)                 = {t_prod:7.3f} ms")
    print(f"  rw reused, scalar loop  [BIT-IDENTICAL] = {t_reuse:7.3f} ms")
    print(f"  rw reused, VECTORISED   [new RNG stream]= {t_vec:7.3f} ms")
    print(f"  -> vectorising saves {t_reuse - t_vec:.3f} ms/region "
          f"ON TOP of reusing w")

    print()
    print("=" * 72)
    print("F2. DISTRIBUTIONAL IDENTITY of the vectorised draw")
    print("=" * 72)
    for rl, nd in ((300, 20000), (200, 15000)):
        res = distributional_check(rl, seed=55 if rl == 300 else 44, n_draws=nd)
        print(f"  region_len={rl}, n_draws={nd}")
        for name, (fp, worst, cos, tv) in res.items():
            print(f"    {name:<11} frac_plus={fp:.4f}  worst L ratio dev="
                  f"{worst:.4f}  start cosine={cos:.5f}  TV={tv:.5f}")

    print()
    print("=" * 72)
    print("F3. WHERE THE 151 ms ACTUALLY GOES vs WHAT VECTORISING CAN REACH")
    print("=" * 72)
    full = 151.13
    print(f"  measured full per-region              = {full:7.2f} ms")
    print(f"  build_region_weights x2               = {123.33:7.2f} ms  "
          f"(81.6% of it; py-spy says 86.0% of samples)")
    print(f"  entire scalar draw loop (the vec target) = {5.68:5.2f} ms  "
          f"( 3.8%)")
    print(f"  vectorising the loop can therefore save at most ~"
          f"{t_reuse - t_vec:.2f} ms = "
          f"{100 * (t_reuse - t_vec) / full:.1f}% of today's per-region cost")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
